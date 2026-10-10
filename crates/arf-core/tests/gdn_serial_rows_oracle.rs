//! CPU oracle for the L313 SERIAL-ROWS contract — the highest-risk code on the speculation path.
//!
//! What L313 did. The verify record used to push a k-row window through the stateful kernels with
//! one dispatch PER ROW and an encoder-wide barrier between them (~100 dispatches per window
//! across 48 GDN layers). L313 folded that loop INSIDE the kernel: `serial_rows` in the dims makes
//! each thread chain its own channel's ring through registers, with no barrier at all, writing the
//! checkpoint slots itself.
//!
//! Why it needs an oracle. That change is invisible three ways at once:
//!   * at b=1 every row index is 0, so a row-mapping bug cannot show;
//!   * on a busy box the timing says nothing (this box sat at load 16-171 throughout this work);
//!   * a wrong ring still produces FLUENT text — recurrent state is the history, so corruption
//!     reads as confident nonsense rather than a crash (L179/L180).
//! The only cheap guard is a model of the semantics that fails when they drift.
//!
//! These tests re-implement `ssm_conv1d`'s ring exactly as
//! crates/arf-gpu/src/shaders/metal/ssm_conv1d_msl.metal specifies it, and assert the invariants a
//! refactor must not break. This is an ORACLE: it proves the SEMANTICS are self-consistent and
//! pins the contract. It does not execute Metal — an on-device run is still the only proof the
//! kernel implements this (measured, L313).

//! SCOPE. These model `ssm_conv1d` (the conv ring). The OTHER stateful kernel,
//! `gated_delta_net`, carries the SAME contract — verified by reading it: the identical
//! `serial_rows` fold (`iters = serial_rows > 0 ? serial_rows : 1`), the identical per-row `seq`
//! selection, and the identical checkpoint guard `r + 1u < d.serial_rows`
//! (gated_delta_net_msl.metal:101-159). So the invariants asserted here — read-before-write,
//! per-row `s_copy`, every-row-but-the-last checkpointed, cross-bank isolation — are the
//! contract for both. Its delta-rule matrix update is NOT modelled; only the ring is.
//!
//! MUTATION-TESTED. These tests were validated by deliberately breaking the model seven ways and
//! requiring each break to fail a test: write-before-read, a hoisted `s_copy` lookup, checkpointing
//! the last row, a reversed shift, an off-by-one kernel tap, an inverted SiLU, and a state write
//! placed before the dot. Six died. The seventh is EQUIVALENT, not a gap: the dot reads the
//! `window[]` registers, never `conv_state` (ssm_conv1d_msl.metal step 2), so a state write before
//! it cannot change the output in the kernel either. Two earlier versions of this file let real
//! mutants through — the fix was to return the assembled window and to model the dot + SiLU, so
//! ordering and output are observable instead of implied.

const SPEC_CKPT_ROWS: usize = 3;

/// One GDN layer's conv ring, modelled per the kernel:
///   `conv_state` is [n_banks, conv_dim * (d_conv - 1)], indexed
///   `bank * conv_dim * cs + channel * cs + i`.
struct ConvRing {
    conv_dim: usize,
    cs: usize, // d_conv - 1 cached columns
    state: Vec<f32>,
    ckpt: Vec<f32>,   // [SPEC_CKPT_ROWS][conv_dim * cs]
    kernel: Vec<f32>, // [conv_dim][d_conv], static — shared by every sequence
    last_out: f32,    // the SiLU'd convolution output of the most recent push_row
}

impl ConvRing {
    fn new(conv_dim: usize, d_conv: usize, banks: usize) -> Self {
        let cs = d_conv - 1;
        Self {
            conv_dim,
            cs,
            // Distinct per-bank values: a cross-bank bug must change the numbers, not just indices.
            state: (0..banks * conv_dim * cs)
                .map(|i| i as f32 * 0.25)
                .collect(),
            ckpt: vec![f32::NAN; SPEC_CKPT_ROWS * conv_dim * cs],
            kernel: (0..conv_dim * d_conv)
                .map(|i| 0.1 + 0.05 * i as f32)
                .collect(),
            last_out: f32::NAN,
        }
    }

    fn sbase(&self, bank: usize, ch: usize) -> usize {
        bank * self.conv_dim * self.cs + ch * self.cs
    }

    /// One row of the serial-rows loop for ONE channel, mirroring the kernel step for step:
    /// read the old ring into registers, dot, then shift-and-append, checkpointing when
    /// `r + 1 < serial_rows` (the LAST row is deliberately not checkpointed).
    ///
    /// Returns the assembled causal window (oldest..newest, with x_new last) so a test can
    /// assert the READ happened against the pre-write ring. Returning it is what makes the
    /// ordering observable: a model that clobbers first still agrees with itself, so comparing
    /// two runs of the same model cannot detect it (a write-before-read mutant survived exactly
    /// that check while this returned nothing).
    fn push_row(
        &mut self,
        r: usize,
        serial_rows: usize,
        s_copy: &[u32],
        ch: usize,
        x_new: f32,
    ) -> Vec<f32> {
        let bank = s_copy[r] as usize;
        let sbase = self.sbase(bank, ch);
        let mut window: Vec<f32> = (0..self.cs).map(|i| self.state[sbase + i]).collect();
        window.push(x_new);
        // (2) the d_conv-tap dot with this channel's kernel, then (3) the SiLU epilogue —
        // modelled because the ring state alone cannot see a write that lands between the
        // window read and the dot (a mutant doing exactly that survived until this existed).
        let sumf: f32 = (0..self.cs + 1)
            .map(|i| window[i] * self.kernel[ch * (self.cs + 1) + i])
            .sum();
        let z = sumf.clamp(-30.0, 30.0);
        self.last_out = sumf / (1.0 + (-z).exp());
        for i in 0..self.cs {
            self.state[sbase + i] = window[i + 1];
            if serial_rows > 0 && r + 1 < serial_rows {
                self.ckpt[r * self.conv_dim * self.cs + ch * self.cs + i] = window[i + 1];
            }
        }
        window
    }
}

/// THE RING INVARIANT. After pushing values v0..v_{n-1}, the ring must hold the LAST `cs` of
/// them, oldest-first. This is what "shift left, append" means, and it is the property that
/// makes the convolution causal. A refactor that writes before reading, or appends at the wrong
/// end, breaks exactly this.
#[test]
fn ring_holds_the_last_cs_values_oldest_first() {
    let (conv_dim, d_conv) = (4usize, 4usize);
    let mut ring = ConvRing::new(conv_dim, d_conv, 2);
    let ch = 1usize;
    let bank = 0usize;
    // Zero the ring so the expected content is unambiguous.
    let sb = ring.sbase(bank, ch);
    for i in 0..ring.cs {
        ring.state[sb + i] = 0.0;
    }
    let pushed = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    for (r, &v) in pushed.iter().enumerate() {
        ring.push_row(r, 0, &[bank as u32; 8], ch, v);
    }
    let got: Vec<f32> = (0..ring.cs).map(|i| ring.state[sb + i]).collect();
    assert_eq!(
        got,
        vec![30.0, 40.0, 50.0],
        "ring must hold the last d_conv-1 values, oldest first"
    );
}

/// READ-BEFORE-WRITE, checked against an INDEPENDENT expectation rather than against another run
/// of the same model. The kernel comment calls this "the critical ordering": the causal window
/// must be read into registers BEFORE the ring is overwritten. A model that clobbers first still
/// agrees with itself, so a same-model A/B cannot see it (verified: that mutant survived the
/// serial-vs-looped test). The fix is to assert the actual convolution output.
#[test]
fn the_window_is_read_before_the_ring_is_written() {
    let (conv_dim, d_conv) = (1usize, 4usize);
    let mut ring = ConvRing::new(conv_dim, d_conv, 1);
    let (ch, bank) = (0usize, 0usize);
    let sb = ring.sbase(bank, ch);
    // A ring with distinct known contents: oldest..newest = 1, 2, 3.
    for (i, v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
        ring.state[sb + i] = *v;
    }
    // The window the kernel must assemble for x_new = 4 is [1,2,3,4] — the OLD ring plus the new
    // input. If the ring is written first, the window reads back the clobbered value instead.
    let x_new = 4.0f32;
    let window = ring.push_row(0, 0, &[bank as u32], ch, x_new);
    assert_eq!(
        window,
        vec![1.0, 2.0, 3.0, 4.0],
        "the window must be assembled from the OLD ring plus x_new — if any cached column reads \
         back as x_new, the ring was written before it was read"
    );
    assert_eq!(
        (0..ring.cs).map(|i| ring.state[sb + i]).collect::<Vec<_>>(),
        vec![2.0, 3.0, 4.0],
        "after the push the ring must be the old window shifted left with x_new appended"
    );
}

/// THE CONVOLUTION OUTPUT ITSELF, against a hand-computed value. The ring tests above check
/// the STATE; this checks the number the layer actually emits. Without it, a write landing
/// between the window read and the dot is invisible (verified: such a mutant survived every
/// state-only test). Computed here independently of the model's own loop.
#[test]
fn the_convolution_output_matches_a_hand_computed_dot() {
    let (conv_dim, d_conv) = (1usize, 4usize);
    let mut ring = ConvRing::new(conv_dim, d_conv, 1);
    let (ch, bank) = (0usize, 0usize);
    let sb = ring.sbase(bank, ch);
    for (i, v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
        ring.state[sb + i] = *v;
    }
    let x_new = 4.0f32;
    // kernel[0..4] = 0.10, 0.15, 0.20, 0.25 (from ConvRing::new's generator).
    let k: Vec<f32> = (0..d_conv).map(|i| 0.1 + 0.05 * i as f32).collect();
    let expect_sum = 1.0 * k[0] + 2.0 * k[1] + 3.0 * k[2] + 4.0 * k[3];
    let z: f32 = expect_sum.clamp(-30.0, 30.0);
    let expect_out = expect_sum / (1.0 + (-z).exp());

    ring.push_row(0, 0, &[bank as u32], ch, x_new);
    assert!(
        (ring.last_out - expect_out).abs() < 1e-6,
        "conv output {} != hand-computed {expect_out}",
        ring.last_out
    );
}

/// SERIAL ROWS MUST EQUAL ROW-AT-A-TIME. This is the whole claim of L313: folding the loop into
/// the kernel changes only WHERE the loop runs, never the result. Modelled by running the same
/// window both ways and requiring identical state.
#[test]
fn serial_rows_matches_dispatch_per_row() {
    let (conv_dim, d_conv, banks) = (8usize, 4usize, 4usize);
    let s_copy: Vec<u32> = vec![2, 2, 2, 2]; // one sequence, pinned bank 2 — NOT row 0
    let xs = [0.5f32, -1.25, 2.0, 0.75];

    for k in 1..=SPEC_CKPT_ROWS {
        let rows = k + 1;
        let mut fused = ConvRing::new(conv_dim, d_conv, banks);
        let mut looped = ConvRing::new(conv_dim, d_conv, banks);
        for ch in 0..conv_dim {
            for r in 0..rows {
                fused.push_row(r, rows, &s_copy, ch, xs[r] + ch as f32);
            }
            // "Dispatch per row" is the same call with serial_rows = 0 (one iteration each).
            for r in 0..rows {
                looped.push_row(r, 0, &s_copy, ch, xs[r] + ch as f32);
            }
        }
        assert_eq!(
            fused.state, looped.state,
            "k={k}: folding the row loop into the kernel changed the ring state"
        );
    }
}

/// THE CHECKPOINT WINDOW. Slots 0..rows-1 must be written and the LAST row must NOT be —
/// `r + 1 < serial_rows` in the kernel. A full accept relies on the bank already holding the last
/// row's state; writing that slot too would be harmless, but NOT writing an earlier one silently
/// restores garbage on a partial accept.
#[test]
fn every_row_but_the_last_is_checkpointed() {
    let (conv_dim, d_conv) = (4usize, 4usize);
    for k in 1..=SPEC_CKPT_ROWS {
        let rows = k + 1;
        let mut ring = ConvRing::new(conv_dim, d_conv, 4);
        let s_copy = vec![1u32; rows];
        for ch in 0..conv_dim {
            for r in 0..rows {
                ring.push_row(r, rows, &s_copy, ch, 1.0 + r as f32);
            }
        }
        for r in 0..rows - 1 {
            let base = r * conv_dim * ring.cs;
            assert!(
                ring.ckpt[base..base + conv_dim * ring.cs]
                    .iter()
                    .all(|v| !v.is_nan()),
                "k={k}: checkpoint slot {r} was never written"
            );
        }
        // The last row is not checkpointed. It only has a SLOT to check when rows-1 <
        // SPEC_CKPT_ROWS; at the maximum window (rows = SPEC_CKPT_ROWS+1) the writes exactly
        // fill the slots, which is why `gdn_prepare_verify` refuses anything larger.
        if rows - 1 < SPEC_CKPT_ROWS {
            let last = (rows - 1) * conv_dim * ring.cs;
            assert!(
                ring.ckpt[last..last + conv_dim * ring.cs]
                    .iter()
                    .all(|v| v.is_nan()),
                "k={k}: the last row must NOT be checkpointed (a full accept needs no restore)"
            );
        }
        assert!(
            rows - 1 <= SPEC_CKPT_ROWS,
            "k={k}: a window writing {} slots cannot fit {SPEC_CKPT_ROWS} —              gdn_prepare_verify must refuse it", rows - 1
        );
    }
}

/// CROSS-BANK ISOLATION — the L179/L180 amnesia guard. Pushing a window for one sequence must
/// leave every other bank byte-identical. This is the failure that reads as fluent garbage, so it
/// gets an explicit test rather than being implied by the others.
#[test]
fn one_sequences_window_never_touches_another_bank() {
    let (conv_dim, d_conv, banks) = (8usize, 4usize, 4usize);
    let mut ring = ConvRing::new(conv_dim, d_conv, banks);
    let before = ring.state.clone();
    let mine = 2usize;
    let s_copy = vec![mine as u32; 4];
    for ch in 0..conv_dim {
        for r in 0..4 {
            ring.push_row(r, 4, &s_copy, ch, 0.5 * r as f32);
        }
    }
    for b in 0..banks {
        let base = b * conv_dim * ring.cs;
        let slice = base..base + conv_dim * ring.cs;
        if b == mine {
            assert_ne!(
                ring.state[slice.clone()],
                before[slice],
                "the owning bank must advance"
            );
        } else {
            assert_eq!(
                ring.state[slice.clone()],
                before[slice],
                "bank {b} was modified by another sequence's window (L179/L180 amnesia)"
            );
        }
    }
}

/// MULTI-STREAM ROWS. When a window's rows map to DIFFERENT banks (the daemon's b>1 case), each
/// row must land on its own bank. `s_copy` is indexed per row precisely so this works; a kernel
/// that hoisted the lookup out of the loop would pass every test above and fail this one.
#[test]
fn each_row_follows_its_own_s_copy_entry() {
    let (conv_dim, d_conv, banks) = (4usize, 4usize, 4usize);
    let mut ring = ConvRing::new(conv_dim, d_conv, banks);
    let before = ring.state.clone();
    let s_copy = [0u32, 3, 1, 3]; // four rows, three distinct banks; bank 2 untouched
    for ch in 0..conv_dim {
        for r in 0..4 {
            ring.push_row(r, 4, &s_copy, ch, 1.0 + r as f32);
        }
    }
    for b in 0..banks {
        let base = b * conv_dim * ring.cs;
        let slice = base..base + conv_dim * ring.cs;
        let touched = s_copy.contains(&(b as u32));
        if touched {
            assert_ne!(
                ring.state[slice.clone()],
                before[slice],
                "bank {b} should have advanced"
            );
        } else {
            assert_eq!(
                ring.state[slice.clone()],
                before[slice],
                "bank {b} must be untouched"
            );
        }
    }
}
