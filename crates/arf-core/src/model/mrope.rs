//! Interleaved multi-axis RoPE ("IMROPE") positions for Qwen3.8 image prompts.
//!
//! Qwen3.8 (llama.cpp arch `qwen35`) rotates each attention layer's q/k with THREE position
//! axes — temporal, height, width — chosen per rotary PAIR, not per token. The GGUF says so:
//! `qwen35.rope.dimension_sections = [11, 11, 10, 0]` over `rope.dimension_count = 64`
//! (32 pairs), and llama.cpp maps the arch to `LLAMA_ROPE_TYPE_IMROPE`
//! (`src/llama-model.cpp`, `case LLM_ARCH_QWEN35`). The gated-delta-net layers have no rope.
//!
//! For a TEXT token all three axes carry the same position `p`, so every pair rotates by
//! `p * theta_i` — exactly the plain NeoX rope Arf already runs. That is the whole
//! compatibility argument, and `text_only_layout_is_the_plain_positions` +
//! `imrope_with_equal_axes_is_plain_neox` pin it.
//!
//! For an IMAGE token at merged-grid cell `(row, col)` the reference sets
//! `(t, h, w) = (p0, p0 + row, p0 + col)` where `p0` is the running text position at the
//! image, and the text after the image resumes at `p0 + max(grid_h, grid_w)` — NOT at
//! `p0 + grid_h * grid_w`. Sources, all read before writing this:
//! - llama.cpp `tools/mtmd/mtmd.cpp` `mtmd_image_tokens_get_decoder_pos` (MROPE case) and
//!   `mtmd_image_tokens_get_n_pos` (`max(nx, ny)`),
//! - another engine `runtime/model/Runtime.mm` `ropePosition` (`delta += max(mergedH, mergedW) -
//!   span.tokens`), which also recomputes the delta for every DECODE row,
//! - HF `Qwen3VLModel.get_rope_index` (`st_idx = llm_pos_ids.max() + 1`).
//!
//! So a prompt with images has a positive ROPE DELTA `= prompt_len - next_text_position`, and
//! every token generated after it (KV index `q >= prompt_len`) rotates at `q - delta` on all
//! three axes. The KV/causal index is unchanged — only the rope angle moves.

/// Qwen3.8's `qwen35.rope.dimension_sections` (the GGUF value; llama.cpp reads the same key).
/// Hardcoded rather than threaded through `ModelConfig`: it is fixed for the arch, and the only
/// consumer is the image path. Revisit if a second M-RoPE arch appears.
pub const QWEN35_MROPE_SECTIONS: [usize; 4] = [11, 11, 10, 0];

/// One image's placement in a prompt: the prompt index of its first `<|image_pad|>` token and
/// its MERGED token grid (`grid_h * grid_w` pad tokens, row-major).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub start: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl ImageGrid {
    pub fn tokens(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// The (t, h, w) rope positions of every prompt token, plus the delta that continues them
/// into generated tokens. `pos.len()` is the prompt length.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MropeLayout {
    pub pos: Vec<[u32; 3]>,
    /// `prompt_len - next_text_position` (>= 0: an image of `gh*gw` tokens advances the text
    /// position by only `max(gh, gw)`).
    pub delta: u32,
}

impl MropeLayout {
    /// Rope positions for KV index `q` of this sequence: the prompt's own table inside the
    /// prompt, `q - delta` on all three axes after it.
    pub fn at(&self, q: usize) -> [u32; 3] {
        match self.pos.get(q) {
            Some(p) => *p,
            None => {
                let p = (q as u32) - self.delta;
                [p, p, p]
            }
        }
    }
}

/// Build the layout for a prompt of `prompt_len` tokens containing `images` (ascending,
/// non-overlapping, each fully inside the prompt). Panics on an overlapping/out-of-range grid
/// — the HTTP layer builds these from the tokenized prompt, so a violation is a bug there.
pub fn mrope_layout(prompt_len: usize, images: &[ImageGrid]) -> MropeLayout {
    let mut pos = Vec::with_capacity(prompt_len);
    let mut p: u32 = 0;
    let mut i = 0usize;
    let mut next_img = 0usize;
    while i < prompt_len {
        if let Some(img) = images.get(next_img).filter(|g| g.start == i) {
            assert!(
                img.grid_h > 0 && img.grid_w > 0 && img.start + img.tokens() <= prompt_len,
                "mrope_layout: image grid {img:?} does not fit a {prompt_len}-token prompt"
            );
            for k in 0..img.tokens() {
                let (row, col) = ((k / img.grid_w) as u32, (k % img.grid_w) as u32);
                pos.push([p, p + row, p + col]);
            }
            p += img.grid_h.max(img.grid_w) as u32;
            i += img.tokens();
            next_img += 1;
        } else {
            pos.push([p, p, p]);
            p += 1;
            i += 1;
        }
    }
    assert_eq!(
        next_img,
        images.len(),
        "mrope_layout: an image start is not on a token boundary / out of order"
    );
    MropeLayout {
        pos,
        delta: prompt_len as u32 - p,
    }
}

/// Which position axis rotary pair `pair` reads under interleaved M-RoPE: 0 = t, 1 = h,
/// 2 = w, 3 = the extra axis (unused by Qwen3.8, whose 4th section is 0). A direct transcription
/// of ggml's `ggml_mrope_cache_init` (`is_imrope` branch, `ggml-cpu/ops.cpp`); with sections
/// `[11, 11, 10, 0]` over 32 pairs it is simply `pair % 3`.
pub fn imrope_axis(pair: usize, sections: [usize; 4]) -> usize {
    let sect_dims: usize = sections.iter().sum();
    let sector = pair % sect_dims;
    if sector % 3 == 1 && sector < 3 * sections[1] {
        1
    } else if sector % 3 == 2 && sector < 3 * sections[2] {
        2
    } else if sector.is_multiple_of(3) && sector < 3 * sections[0] {
        0
    } else {
        3
    }
}

/// CPU reference: rotate one head vector `x[head_dim]` in place with interleaved M-RoPE —
/// NeoX pairing `(x[i], x[i + rot/2])` over the first `rot_dim` elements, pair `i` at angle
/// `pos[imrope_axis(i)] * base^(-2i/rot_dim)`. The spec the GPU `rope_qk_b_mrope` kernel
/// must reproduce (and, with equal axes, the plain rope the GPU already runs).
pub fn apply_imrope(x: &mut [f32], rot_dim: usize, pos: [u32; 3], base: f32, sections: [usize; 4]) {
    let half = rot_dim / 2;
    for i in 0..half {
        let axis = imrope_axis(i, sections);
        let p = if axis < 3 { pos[axis] } else { 0 } as f32;
        let theta = p * base.powf(-2.0 * i as f32 / rot_dim as f32);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN38_SECTIONS: [usize; 4] = [11, 11, 10, 0];

    /// A text-only prompt's layout is exactly the positions Arf uses today — `(p, p, p)` for
    /// p = 0..n — and continues without a delta.
    #[test]
    fn text_only_layout_is_the_plain_positions() {
        let l = mrope_layout(37, &[]);
        assert_eq!(l.delta, 0);
        for (q, p) in l.pos.iter().enumerate() {
            assert_eq!(*p, [q as u32; 3]);
        }
        for q in 37..80 {
            assert_eq!(l.at(q), [q as u32; 3]);
        }
    }

    /// Hand-worked: `A B <vs> [2x3 image] <ve> C` then generation.
    ///
    ///   idx : 0  1  2   3..8 (image, row-major 2 rows x 3 cols)            9    10
    ///   tok : A  B  vs  pad pad pad / pad pad pad                           ve   C
    ///   t   : 0  1  2   3  3  3  3  3  3                                    6    7
    ///   h   : 0  1  2   3  3  3  4  4  4                                    6    7
    ///   w   : 0  1  2   3  4  5  3  4  5                                    6    7
    ///
    /// The image starts at text position 3 and advances it by max(2, 3) = 3, so `<vs_end>`
    /// sits at 6, `C` at 7, and the first generated token (KV index 11) at 8: delta = 11 - 8 = 3
    /// (= 6 tokens - max(2,3)).
    #[test]
    fn image_layout_matches_the_hand_worked_reference() {
        let l = mrope_layout(
            11,
            &[ImageGrid {
                start: 3,
                grid_h: 2,
                grid_w: 3,
            }],
        );
        let want: Vec<[u32; 3]> = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 4],
            [3, 3, 5],
            [3, 4, 3],
            [3, 4, 4],
            [3, 4, 5],
            [6, 6, 6],
            [7, 7, 7],
        ];
        assert_eq!(l.pos, want);
        assert_eq!(l.delta, 3);
        assert_eq!(l.at(11), [8, 8, 8]);
        assert_eq!(l.at(20), [17, 17, 17]);
    }

    /// Two images: each advances by its own max side; the deltas add.
    #[test]
    fn two_images_accumulate_their_deltas() {
        // text(1) img 1x4 (4 tok) text(1) img 3x2 (6 tok) text(1)  = 13 tokens.
        let imgs = [
            ImageGrid {
                start: 1,
                grid_h: 1,
                grid_w: 4,
            },
            ImageGrid {
                start: 6,
                grid_h: 3,
                grid_w: 2,
            },
        ];
        let l = mrope_layout(13, &imgs);
        assert_eq!(l.pos[0], [0, 0, 0]);
        assert_eq!(&l.pos[1..5], &[[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]]);
        assert_eq!(l.pos[5], [5, 5, 5]); // 1 + max(1,4)
        assert_eq!(l.pos[6], [6, 6, 6]);
        assert_eq!(l.pos[11], [6, 8, 7]); // last cell (row 2, col 1)
        assert_eq!(l.pos[12], [9, 9, 9]); // 6 + max(3,2)
        assert_eq!(l.delta, 13 - 10);
    }

    /// VIDEO: HF `get_rope_index` splits `video_grid_thw` into one `t = 1` grid per frame pair
    /// ("timestamps are used to separate videos"), so each pair is laid out exactly as an image
    /// and the timestamp / bracket text between pairs advances like any text. Expected values
    /// printed by a numpy transcription of transformers 5.17.0 `get_rope_index` +
    /// `get_vision_position_ids` (2026-09-27) for
    /// `A B | ts ts ts vs [pair 0: 2x3] ve | ts ts ts vs [pair 1: 2x3] ve | C`,
    /// `video_grid_thw = [[2, 4, 6]]` (2 pairs of a 4x6 patch grid = 2x3 merged). HF's delta is
    /// `-6` (`max + 1 - len`), which is this module's `+6` (`len - next position`).
    #[test]
    fn video_frame_pairs_are_laid_out_as_images() {
        let l = mrope_layout(
            25,
            &[
                ImageGrid {
                    start: 6,
                    grid_h: 2,
                    grid_w: 3,
                },
                ImageGrid {
                    start: 17,
                    grid_h: 2,
                    grid_w: 3,
                },
            ],
        );
        let hf: Vec<[u32; 3]> = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 7],
            [6, 6, 8],
            [6, 7, 6],
            [6, 7, 7],
            [6, 7, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [14, 14, 15],
            [14, 14, 16],
            [14, 15, 14],
            [14, 15, 15],
            [14, 15, 16],
            [17, 17, 17],
            [18, 18, 18],
        ];
        assert_eq!(l.pos, hf);
        assert_eq!(l.delta, 6);
        assert_eq!(l.at(25), [19, 19, 19]);
    }

    /// An image (2x2 merged) followed by a video of two 1x2 pairs, same transcription:
    /// `video_grid_thw = [[2, 2, 4]]`, `image_grid_thw = [[1, 4, 4]]`, HF delta -2.
    #[test]
    fn image_then_video_matches_hf_rope_index() {
        // T | img 2x2 | T T T | pair 1x2 | T T T | pair 1x2 | T T  = 17 tokens
        let l = mrope_layout(
            17,
            &[
                ImageGrid {
                    start: 1,
                    grid_h: 2,
                    grid_w: 2,
                },
                ImageGrid {
                    start: 8,
                    grid_h: 1,
                    grid_w: 2,
                },
                ImageGrid {
                    start: 13,
                    grid_h: 1,
                    grid_w: 2,
                },
            ],
        );
        let hf: Vec<[u32; 3]> = vec![
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [6, 6, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [11, 11, 12],
            [13, 13, 13],
            [14, 14, 14],
        ];
        assert_eq!(l.pos, hf);
        assert_eq!(l.delta, 2);
    }

    /// Pair -> axis for Qwen3.8's sections is `pair % 3` across all 32 pairs (11 t, 11 h,
    /// 10 w) — the rule another engine's kernel hardcodes (`target_positions[row*3 + dim%3]`).
    #[test]
    fn qwen38_sections_interleave_mod_three() {
        let mut count = [0usize; 4];
        for pair in 0..32 {
            let a = imrope_axis(pair, QWEN38_SECTIONS);
            assert_eq!(a, pair % 3, "pair {pair}");
            count[a] += 1;
        }
        assert_eq!(count, [11, 11, 10, 0]);
    }

    /// With t = h = w the interleaved rope IS plain NeoX rope: text tokens rotate exactly as
    /// they did before M-RoPE existed.
    #[test]
    fn imrope_with_equal_axes_is_plain_neox() {
        let hd = 256;
        let rot = 64;
        let base = 1.0e7f32;
        let x0: Vec<f32> = (0..hd)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        for p in [0u32, 1, 17, 4095] {
            let mut a = x0.clone();
            apply_imrope(&mut a, rot, [p, p, p], base, QWEN38_SECTIONS);
            let mut b = x0.clone();
            let half = rot / 2;
            for i in 0..half {
                let th = p as f32 * base.powf(-2.0 * i as f32 / rot as f32);
                let (s, c) = th.sin_cos();
                let (u, v) = (b[i], b[i + half]);
                b[i] = u * c - v * s;
                b[i + half] = v * c + u * s;
            }
            assert_eq!(a, b, "p={p}");
            assert_eq!(&a[rot..], &x0[rot..], "dims past rot_dim pass through");
        }
    }
}
