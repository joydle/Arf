// A prompt's journey through Qwen3.8-27B, faint, behind the page: its 64 layers as a slowly turning
// stack of slabs in perspective, every fourth one attention (16), the rest Gated DeltaNet (48),
// inside a drifting cloud of activations.
//   prefill  the five prompt tokens cross every layer together. At an attention layer each token
//            looks back at the earlier ones (causal, magenta) and leaves its keys behind (a dot per
//            token: the cache grows). At a recurrent layer the state ring pulses (one fixed-size
//            state per layer, carried from token to token).
//   decode   one pass emits a word and a draft of seven beside it; the kept drafts become words at
//            once, the rest fade (block speculation). The last word returns to the input along a
//            low arc and crosses again, alone, reading the cache and the states the prompt left.
// The stage and the answer are written on the caption lines at the upper left, under the nav. The
// answer is the real one. One loop, 30 fps, paused when hidden; a still frame with reduced motion.
// Cost: everything is precomputed or projected into typed arrays; paths are batched per colour;
// variable alpha goes through globalAlpha, so a frame allocates nothing.
(function () {
  'use strict';
  const canvas = document.getElementById('flow');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const PROMPT = ['Name', 'three', 'prime', 'numbers', '.'];
  // each pass: the word it emits, then the drafted words that are kept (of seven)
  const PASSES = [{ out: ['2', ',', ' 3'] }, { out: [',', ' and', ' 5'] }];
  const PASS_TXT = PASSES.map(p => p.out.join(''));
  const LAYERS = 64, DRAFT = 7, NP = PROMPT.length, NT = NP + 1;
  const INK = 'rgb(200,210,235)', AMBER = 'rgb(255,190,40)', ORANGE = 'rgb(255,110,30)', MAGENTA = 'rgb(236,40,110)';
  const MONO = 'ui-monospace, SF Mono, Menlo, monospace';
  const F11 = '11px ' + MONO, F12 = '12px ' + MONO, F13 = '13px ' + MONO, F15 = '15px ' + MONO;
  const TITLE = 'Qwen3.8-27B · 48 recurrent + 16 attention layers';
  const TAU = Math.PI * 2;
  const cl = v => (v < 0 ? 0 : v > 1 ? 1 : v);

  // ---- the world (units: one layer apart) -------------------------------------------------------
  const HY = 6, HZ = 6;                                   // slab half-height, half-depth
  const X0 = -(LAYERS - 1) / 2, XN = -X0;                 // first and last layer
  const XIN = X0 - 7, XENTER = X0 - 2.5, XOUT = XN + 1.5; // approach, decode entry, exit
  const TR = 3.4, RS = 4.7;                               // token circle, state ring radius
  const TY = new Float32Array(NT), TZ = new Float32Array(NT);
  for (let k = 0; k < NT; k++) { const a = 0.35 + k; TY[k] = TR * Math.sin(a); TZ[k] = TR * Math.cos(a); }
  const ATTN = new Uint8Array(LAYERS);
  for (let i = 0; i < LAYERS; i++) ATTN[i] = i % 4 === 3 ? 1 : 0;
  const SEG = 18, CS = new Float32Array(SEG + 1), SN = new Float32Array(SEG + 1);
  for (let s = 0; s <= SEG; s++) { CS[s] = Math.cos(s / SEG * TAU); SN[s] = Math.sin(s / SEG * TAU); }
  const CYO = [-1, 1, 1, -1], CZO = [-1, -1, 1, 1];

  // the way back: from the exit of the last prompt token, under and in front of the stack, to the
  // entry of the decode lane (a quadratic curve, sampled once)
  const ARCN = 40, ARC = new Float32Array((ARCN + 1) * 3);
  {
    const sx = XOUT, sy = TY[NP - 1], sz = TZ[NP - 1], ex = XENTER, ey = TY[NP], ez = TZ[NP];
    const cx = 0, cy = -HY - 15, cz = -HZ - 3;
    for (let i = 0; i <= ARCN; i++) {
      const u = i / ARCN, a = (1 - u) * (1 - u), b = 2 * (1 - u) * u, c = u * u;
      ARC[i * 3] = a * sx + b * cx + c * ex; ARC[i * 3 + 1] = a * sy + b * cy + c * ey; ARC[i * 3 + 2] = a * sz + b * cz + c * ez;
    }
  }

  // the cloud: a box of faint points drifting along the stream (uniform, clustered, and a denser
  // core inside the stack); a seeded generator, so the cloud looks the same on every visit
  const NMAX = 2600, CX = 46, CY = 15, CZ = 15;
  const QX = new Float32Array(NMAX), QY = new Float32Array(NMAX), QZ = new Float32Array(NMAX), QV = new Float32Array(NMAX);
  const SX = new Float32Array(NMAX), SY = new Float32Array(NMAX), BK = new Uint8Array(NMAX);
  {
    let seed = 20261005;
    const rnd = () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = () => (rnd() + rnd() + rnd() - 1.5) * 2;
    const NC = 11, cc = [];
    for (let c = 0; c < NC; c++) cc.push([(rnd() * 2 - 1) * CX, (rnd() * 2 - 1) * CY * 0.8, (rnd() * 2 - 1) * CZ * 0.8, 1.8 + rnd() * 2.4]);
    for (let i = 0; i < NMAX; i++) {
      const r = rnd();
      if (r < 0.45) {
        QX[i] = (rnd() * 2 - 1) * CX; QY[i] = (rnd() * 2 - 1) * CY; QZ[i] = (rnd() * 2 - 1) * CZ;
      } else if (r < 0.75) {
        const c = cc[(rnd() * NC) | 0];
        QX[i] = c[0] + gauss() * c[3] * 1.7;
        QY[i] = Math.max(-CY, Math.min(CY, c[1] + gauss() * c[3]));
        QZ[i] = Math.max(-CZ, Math.min(CZ, c[2] + gauss() * c[3]));
        if (QX[i] > CX) QX[i] -= 2 * CX; else if (QX[i] < -CX) QX[i] += 2 * CX;
      } else {
        QX[i] = (rnd() * 2 - 1) * CX; QY[i] = (rnd() * 2 - 1) * HY * 0.85; QZ[i] = (rnd() * 2 - 1) * HZ * 0.85;
      }
      QV[i] = (0.22 + rnd() * 0.5) / 1000;                // 0.2–0.7 layers a second
    }
  }
  const DOT_A = [0, 0.055, 0.09, 0.14, 0.3], DOT_S = [0, 1.1, 1.2, 1.6, 1.7];

  // ---- the camera: yaw about the vertical, a downward pitch, perspective ------------------------
  const DIST = 78;
  let YAW0 = 0.62, cY = 1, sY = 0, cP = 1, sP = 0, K = 1, OX = 0, OY = 0, PX = 0, PY = 0, PP = 1;
  function setCam(yaw, pitch) { cY = Math.cos(yaw); sY = Math.sin(yaw); cP = Math.cos(pitch); sP = Math.sin(pitch); }
  function camAt(tt) { setCam(YAW0 + 0.15 * Math.sin(tt * TAU / 58000), 0.3 + 0.05 * Math.sin(tt * TAU / 86000 + 1.3)); }
  function proj(x, y, z) {                                // result in PX, PY (screen) and PP (scale)
    const xr = x * cY - z * sY, zr = x * sY + z * cY;
    const yr = y * cP + zr * sP, zd = zr * cP - y * sP;
    PP = DIST / (DIST + zd);
    PX = OX + xr * PP * K; PY = OY - yr * PP * K;
  }
  // scale and place the stack (with its draft and the way back) for every angle the camera takes
  function fit() {
    K = 1; OX = 0; OY = 0;
    let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
    const upd = () => { if (PX < a) a = PX; if (PX > b) b = PX; if (PY < c) c = PY; if (PY > d) d = PY; };
    for (let s = -1; s <= 1; s++) for (let q = 0; q < 2; q++) {
      setCam(YAW0 + s * 0.15, q ? 0.35 : 0.25);
      for (let m = 0; m < 8; m++) { proj(m & 1 ? XOUT + 2 + DRAFT * 1.3 : XIN, m & 2 ? HY : -HY, m & 4 ? HZ : -HZ); upd(); }
      for (let i = 0; i <= ARCN; i += 4) { proj(ARC[i * 3], ARC[i * 3 + 1], ARC[i * 3 + 2]); upd(); }
    }
    const top = W < 760 ? 0.36 : 0.3, bot = 0.95;
    K = Math.min(W * 0.92 / (b - a), H * (bot - top) / (d - c));
    OX = W * 0.5 - (a + b) / 2 * K;
    OY = H * (top + bot) / 2 - (c + d) / 2 * K;
  }

  // ---- the story -------------------------------------------------------------------------------
  const PRE_MS = 8200, DEC_MS = 3400, RET_MS = 1700, REST_MS = 4800;
  const GLOW = new Float32Array(LAYERS), FLASH = new Float32Array(LAYERS), PULSE = new Float32Array(LAYERS);
  const STATE = new Uint8Array(LAYERS), KV = new Uint8Array(LAYERS), QS = new Uint8Array(LAYERS);
  const SLB = new Float32Array(LAYERS * 8), NX = new Float32Array(NT), NY = new Float32Array(NT), NS = new Float32Array(NT);
  let W = 0, H = 0, x0 = 0, n = 0, t = 0, last = 0, ansW = -1;
  let phase = 'prefill', fx = XIN, v = 0, laneA = 0, laneB = NP, lastL = -1, pass = 0, emitPass = 0, emitLane = 0;
  let words = [], emitAt = 0, draftDone = false, retAt = -1e9, restAt = 0, decWord = '', stage = '', stageKey = -1;

  function resize() {
    W = innerWidth; H = innerHeight;
    canvas.width = W; canvas.height = H;                  // a background: normal resolution is enough
    // the caption starts where the page's text does (the headline's left edge)
    const h1 = document.querySelector('.hero h1');
    x0 = h1 ? Math.round(h1.getBoundingClientRect().left) : W * (W < 760 ? 0.05 : 0.06);
    YAW0 = W < 760 ? 1.1 : 0.62;                          // narrow screens look further down the stack
    n = Math.round(Math.min(NMAX, Math.max(900, W * H / 500)));
    fit();
    ansW = -1;
    for (let i = 0; i < words.length; i++) words[i].wd = -1;
    if (reduce) draw();
  }

  function start() {
    GLOW.fill(0); FLASH.fill(0); PULSE.fill(0); STATE.fill(0); KV.fill(0);
    phase = 'prefill'; fx = XIN; v = (XOUT - XIN) / PRE_MS; laneA = 0; laneB = NP; lastL = -1; pass = 0;
    words = []; draftDone = false; retAt = -1e9; stageKey = -1;
    updateStage();
  }

  function cross(i) {                                     // the tokens reach layer i
    GLOW[i] = 1;
    if (ATTN[i]) { FLASH[i] = 1; KV[i] = laneB; QS[i] = laneA; } else { PULSE[i] = 1; STATE[i] = 1; }
  }

  function emit() {
    phase = 'emit'; emitAt = t; emitPass = pass; emitLane = laneB - 1; draftDone = false;
    words.push({ w: PASSES[pass].out[0], at: t, wd: -1 });
  }

  function step(dt) {
    t += dt;
    const g = Math.exp(-dt / 520), f = Math.exp(-dt / 760), p = Math.exp(-dt / 480);
    for (let i = 0; i < LAYERS; i++) { GLOW[i] *= g; FLASH[i] *= f; PULSE[i] *= p; }
    for (let i = 0; i < n; i++) { let x = QX[i] + QV[i] * dt; if (x > CX) x -= 2 * CX; QX[i] = x; }
    if (phase === 'prefill' || phase === 'decode') {
      fx += v * dt;
      if (fx >= X0) { const L = Math.min(LAYERS - 1, Math.floor(fx - X0)); while (lastL < L) cross(++lastL); }
      if (fx >= XOUT) emit();
    } else if (phase === 'emit') {
      const age = t - emitAt, out = PASSES[pass].out;
      if (!draftDone && age > 950) {                      // the kept drafts become words together
        for (let i = 1; i < out.length; i++) words.push({ w: out[i], at: t + (i - 1) * 70, wd: -1 });
        draftDone = true;
      }
      if (draftDone && age > 1500) {
        if (pass < PASSES.length - 1) { pass++; phase = 'return'; retAt = t; decWord = words[words.length - 1].w.trim(); }
        else { phase = 'rest'; restAt = t; }
      }
    } else if (phase === 'return') {
      if (t - retAt >= RET_MS) {                          // the last word crosses again, alone
        phase = 'decode'; fx = XENTER; v = (XOUT - XENTER) / DEC_MS; laneA = NP; laneB = NP + 1; lastL = -1;
      }
    } else if (phase === 'rest' && t - restAt >= REST_MS) start();
    updateStage();
  }

  function updateStage() {                                // rebuilt only when it changes
    const id = phase === 'prefill' ? 100 + lastL : phase === 'decode' ? 200 + lastL
      : phase === 'emit' ? 300 + pass * 2 + (draftDone ? 1 : 0) : phase === 'return' ? 400 : 500;
    if (id === stageKey) return;
    stageKey = id;
    if (phase === 'prefill' || phase === 'decode') {
      const pre = phase === 'prefill';
      const who = pre ? 'prefill · 5 tokens together' : `decode · "${decWord}" alone`;
      stage = lastL < 0 ? who + ' · entering'
        : `${who} · layer ${lastL + 1}/64 · ` + (ATTN[lastL]
          ? (pre ? 'attention, each looks back' : 'attention over every earlier token')
          : 'Gated DeltaNet, a fixed-size state');
    } else if (phase === 'emit') {
      const out = PASSES[pass].out;
      stage = draftDone ? `pass ${pass + 1} · ${out.length - 1} of 7 drafts kept · "${PASS_TXT[pass]}" in one pass`
        : `pass ${pass + 1} emits "${out[0]}" · DFlash drafts seven more`;
    } else if (phase === 'return') {
      stage = `autoregression · "${decWord}" goes back in`;
    } else {
      stage = `done · "${PASS_TXT.join('')}" · ${words.length} tokens from ${PASSES.length} passes`;
    }
  }

  // ---- drawing ---------------------------------------------------------------------------------
  function ring(x, r) {                                   // a circle in a slab's plane (affine)
    proj(x, 0, 0); const cx = PX, cy = PY;
    proj(x, r, 0); const ax = PX - cx, ay = PY - cy;
    proj(x, 0, r); const bx = PX - cx, by = PY - cy;
    ctx.moveTo(cx + bx, cy + by);
    for (let s = 1; s <= SEG; s++) ctx.lineTo(cx + ax * SN[s] + bx * CS[s], cy + ay * SN[s] + by * CS[s]);
  }
  function slabPath(i) {
    const o = i * 8;
    ctx.moveTo(SLB[o], SLB[o + 1]); ctx.lineTo(SLB[o + 2], SLB[o + 3]);
    ctx.lineTo(SLB[o + 4], SLB[o + 5]); ctx.lineTo(SLB[o + 6], SLB[o + 7]); ctx.closePath();
  }
  function dot(x, y, r) { ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, TAU); }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    camAt(t);
    ctx.lineWidth = 1;
    const mem = phase === 'rest' ? cl(1 - (t - restAt - (REST_MS - 1400)) / 1200) : 1;
    const active = phase === 'prefill' || phase === 'decode';
    const afx = active && fx > X0 - 1 && fx < XN + 2.4 ? fx : 1e9;

    // the cloud: project once, sort into four brightness buckets, one path per bucket
    for (let i = 0; i < n; i++) {
      const x = QX[i], y = QY[i], z = QZ[i];
      const xr = x * cY - z * sY, zr = x * sY + z * cY, yr = y * cP + zr * sP, zd = zr * cP - y * sP;
      const p = DIST / (DIST + zd), px = OX + xr * p * K, py = OY - yr * p * K;
      if (px < -2 || px > W + 2 || py < -2 || py > H + 2) { BK[i] = 0; continue; }
      SX[i] = px; SY[i] = py;
      const dx = x - afx;                                 // activations lit where the tokens are
      if (dx > -2.4 && dx < 1.2 && y < HY + 2 && y > -HY - 2 && z < HZ + 2 && z > -HZ - 2) { BK[i] = 4; continue; }
      let e = (CX - (x < 0 ? -x : x)) * 0.12; if (e > 1) e = 1;
      const b = (0.3 + 0.7 * cl((p - 0.72) * 1.7)) * e;
      BK[i] = b < 0.2 ? 0 : b < 0.45 ? 1 : b < 0.72 ? 2 : 3;
    }
    for (let b = 1; b <= 4; b++) {
      const s = DOT_S[b], h = s / 2;
      ctx.globalAlpha = DOT_A[b]; ctx.fillStyle = b === 4 ? AMBER : INK;
      ctx.beginPath();
      for (let i = 0; i < n; i++) if (BK[i] === b) ctx.rect(SX[i] - h, SY[i] - h, s, s);
      ctx.fill();
    }

    // the cloud's box
    ctx.globalAlpha = 0.035; ctx.strokeStyle = INK; ctx.beginPath();
    for (let e = 0; e < 3; e++) for (let m = 0; m < 8; m++) {
      const bit = 1 << e;
      if (m & bit) continue;
      proj(m & 1 ? CX : -CX, m & 2 ? CY : -CY, m & 4 ? CZ : -CZ); ctx.moveTo(PX, PY);
      const m2 = m | bit;
      proj(m2 & 1 ? CX : -CX, m2 & 2 ? CY : -CY, m2 & 4 ? CZ : -CZ); ctx.lineTo(PX, PY);
    }
    ctx.stroke();

    // the slabs
    for (let i = 0; i < LAYERS; i++) {
      const x = X0 + i;
      for (let c = 0; c < 4; c++) { proj(x, CYO[c] * HY, CZO[c] * HZ); SLB[i * 8 + c * 2] = PX; SLB[i * 8 + c * 2 + 1] = PY; }
    }
    ctx.globalAlpha = 0.05; ctx.strokeStyle = INK; ctx.beginPath();
    for (let c = 0; c < 4; c++) {                         // the stack's long edges and its axis
      ctx.moveTo(SLB[c * 2], SLB[c * 2 + 1]); ctx.lineTo(SLB[(LAYERS - 1) * 8 + c * 2], SLB[(LAYERS - 1) * 8 + c * 2 + 1]);
    }
    for (let i = 0; i < LAYERS; i++) if (!ATTN[i]) slabPath(i);
    ctx.stroke();
    ctx.globalAlpha = 0.06; ctx.strokeStyle = AMBER; ctx.beginPath();
    proj(XIN, 0, 0); ctx.moveTo(PX, PY); proj(XOUT, 0, 0); ctx.lineTo(PX, PY); ctx.stroke();
    ctx.fillStyle = MAGENTA; ctx.globalAlpha = 0.022;
    for (let i = 3; i < LAYERS; i += 4) { ctx.beginPath(); slabPath(i); ctx.fill(); }
    ctx.globalAlpha = 0.12; ctx.strokeStyle = MAGENTA; ctx.beginPath();
    for (let i = 3; i < LAYERS; i += 4) slabPath(i);
    ctx.stroke();
    for (let i = 0; i < LAYERS; i++) {                    // the layers the tokens just crossed
      if (GLOW[i] < 0.04) continue;
      ctx.globalAlpha = (ATTN[i] ? 0.32 : 0.22) * GLOW[i]; ctx.strokeStyle = ATTN[i] ? MAGENTA : ORANGE;
      ctx.beginPath(); slabPath(i); ctx.stroke();
    }

    // recurrent state: one ring per layer, fixed size, pulsing when a token passes
    ctx.strokeStyle = ORANGE;
    if (mem > 0) {
      ctx.globalAlpha = 0.07 * mem; ctx.beginPath();
      for (let i = 0; i < LAYERS; i++) if (STATE[i]) ring(X0 + i, RS);
      ctx.stroke();
    }
    for (let i = 0; i < LAYERS; i++) {
      if (PULSE[i] < 0.03) continue;
      ctx.globalAlpha = 0.4 * PULSE[i]; ctx.beginPath(); ring(X0 + i, RS * (1 + 0.22 * (1 - PULSE[i]))); ctx.stroke();
    }

    // attention: the cache (a dot per token, growing) and the causal lines back
    ctx.fillStyle = MAGENTA;
    if (mem > 0) {
      ctx.globalAlpha = 0.3 * mem; ctx.beginPath();
      for (let i = 3; i < LAYERS; i += 4) for (let k = 0; k < KV[i]; k++) { proj(X0 + i, TY[k], TZ[k]); ctx.rect(PX - 1, PY - 1, 2, 2); }
      ctx.fill();
    }
    ctx.strokeStyle = MAGENTA;
    for (let i = 3; i < LAYERS; i += 4) {
      if (FLASH[i] < 0.03) continue;
      const x = X0 + i, nk = KV[i];
      ctx.globalAlpha = (QS[i] ? 0.4 : 0.3) * FLASH[i] * mem; ctx.beginPath();
      for (let k = QS[i] > 0 ? QS[i] : 1; k < nk; k++) {
        proj(x, TY[k], TZ[k]); const ax = PX, ay = PY;
        for (let j = 0; j < k; j++) {
          proj(x - 1.6, (TY[k] + TY[j]) * 0.62, (TZ[k] + TZ[j]) * 0.62); const qx = PX, qy = PY;
          proj(x, TY[j], TZ[j]);
          ctx.moveTo(ax, ay); ctx.quadraticCurveTo(qx, qy, PX, PY);
        }
      }
      ctx.stroke();
    }

    // the tokens crossing
    if (active) {
      const xs = phase === 'prefill' ? XIN : XENTER, xe = fx < XOUT ? fx : XOUT;
      ctx.strokeStyle = AMBER; ctx.globalAlpha = 0.07; ctx.beginPath();
      for (let k = laneA; k < laneB; k++) { proj(xs, TY[k], TZ[k]); ctx.moveTo(PX, PY); proj(xe, TY[k], TZ[k]); ctx.lineTo(PX, PY); }
      ctx.stroke();
      const hs = xe - 5 > xs ? xe - 5 : xs;
      ctx.strokeStyle = ORANGE; ctx.globalAlpha = 0.34; ctx.beginPath();
      for (let k = laneA; k < laneB; k++) { proj(hs, TY[k], TZ[k]); ctx.moveTo(PX, PY); proj(xe, TY[k], TZ[k]); ctx.lineTo(PX, PY); }
      ctx.stroke();
      for (let k = laneA; k < laneB; k++) { proj(xe, TY[k], TZ[k]); NX[k] = PX; NY[k] = PY; NS[k] = PP; }
      ctx.fillStyle = AMBER; ctx.globalAlpha = 0.09; ctx.beginPath();
      for (let k = laneA; k < laneB; k++) dot(NX[k], NY[k], 6.5 * NS[k]);
      ctx.fill();
      ctx.globalAlpha = 0.85; ctx.beginPath();
      for (let k = laneA; k < laneB; k++) dot(NX[k], NY[k], 2.2 * NS[k]);
      ctx.fill();
      ctx.font = F11; ctx.fillStyle = INK;
      if (phase === 'prefill') {                          // the words, until they are inside
        const la = cl((X0 + 3 - fx) / 4) * 0.34;
        if (la > 0.01) { ctx.globalAlpha = la; for (let k = 0; k < NP; k++) ctx.fillText(PROMPT[k], NX[k] + 7, NY[k] - 7); }
      } else { ctx.globalAlpha = 0.3; ctx.fillText(decWord, NX[NP] + 7, NY[NP] - 7); }
    }

    // the word out, and the draft of seven beyond it
    let ef = 0;
    if (phase === 'emit') ef = 1;
    else if (phase === 'return') ef = cl(1 - (t - retAt) / 600);
    else if (phase === 'rest') ef = cl(1 - (t - restAt) / 900);
    if (ef > 0) {
      const age = t - emitAt, out = PASSES[emitPass].out, ty = TY[emitLane], tz = TZ[emitLane];
      const keep = out.length - 1, done = phase !== 'emit' || draftDone;
      proj(XOUT, ty, tz); const ex = PX, ey = PY, es = PP;
      ctx.fillStyle = AMBER; ctx.globalAlpha = 0.7 * ef; ctx.beginPath(); dot(ex, ey, 2.4 * es); ctx.fill();
      // (no word here: the stack's far end sits behind the hero's knot; the caption carries it)
      if (age < 1300) {
        ctx.strokeStyle = INK; ctx.globalAlpha = 0.06 * ef; ctx.beginPath(); ctx.moveTo(ex, ey);
        proj(XOUT + 2 + (DRAFT - 1) * 1.3, ty, tz); ctx.lineTo(PX, PY); ctx.stroke();
        const fade = 1 - cl((age - 650) / 450);
        for (let i = 0; i < DRAFT; i++) {
          const kept = i < keep;
          if (kept && done) continue;                     // became words
          const a = cl((age - 100 - i * 45) / 150) * (kept ? 0.6 : 0.4 * fade);
          if (a < 0.01) continue;
          proj(XOUT + 2 + i * 1.3, ty, tz);
          ctx.fillStyle = kept && age > 450 ? AMBER : INK; ctx.globalAlpha = a * ef;
          ctx.beginPath(); dot(PX, PY, 1.9 * PP); ctx.fill();
        }
      }
    }

    // the way back: the last word returns to the input
    const ra = t - retAt;
    if (ra >= 0 && ra < RET_MS + 1500) {
      const fa = ra < RET_MS ? cl(ra / 300) : 1 - (ra - RET_MS) / 1500;
      ctx.strokeStyle = AMBER; ctx.globalAlpha = 0.13 * fa; ctx.beginPath();
      for (let i = 0; i <= ARCN; i++) { proj(ARC[i * 3], ARC[i * 3 + 1], ARC[i * 3 + 2]); if (i) ctx.lineTo(PX, PY); else ctx.moveTo(PX, PY); }
      ctx.stroke();
      if (ra < RET_MS) {
        const u = ra / RET_MS, e = u * u * (3 - 2 * u), f = e * ARCN, i = Math.min(ARCN - 1, f | 0), w = f - i, o = i * 3;
        proj(ARC[o] + (ARC[o + 3] - ARC[o]) * w, ARC[o + 1] + (ARC[o + 4] - ARC[o + 1]) * w, ARC[o + 2] + (ARC[o + 5] - ARC[o + 2]) * w);
        ctx.fillStyle = AMBER; ctx.globalAlpha = 0.75; ctx.beginPath(); dot(PX, PY, 2.4 * PP); ctx.fill();
        ctx.font = F11; ctx.fillStyle = INK; ctx.globalAlpha = 0.3; ctx.fillText(decWord, PX + 7, PY - 7);
      }
    }

    // the caption lines, under the nav — only beside a two-column hero (the page's own 860px
    // breakpoint: below it the hero's first line sits there), and fading out as the hero scrolls
    // away, so no later section's text ever runs over it
    const capK = W > 860 ? cl(1 - scrollY / 220) : 0;
    if (capK <= 0) { ctx.globalAlpha = 1; return; }
    const cy0 = Math.max(104, H * 0.13);
    ctx.font = F11; ctx.fillStyle = INK; ctx.globalAlpha = 0.24 * capK; ctx.fillText(TITLE, x0, cy0);
    ctx.fillStyle = AMBER; ctx.globalAlpha = 0.4 * capK; ctx.fillText(stage, x0, cy0 + 18);
    const by = cy0 + 40;
    ctx.font = F13; ctx.fillStyle = INK; ctx.globalAlpha = 0.24 * capK; ctx.fillText('answer', x0, by);
    if (ansW < 0) ansW = ctx.measureText('answer  ').width;
    let x = x0 + ansW;
    ctx.font = F15;
    for (let i = 0; i < words.length; i++) {
      const o = words[i];
      if (o.wd < 0) o.wd = ctx.measureText(o.w).width;
      const a = cl((t - o.at) / 350) * 0.62 * capK;
      if (a > 0) { ctx.globalAlpha = a; ctx.fillText(o.w, x, by); }
      x += o.wd;
    }
    ctx.globalAlpha = 1;
  }

  // ONE loop: it stops itself while the page is hidden and is restarted only if it did stop (a
  // hidden-then-visible flip inside one frame would otherwise start a second loop, and each
  // would advance the story: twice the speed).
  let visible = !document.hidden, running = false;
  document.addEventListener('visibilitychange', () => {
    visible = !document.hidden;
    if (visible && !reduce && !running) { running = true; last = performance.now(); requestAnimationFrame(loop); }
  });
  function loop(now) {
    if (!visible) { running = false; return; }
    const dt = now - last;
    if (dt >= 33) { last = now; step(Math.min(dt, 100)); draw(); }   // ~30 fps
    requestAnimationFrame(loop);
  }
  addEventListener('resize', resize);

  // Where the page has things to read or look at, the stack steps back: a mask fades it to ~30%
  // under the text column (all of it on a phone, to ~32%), opens a soft hole where the knot is, and
  // recedes to ~25% everywhere once the hero has scrolled away (every later section is text). Viewport coordinates, so it is
  // recomputed on scroll (once per frame at most) and resize.
  const knotEl = document.getElementById('knot'), textEl = document.querySelector('.hero h1');
  let maskQueued = false;
  function mask() {
    maskQueued = false;
    const parts = [];
    if (knotEl) {
      const r = knotEl.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const rad = Math.max(1, r.width * 0.55);
      parts.push(`radial-gradient(circle ${rad.toFixed(0)}px at ${cx.toFixed(0)}px ${cy.toFixed(0)}px, rgba(0,0,0,.15) 55%, #000 100%)`);
    }
    if (innerWidth <= 860) {
      // a phone has no free side: the text runs over the whole stack, so all of it steps back
      parts.push('linear-gradient(rgba(0,0,0,.32), rgba(0,0,0,.32))');
    } else if (textEl) {
      const right = textEl.getBoundingClientRect().left + Math.min(640, innerWidth * 0.45);
      parts.push(`linear-gradient(90deg, rgba(0,0,0,.3) ${right.toFixed(0)}px, #000 ${(right + 220).toFixed(0)}px)`);
    }
    // past the hero, every section is text: the whole stack recedes to ~25% as the hero scrolls away
    const hero = document.querySelector('.hero');
    if (hero) {
      const k = cl(scrollY / Math.max(1, hero.offsetHeight * 0.7)), a = (1 - 0.75 * k).toFixed(3);
      if (k > 0) parts.push(`linear-gradient(rgba(0,0,0,${a}), rgba(0,0,0,${a}))`);
    }
    const m = parts.join(', ');
    canvas.style.webkitMaskImage = canvas.style.maskImage = m;
    canvas.style.webkitMaskComposite = 'source-in';
    canvas.style.maskComposite = 'intersect';
  }
  const queueMask = () => { if (!maskQueued) { maskQueued = true; requestAnimationFrame(mask); } };
  addEventListener('scroll', queueMask, { passive: true });
  addEventListener('resize', queueMask);
  mask();
  start();
  if (reduce) {                       // a still frame: the prompt part-way through, the answer written
    while (fx < X0 + 0.42 * (XN - X0)) step(33);
    words = [];
    for (let p = 0; p < PASSES.length; p++) for (const w of PASSES[p].out) words.push({ w, at: -1e6, wd: -1 });
    stage = 'prefill, then decode with drafts of seven';
    resize();                         // sizes the canvas and draws
  } else { resize(); running = true; last = performance.now(); requestAnimationFrame(loop); }
})();
