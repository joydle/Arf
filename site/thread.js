// The thread: the trefoil as ONE line. Past the hero it is the reading line under the nav (its length
// is how far you have read; on a wide screen the hero's particle knot fades out as it comes in); at the
// end it draws itself, stroke by stroke, into the knot above the closing install line. The curve is the
// hero's (x = sin t + 2 sin 2t, y = cos t - 2 cos 2t, z = -sin 3t), 640 points drawn in depth order with
// a gap where a strand passes under another, in the ground layer behind the content.
// Both changes PLAY over about a second once their threshold is crossed (and reverse when scrolled
// back), in place: an earlier version morphed the line across the page with the scroll, which left the
// knot half-made wherever the reader stopped and swept a stroke through the content.
(function () {
  const cv = document.getElementById('thread');
  const startEl = document.getElementById('knot'), endEl = document.getElementById('endslot');
  const nav = document.querySelector('nav'), hero = document.querySelector('.hero');
  if (!cv || !startEl || !endEl || !nav || !hero) return;
  const fadeEl = startEl.closest('figure') || startEl;   // the knot with its labels and caption
  const cx = cv.getContext('2d'), off = document.createElement('canvas'), ox = off.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const N = 640, TAU = Math.PI * 2;
  const KN = [];
  for (let i = 0; i < N; i++) {
    const t = TAU * i / N + 0.62;
    KN.push([Math.sin(t) + 2 * Math.sin(2 * t), -(Math.cos(t) - 2 * Math.cos(2 * t)), -Math.sin(3 * t)]);
  }
  const STOPS = [[255, 190, 40], [255, 110, 30], [236, 40, 110]];
  const cl = v => (v < 0 ? 0 : v > 1 ? 1 : v), lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => { t = cl(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  const grad = s => { s = cl(s); const k = s < 0.5 ? 0 : 1, u = s < 0.5 ? s * 2 : s * 2 - 1; return STOPS[k].map((c, j) => lerp(c, STOPS[k + 1][j], u)); };
  const rgb = c => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
  let W = 0, H = 0, DPR = 1, rx = 0, ry = 0, tx = 0, ty = 0, tNow = 0, raf = 0;
  // untie (u1) and tie again (u2) PLAY once a threshold is crossed, instead of following the scroll:
  // a scroll-scrubbed morph left the knot half-tied wherever the reader stopped
  let u1 = 0, u2 = 0, tPrev = 0;
  const SPAN = 1.1;                                     // seconds for a whole untie or tie
  const toward = (v, goal, dt) => goal > v ? Math.min(goal, v + dt / SPAN) : Math.max(goal, v - dt / SPAN);
  const settled = () => u1 === goal1() && u2 === goal2();
  const goal1 = () => (scrollY > hero.offsetHeight * 0.18 ? 1 : 0);
  const goal2 = () => (endEl.getBoundingClientRect().top < H * 0.78 ? 1 : 0);
  const P = new Float32Array(N * 3);

  function size() {
    DPR = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight;
    for (const c of [cv, off]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
  }
  // a slot: the centre and radius of a knot drawn into an element's box
  function slot(el, scale) {
    const r = el.getBoundingClientRect();
    return { c: [r.left + r.width / 2, r.top + r.height / 2], R: r.width / scale };
  }
  function knot(out, s, yaw, pitch) {
    const a = Math.cos(yaw), b = Math.sin(yaw), ct = Math.cos(pitch), st = Math.sin(pitch);
    for (let i = 0; i < N; i++) {
      const [x, y, z] = KN[i];
      const X = x * a + z * 1.1 * b; let Z = -x * b + z * 1.1 * a;
      const Y = y * ct + Z * st; Z = -y * st + Z * ct;
      out[i * 3] = s.c[0] + X * s.R; out[i * 3 + 1] = s.c[1] + Y * s.R; out[i * 3 + 2] = Z;
    }
  }
  function line(out, x1, y) {
    for (let i = 0; i < N; i++) { out[i * 3] = x1 * i / (N - 1); out[i * 3 + 1] = y; out[i * 3 + 2] = 0; }
  }

  function frame() {
    const max = document.documentElement.scrollHeight - H, prog = max > 0 ? cl(scrollY / max) : 0;
    const navB = nav.getBoundingClientRect().bottom;
    const dt = reduce ? SPAN : cl((tNow - tPrev) / 1000); tPrev = tNow;
    u1 = toward(u1, goal1(), dt);                                          // the line appears
    u2 = toward(u2, goal2(), dt);                                          // the knot draws itself
    rx += (tx - rx) * 0.08; ry += (ty - ry) * 0.08;
    const idle = reduce ? 0 : tNow / 1000;
    const s2 = slot(endEl, 6.6);
    const wide = W > 860;                                                  // the page's own breakpoint
    // wide: the hero's particle knot fades in place as the reading line comes in under the nav
    if (wide) fadeEl.style.opacity = String(1 - ease(u1));
    else if (fadeEl.style.opacity) fadeEl.style.opacity = '';
    clear();
    const lineVis = (wide ? ease(u1) : 1) * (1 - ease(u2));
    if (lineVis > 0.01) { line(P, W * prog, navB + 0.5); draw(2.5, 0, lineVis, navB - 1, s2, N); }
    if (u2 > 0) {                                                          // stroke by stroke, in place
      knot(P, s2, 0.2 * Math.sin(idle * 0.5) + ry * 0.6, 0.08 * Math.sin(idle * 0.4) + rx * 0.6);
      draw(s2.R * 0.52, 1, 1, navB - 1, s2, Math.round(ease(u2) * N));
    }
  }

  function clear() { cx.setTransform(1, 0, 0, 1, 0, 0); cx.clearRect(0, 0, cv.width, cv.height); }
  function draw(w, knotW, vis, clipTop, G, upto) {
    ox.setTransform(DPR, 0, 0, DPR, 0, 0); cx.setTransform(1, 0, 0, 1, 0, 0);
    ox.clearRect(0, 0, W, H);
    if (vis <= 0.01 || upto < 2) return;
    let deep = false;
    for (let i = 0; i < N; i++) if (Math.abs(P[i * 3 + 2]) > 0.02) { deep = true; break; }
    const closed = upto >= N && Math.hypot(P[0] - P[(N - 1) * 3], P[1] - P[(N - 1) * 3 + 1]) < w * 2;
    const last = closed ? N : Math.min(upto, N) - 1, CH = 16, chunks = [];
    for (let a = 0; a < last; a += CH) {
      const b = Math.min(a + CH, last); let z = 0;
      for (let i = a; i <= b; i++) z += P[(i % N) * 3 + 2];
      chunks.push({ a, b, z: z / (b - a + 1) });
    }
    chunks.sort((u, v) => u.z - v.z || u.a - v.a);                        // far strands first
    // along the line: the brand gradient by position; as a knot: by place in the knot's plane
    const f = G.R / 31.25;
    const col = i => {
      const ci = grad(i / (N - 1));
      if (knotW <= 0) return ci;
      const k = (i % N) * 3, gx = (P[k] - G.c[0]) / f, gy = (P[k + 1] - G.c[1]) / f;
      const kc = grad(((gx + 80) * 165 + (gy + 95) * 185) / (165 * 165 + 185 * 185));
      return ci.map((v, q) => lerp(v, kc[q], knotW));
    };
    const path = (a, b) => {
      ox.beginPath();
      for (let i = a; i <= b; i++) { const k = (((i % N) + N) % N) * 3; i === a ? ox.moveTo(P[k], P[k + 1]) : ox.lineTo(P[k], P[k + 1]); }
    };
    ox.save(); ox.beginPath(); ox.rect(0, clipTop, W, H); ox.clip(); ox.lineJoin = 'round';
    for (const c of chunks) {
      if (deep) {                                                           // the gap of an under-crossing
        ox.globalCompositeOperation = 'destination-out'; ox.lineCap = 'butt'; ox.lineWidth = w * 2.15; ox.strokeStyle = '#000';
        path(closed ? c.a - 1 : Math.max(0, c.a - 1), closed ? c.b + 1 : Math.min(last, c.b + 1)); ox.stroke();
        ox.globalCompositeOperation = 'source-over';
      }
      const ka = (c.a % N) * 3, kb = (c.b % N) * 3, g = ox.createLinearGradient(P[ka], P[ka + 1], P[kb] + 0.01, P[kb + 1]);
      g.addColorStop(0, rgb(col(c.a))); g.addColorStop(1, rgb(col(c.b)));
      ox.strokeStyle = g; ox.lineCap = 'round'; ox.lineWidth = w; path(c.a, c.b); ox.stroke();
    }
    ox.restore();
    cx.globalAlpha = vis;
    if (knotW > 0.05) { cx.globalAlpha = 0.45 * knotW * vis; cx.filter = `blur(${Math.round(w * 1.4 * DPR)}px)`; cx.drawImage(off, 0, 0); cx.filter = 'none'; cx.globalAlpha = vis; }
    cx.drawImage(off, 0, 0); cx.globalAlpha = 1;
  }

  // animate only while one end is on screen (the idle sway); otherwise redraw on scroll alone
  function loop(t) {
    tNow = t; frame();
    const live = scrollY < hero.offsetHeight || endEl.getBoundingClientRect().top < H || !settled();
    raf = !reduce && live ? requestAnimationFrame(loop) : 0;
  }
  const kick = () => { if (!raf) { tPrev = performance.now(); raf = requestAnimationFrame(loop); } };
  size();
  addEventListener('resize', () => { size(); frame(); });
  addEventListener('scroll', kick, { passive: true });
  addEventListener('pointermove', e => { ty = (e.clientX / W - 0.5) * 0.9; tx = (e.clientY / H - 0.5) * -0.6; kick(); }, { passive: true });
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => {
    u1 = goal1(); u2 = goal2(); tPrev = performance.now(); kick();     // a reload mid-page starts settled
  });
})();
