// Knots of light. The Arf trefoil (x = sin t + 2 sin 2t, y = cos t - 2 cos 2t, z = -sin 3t) drawn as
// particles flowing along the curve — the tokens — leaving short luminous trails, turning slowly in 3D.
// mountKnot(canvas, opts) runs one; it pauses off screen and draws a still frame when motion is reduced.
// The canvas is transparent: the trails fade by erasing, so the page and its background show through.
(function () {
  const TAU = Math.PI * 2;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const STOPS = [[0, [255, 190, 40]], [0.33, [255, 110, 30]], [0.66, [236, 40, 110]], [1, [255, 190, 40]]];
  function color(u) {
    for (let i = 1; i < STOPS.length; i++) {
      const [u1, c1] = STOPS[i], [u0, c0] = STOPS[i - 1];
      if (u <= u1) { const k = (u - u0) / (u1 - u0); return c0.map((v, j) => Math.round(v + (c1[j] - v) * k)); }
    }
    return STOPS[0][1];
  }
  const P = t => [Math.sin(t) + 2 * Math.sin(2 * t), Math.cos(t) - 2 * Math.cos(2 * t), -Math.sin(3 * t)];

  // The three crossings: for each, the t of the strand passing OVER (larger z).
  function crossings() {
    const M = 1500, ts = [], Q = [];
    for (let i = 0; i < M; i++) { ts.push(i / M * TAU); Q.push(P(ts[i])); }
    const out = [];
    for (let i = 0; i < M; i++) for (let j = i + M / 10; j < M; j++) {
      if (j - i > M - M / 10) continue;
      const dx = Q[i][0] - Q[j][0], dy = Q[i][1] - Q[j][1];
      if (dx * dx + dy * dy < 0.003) {
        const t = Q[i][2] > Q[j][2] ? ts[i] : ts[j];
        if (out.every(o => Math.abs(Math.atan2(Math.sin(t - o), Math.cos(t - o))) > 0.4)) out.push(t);
      }
    }
    // order them as the page names them: right, bottom, left
    return out.map(t => [t, P(t)]).sort((a, b) => (b[1][0] - a[1][0]) - 0.001 * (b[1][1] - a[1][1])).map(x => x[0]);
  }

  window.mountKnot = function (canvas, opt) {
    opt = Object.assign({ particles: 1300, spin: 0.0016, tilt: -0.35, follow: true, comets: 7, markers: null }, opt);
    const ctx = canvas.getContext('2d');
    const parts = [];
    for (let i = 0; i < opt.particles; i++) {
      const a = Math.random() * TAU, r = Math.sqrt(Math.random()) * 0.3;
      parts.push({ t: Math.random() * TAU, v: 0.0016 + Math.random() * 0.002, ox: Math.cos(a) * r, oy: Math.sin(a) * r, s: 0.55 + Math.random() * 1.25 });
    }
    const comets = [];
    for (let i = 0; i < opt.comets; i++) comets.push({ t: Math.random() * TAU, v: 0.006 + Math.random() * 0.004 });
    const cross = opt.markers ? crossings() : [];
    let W = 0, H = 0, rx = opt.tilt, ry = 0, tx = opt.tilt, ty = 0, spin = 0, running = true, focus = -1;
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2), r = canvas.getBoundingClientRect();
      W = r.width; H = r.height;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
    }
    addEventListener('resize', resize);
    if (opt.follow) addEventListener('pointermove', e => {
      const r = canvas.getBoundingClientRect();
      tx = opt.tilt + ((e.clientY - (r.top + r.height / 2)) / innerHeight) * 0.5;
      ty = ((e.clientX - (r.left + r.width / 2)) / innerWidth) * 0.8;
    }, { passive: true });
    // ONE loop: `frame` stops itself off screen; it is restarted only if it did stop
    let looping = true;
    new IntersectionObserver(es => {
      running = es[0].isIntersecting;
      if (running && !looping && !reduce) { looping = true; requestAnimationFrame(frame); }
    }).observe(canvas);
    function project(p, s) {
      const cy = Math.cos(ry + spin), sy = Math.sin(ry + spin), cx = Math.cos(rx), sx = Math.sin(rx);
      let [x, y, z] = p;
      [x, z] = [x * cy + z * sy, -x * sy + z * cy];
      [y, z] = [y * cx - z * sx, y * sx + z * cx];
      const f = 1 / (1 - z * 0.12);
      return [W / 2 + x * s * f, H / 2 + y * s * f, z];
    }
    function at(t, ox, oy) {
      const c = P(t), d = P(t + 0.01), tx_ = d[0] - c[0], ty_ = d[1] - c[1], len = Math.hypot(tx_, ty_) || 1;
      return [c[0] - ty_ / len * ox, c[1] + tx_ / len * ox, c[2] + oy];
    }
    function frame() {
      if (!running) { looping = false; return; }
      rx += (tx - rx) * 0.04; ry += (ty - ry) * 0.04; if (!reduce) spin += opt.spin;
      const s = Math.min(W, H) / 7.4;
      // trails: fade the last frame toward transparent instead of clearing it
      if (reduce) ctx.clearRect(0, 0, W, H);
      else {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = 'rgba(0,0,0,0.22)'; ctx.fillRect(0, 0, W, H);
      }
      const fp = focus >= 0 ? project(P(cross[focus]), s) : null;
      ctx.globalCompositeOperation = 'lighter';
      for (const q of parts) {
        if (!reduce) q.t = (q.t + q.v) % TAU;
        const [x, y, z] = project(at(q.t, q.ox, q.oy), s);
        const near = (z + 1.4) / 2.8;
        let a = 0.14 + near * 0.5, rad = q.s * (0.7 + near * 0.8);
        if (fp) { const d = Math.hypot(x - fp[0], y - fp[1]); if (d < 46) { const k = 1 - d / 46; a += k * 0.6; rad += k * 1.6; } }
        const [r, g, b] = color(q.t / TAU);
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(a, 1).toFixed(3)})`;
        ctx.beginPath(); ctx.arc(x, y, rad, 0, TAU); ctx.fill();
      }
      // comets: bright tokens racing ahead
      for (const c of comets) {
        if (!reduce) c.t = (c.t + c.v) % TAU;
        for (let k = 0; k < 14; k++) {
          const [x, y, z] = project(P(c.t - k * 0.012), s), near = (z + 1.4) / 2.8;
          ctx.fillStyle = `rgba(255,236,200,${((1 - k / 14) * (0.35 + near * 0.55)).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(x, y, (1 - k / 14) * 2.6 + 0.4, 0, TAU); ctx.fill();
        }
      }
      ctx.globalCompositeOperation = 'source-over';
      if (opt.markers) cross.forEach((t, i) => {
        const [x, y] = project(P(t), s), m = opt.markers[i];
        if (m) {
          m.style.left = (x / W * 100) + '%'; m.style.top = (y / H * 100) + '%';
          // the label sits outward from the knot's centre: a crossing faces the gap between two
          // lobes, so that is where no strand passes
          const lab = m.firstElementChild, dx = x - W / 2, dy = y - H / 2, d = Math.hypot(dx, dy) || 1;
          if (lab) lab.style.transform = `translate(calc(-50% + ${(dx / d * 50).toFixed(1)}px), calc(-50% + ${(dy / d * 40).toFixed(1)}px))`;
        }
      });
      if (!reduce) requestAnimationFrame(frame);
    }
    resize(); frame();
    // the live pose, for thread.js: it starts as this knot's own centreline, rotated as it is now
    return {
      focus(i) { focus = i; },
      pose() {
        const r = canvas.getBoundingClientRect();
        return { yaw: ry + spin, pitch: rx, s: Math.min(W, H) / 7.4, cx: r.left + W / 2, cy: r.top + H / 2 };
      },
    };
  };
})();
