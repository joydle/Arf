#!/usr/bin/env python3
"""Generate every brand asset in assets/ (run from the repository root): python3 assets/gen_logo.py

The mark is a TREFOIL KNOT — the simplest knot whose Arf invariant is 1; the project is named for Cahit Arf
(1910-1997) — drawn as a lit 3D tube from the knot's own parametric form
(x = sin t + 2 sin 2t, y = cos t - 2 cos 2t, z = -sin 3t). The strand is painted once in order along t (no
seams), then the over-strand of each of the three crossings is painted again with its edge, so every
crossing alternates over and under as a real knot does. The colour runs amber -> orange -> magenta -> amber
along the strand; nearer parts are lighter. The wordmark is drawn, not typeset: no font dependency.
"""
import math
# Trefoil: x = sin t + 2 sin 2t, y = cos t - 2 cos 2t, z = -sin 3t  (the knot with Arf invariant 1)
N = 720
def P(t):
    return (math.sin(t) + 2*math.sin(2*t), math.cos(t) - 2*math.cos(2*t), -math.sin(3*t))
def lerp(a, b, u): return tuple(round(a[i] + (b[i]-a[i])*u) for i in range(3))
STOPS = [(0.0, (255,190,40)), (0.33, (255,110,30)), (0.66, (236,40,110)), (1.0, (255,190,40))]
def color(u):
    for (u0,c0),(u1,c1) in zip(STOPS, STOPS[1:]):
        if u <= u1: return lerp(c0, c1, (u-u0)/(u1-u0))
    return STOPS[-1][1]
def hexc(c): return '#%02x%02x%02x' % c
def over_intervals():
    # The three crossings: pairs (t1, t2) far apart along the curve whose projections meet; the
    # strand with the larger z passes over. Returns the t of each over-strand.
    M=2000; ts=[2*math.pi*i/M for i in range(M)]; Q=[P(t) for t in ts]
    overs=[]
    for i in range(M):
        for j in range(i+M//10, M):
            if (j-i) > M-M//10: continue
            dx=Q[i][0]-Q[j][0]; dy=Q[i][1]-Q[j][1]
            if dx*dx+dy*dy < 0.004:
                t=ts[i] if Q[i][2]>Q[j][2] else ts[j]
                if all(abs(math.remainder(t-o, 2*math.pi))>0.3 for o in overs): overs.append(t)
    return overs
def knot_svg(cx, cy, scale, width, dark="#0b1020", shadow=True, pulse=False):
    X=lambda p:(cx+p[0]*scale, cy+p[1]*scale)
    def strokes(t0, t1, n, outline):
        out=[]
        ts=[t0+(t1-t0)*k/n for k in range(n+1)]
        if outline and dark!="none":
            poly=" ".join("%.2f,%.2f"%X(P(t)) for t in ts)
            out.append(f'<polyline points="{poly}" stroke="{dark}" stroke-width="{width*1.38:.1f}" stroke-linecap="butt"/>')
        for a,b in zip(ts,ts[1:]):
            pa,pb=X(P(a)),X(P(b)); u=((a/(2*math.pi))%1)
            zz=(P(a)[2]+P(b)[2])/2; shade=0.70+0.30*(zz+1)/2
            col=tuple(min(255,int(v*shade)) for v in color(u))
            out.append(f'<path d="M{pa[0]:.2f} {pa[1]:.2f}L{pb[0]:.2f} {pb[1]:.2f}" stroke="{hexc(col)}" stroke-width="{width:.1f}" stroke-linecap="round"/>')
        return out
    def highlight(t0, t1, n):
        out=[]; ts=[t0+(t1-t0)*k/n for k in range(n+1)]
        dx,dy=-width*0.15,-width*0.17
        for a,b in zip(ts,ts[1:]):
            pa,pb=X(P(a)),X(P(b)); u=((a/(2*math.pi))%1)
            zz=(P(a)[2]+P(b)[2])/2; shade=0.70+0.30*(zz+1)/2
            col=tuple(min(255,int(v*shade)) for v in color(u)); hi=tuple(min(255,int(v+(255-v)*0.34)) for v in col)
            out.append(f'<path d="M{pa[0]+dx:.2f} {pa[1]+dy:.2f}L{pb[0]+dx:.2f} {pb[1]+dy:.2f}" stroke="{hexc(hi)}" stroke-width="{width*0.17:.1f}" stroke-linecap="round"/>')
        return out
    out=[]
    if shadow:
        full=" ".join("%.2f,%.2f"%(X(P(2*math.pi*k/N))[0]+width*0.25, X(P(2*math.pi*k/N))[1]+width*0.55) for k in range(N+1))
        out.append(f'<polyline points="{full}" stroke="#000" stroke-opacity="0.55" stroke-width="{width*1.2:.1f}" filter="url(#soft)"/>')
    out += strokes(0, 2*math.pi, N, True)
    out += highlight(0, 2*math.pi, N)
    if pulse:  # a bead of light running along the strand, painted BEFORE the crossings so it dives under them
        full = " ".join("%.2f,%.2f" % X(P(2*math.pi*k/N)) for k in range(N+1))
        out.append(f'<polyline class="pulse" pathLength="1000" points="{full}" stroke="#fff6e0" stroke-width="{width*0.55:.1f}" stroke-linecap="round" filter="url(#bloom)"/>')
        out.append(f'<polyline class="pulse" pathLength="1000" points="{full}" stroke="#ffffff" stroke-width="{width*0.28:.1f}" stroke-linecap="round"/>')
    for t in over_intervals():
        out += strokes(t-0.20, t+0.20, 40, True)       # the over-strand's edge, inside the repaint
        out += strokes(t-0.30, t+0.30, 60, False)      # its body, a little longer, hides the edge ends
        out += highlight(t-0.30, t+0.30, 60)
    return '<g fill="none" stroke-linejoin="round">'+''.join(out)+'</g>'
def squircle(s, r=0.2237):
    # Apple-like continuous corner: a superellipse, n=5
    pts=[]; n=5; a=s/2
    for k in range(361):
        th=2*math.pi*k/360; c,s_=math.cos(th),math.sin(th)
        x=a+a*math.copysign(abs(c)**(2/n),c); y=a+a*math.copysign(abs(s_)**(2/n),s_)
        pts.append(f"{x:.2f},{y:.2f}")
    return "M"+" L".join(pts)+"Z"

def word(ink):
    return f'''  <g transform="translate(262,58)" fill="none" stroke="{ink}" stroke-width="17" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="48" cy="92" r="40"/>
    <path d="M88,52 L88,132"/>
    <path d="M150,52 L150,132"/>
    <path d="M150,96 C150,66 172,50 204,54"/>
    <path d="M262,132 L262,34 C262,8 282,-4 310,2"/>
    <path d="M236,62 L300,62"/>
  </g>'''

DESC = "A trefoil knot, the simplest knot whose Arf invariant is 1. Named for Cahit Arf (1910-1997)."
SOFT = '<filter id="soft" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="{s}"/></filter>'

def mark_svg(gap, size=512):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="Arf">
  <title>Arf</title><desc>{DESC}</desc>
{knot_svg(size/2, size*0.52, size*0.135, size*0.07, dark=gap, shadow=False)}
</svg>
'''

def logo_svg(gap, ink, name):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 230" role="img" aria-label="Arf">
  <title>Arf</title>
  <desc>Arf: the trefoil mark beside the drawn wordmark, for a {name} background. {DESC}</desc>
{knot_svg(118, 126, 33, 16, dark=gap, shadow=False)}
{word(ink)}
</svg>
'''

def icon_svg(size=1024):
    grid = "".join(f'<circle cx="{x}" cy="{y}" r="2.2" fill="#ffffff" fill-opacity="0.05"/>'
                   for x in range(64, size, 48) for y in range(64, size, 48))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="Arf">
  <title>Arf</title><desc>{DESC}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1b2140"/><stop offset="1" stop-color="#070a16"/></linearGradient>
    <radialGradient id="glow" cx="0.5" cy="0.52" r="0.5"><stop offset="0" stop-color="#ff6a2a" stop-opacity="0.35"/><stop offset="1" stop-color="#ff6a2a" stop-opacity="0"/></radialGradient>
    {SOFT.format(s=14)}
    <clipPath id="sq"><path d="{squircle(size)}"/></clipPath>
  </defs>
  <g clip-path="url(#sq)">
    <rect width="{size}" height="{size}" fill="url(#bg)"/>
    {grid}
    <circle cx="{size/2}" cy="{size*0.53}" r="{size*0.42}" fill="url(#glow)"/>
{knot_svg(size/2, size*0.53, size*0.118, size*0.062)}
  </g>
  <path d="{squircle(size)}" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="3"/>
</svg>
'''

def hero_svg(size=640):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="Arf">
  <title>Arf</title><desc>{DESC} A bead of light runs along the strand.</desc>
  <defs>
    {SOFT.format(s=10)}
    <filter id="bloom" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
    <radialGradient id="glow" cx="0.5" cy="0.52" r="0.5"><stop offset="0" stop-color="#ff6a2a" stop-opacity="0.30"/><stop offset="1" stop-color="#ff6a2a" stop-opacity="0"/></radialGradient>
  </defs>
  <style>
    .pulse {{ stroke-dasharray: 34 966; animation: run 5.5s linear infinite; }}
    @keyframes run {{ to {{ stroke-dashoffset: -1000; }} }}
    @media (prefers-reduced-motion: reduce) {{ .pulse {{ display: none; }} }}
  </style>
  <circle cx="{size/2}" cy="{size*0.52}" r="{size*0.46}" fill="url(#glow)"/>
{knot_svg(size/2, size*0.53, size*0.128, size*0.066, dark="#0b0f1c", shadow=True, pulse=True)}
</svg>
'''

def card_svg():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" role="img" aria-label="Arf: fast local models on Apple silicon">
  <title>Arf</title><desc>Social card: the trefoil mark and the drawn wordmark on a dark ground, with the tagline.</desc>
  <defs>
    <radialGradient id="glow" cx="0.33" cy="0.42" r="0.5"><stop offset="0" stop-color="#ff6a2a" stop-opacity="0.28"/><stop offset="1" stop-color="#ff6a2a" stop-opacity="0"/></radialGradient>
    {SOFT.format(s=10)}
  </defs>
  <rect width="1280" height="640" fill="#0b0f1c"/>
  <rect width="1280" height="640" fill="url(#glow)"/>
  <g transform="translate(190,120) scale(1.55)">
{knot_svg(118, 126, 33, 16, dark="#0b0f1c", shadow=False)}
{word("#F4F1EC")}
  </g>
  <text x="640" y="535" text-anchor="middle" fill="#aab3c2" font-size="36"
        font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">Fast local models on Apple silicon · written from scratch in Rust</text>
</svg>
'''

if __name__ == "__main__":
    out = {
        "assets/arf-mark.svg": mark_svg("#0d1117"),
        "assets/arf-mark-light.svg": mark_svg("#ffffff"),
        "assets/arf-logo-light.svg": logo_svg("#ffffff", "#14110F", "light"),
        "assets/arf-logo-dark.svg": logo_svg("#0d1117", "#F4F1EC", "dark"),
        "assets/arf-icon.svg": icon_svg(),
        "assets/arf-lockup.svg": card_svg(),
        "site/knot.svg": hero_svg(),
    }
    for path, svg in out.items():
        open(path, "w").write(svg)
    print("wrote", ", ".join(out))
