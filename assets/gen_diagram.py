#!/usr/bin/env python3
"""Generate the README's "How it works" diagram, for light and dark backgrounds
(run from the repository root): python3 assets/gen_diagram.py

It reads top to bottom, the way a request travels: the client, the server, the engine's three ways of not
doing work twice (the same three the site's knot explains), the GPU, and the tokens streaming back.
"""
THEMES = {
    "dark":  dict(ink="#eceff5", dim="#9aa2b6", faint="#6b7490", box="#141a30", line="#2b3456", eng="#0f1427"),
    "light": dict(ink="#14171f", dim="#525a6e", faint="#7d8599", box="#f6f7fa", line="#d5d9e3", eng="#fbfbfd"),
}
ACC = "#ff6e1e"
FONT = "ui-sans-serif,-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SF Mono,Menlo,Consolas,monospace"

def svg(t):
    W = 960
    def box(x, y, w, h, title, sub, fill, rx=14):
        ty = y + h/2 - 4 if sub else y + h/2 + 6
        out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{t["line"]}"/>'
               f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="650" fill="{t["ink"]}">{title}</text>')
        if sub:
            out += f'<text x="{x + w/2}" y="{y + h/2 + 20}" text-anchor="middle" font-family="{FONT}" font-size="14.5" fill="{t["dim"]}">{sub}</text>'
        return out
    def down(x, y1, y2, label=None):
        s = f'<path d="M{x} {y1} V{y2 - 8}" stroke="{t["faint"]}" stroke-width="1.6"/><path d="M{x - 5} {y2 - 10} L{x} {y2} L{x + 5} {y2 - 10}" fill="none" stroke="{t["faint"]}" stroke-width="1.6"/>'
        if label:
            s += f'<text x="{x + 14}" y="{(y1 + y2)/2 + 5}" font-family="{MONO}" font-size="13" fill="{t["faint"]}">{label}</text>'
        return s
    def card(x, y, n, name, title, l1, l2):
        w, h = 252, 150
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{t["box"]}" stroke="{t["line"]}"/>'
                f'<text x="{x + 20}" y="{y + 32}" font-family="{MONO}" font-size="12.5" letter-spacing="1.5" fill="{ACC}">{n} {name.upper()}</text>'
                f'<text x="{x + 20}" y="{y + 64}" font-family="{FONT}" font-size="18" font-weight="650" fill="{t["ink"]}">{title}</text>'
                f'<text x="{x + 20}" y="{y + 96}" font-family="{FONT}" font-size="14.5" fill="{t["dim"]}">{l1}</text>'
                f'<text x="{x + 20}" y="{y + 119}" font-family="{FONT}" font-size="14.5" fill="{t["dim"]}">{l2}</text>')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 640" role="img" aria-label="How Arf works: a client calls arf-serve; the engine reuses shared prompts, batches conversations and drafts tokens; the GPU runs the step; tokens stream back.">',
             '<title>How Arf works</title>']
    parts.append(box(250, 20, 400, 76, "Your agent or app", "Claude Code · OpenCode · any OpenAI client", t["box"]))
    parts.append(down(450, 96, 146, "OpenAI · Anthropic API on localhost:8080"))
    parts.append(box(250, 146, 400, 66, "arf-serve", "", t["box"]))
    parts.append(down(450, 212, 262))
    parts.append(f'<rect x="20" y="262" width="860" height="230" rx="18" fill="{t["eng"]}" stroke="{t["line"]}" stroke-dasharray="5 6"/>')
    parts.append(f'<text x="44" y="296" font-family="{MONO}" font-size="12.5" letter-spacing="1.5" fill="{t["faint"]}">THE ENGINE — THREE WAYS OF NOT DOING WORK TWICE</text>')
    parts.append(card(44, 318, "01", "Reuse", "Prefix cache", "Shared prompts and tools", "are read once."))
    parts.append(card(324, 318, "02", "Share", "Scheduler", "Conversations decode", "together, paged KV."))
    parts.append(card(604, 318, "03", "Guess", "Draft, then verify", "Seven tokens proposed,", "checked in one pass."))
    parts.append(down(450, 492, 542))
    parts.append(box(250, 542, 400, 76, "Native Metal kernels", "the whole step on your Mac's GPU", t["box"]))
    # tokens stream back: a return path on the right, up to the client
    parts.append(f'<path d="M650 580 H918 V58 H662" fill="none" stroke="{ACC}" stroke-width="1.6" stroke-dasharray="4 5"/>'
                 f'<path d="M670 53 L660 58 L670 63" fill="none" stroke="{ACC}" stroke-width="1.6"/>'
                 f'<text x="938" y="390" text-anchor="middle" font-family="{MONO}" font-size="13" fill="{ACC}" transform="rotate(-90 938 390)">tokens stream back</text>')
    parts.append('</svg>\n')
    return "".join(parts)

if __name__ == "__main__":
    for name, t in THEMES.items():
        open(f"assets/how-it-works-{name}.svg", "w").write(svg(t))
    print("wrote assets/how-it-works-dark.svg, assets/how-it-works-light.svg")
