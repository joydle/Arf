# Brand

The mark is a **trefoil knot** — the simplest knot whose Arf invariant is 1 — drawn as a lit 3D tube from
the knot's own equations, every crossing over and under, beside a drawn (not typeset) wordmark. The project
is named for [Cahit Arf](https://en.wikipedia.org/wiki/Cahit_Arf) (1910–1997).

| file | use |
|---|---|
| `arf-logo-light.svg` / `arf-logo-dark.svg` | the logo, for light / dark backgrounds (the README picks by `prefers-color-scheme`) |
| `arf-mark.svg` / `arf-mark-light.svg` | the mark alone; the crossing gaps match a dark / a light background |
| `arf-icon.svg` | the app icon: the mark in a rounded square on a dark ground |
| `arf-lockup.svg`, `social-preview.png` | 1280x640 card with the tagline (GitHub: Settings > Social preview) |
| `arf-mark-512.png`, `arf-mark-1024.png` | square rasters of the mark |
| `favicon.ico` (16/32/48), `favicon-32.png`, `apple-touch-icon.png` (180), `avatar-400.png` | from the app icon |

Everything is generated — do not hand-edit the SVGs. `assets/gen_logo.py` also writes the site's animated
hero (`site/knot.svg`):

```sh
python3 assets/gen_logo.py            # from the repo root
cd assets
rsvg-convert -w 512 arf-mark.svg -o arf-mark-512.png && rsvg-convert -w 1024 arf-mark.svg -o arf-mark-1024.png
rsvg-convert -w 180 arf-icon.svg -o apple-touch-icon.png && rsvg-convert -w 400 arf-icon.svg -o avatar-400.png
rsvg-convert -w 1280 arf-lockup.svg -o social-preview.png && rsvg-convert -w 32 arf-icon.svg -o favicon-32.png
for s in 16 48; do rsvg-convert -w $s arf-icon.svg -o /tmp/fav$s.png; done
magick /tmp/fav16.png favicon-32.png /tmp/fav48.png favicon.ico
```
