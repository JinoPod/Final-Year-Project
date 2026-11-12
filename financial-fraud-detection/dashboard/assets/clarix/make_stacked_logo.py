"""Build the stacked CLARIX logo (shield above the wordmark, no tagline) from the
original horizontal logo files, reusing their exact shapes and colours.

    python dashboard/assets/clarix/make_stacked_logo.py

Writes clarix-logo-stacked-navy.svg (navy background, like clarix-logo-navy.svg)
and clarix-logo-stacked-transparent.svg (for light backgrounds).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH, HEIGHT = 440, 272

# Measured in the original 880x264 artwork.
SHIELD_SCALE = 2.625      # the shield group's scale; its 64-unit grid spans x 12..52, y 6..58
WORD_X, WORD_Y, WORD_W = 260.3, 79.8, 383.5   # "CLARI" + "X" paths
UNDERLINE_W = 120


def build(source, target, keep_background):
    svg = open(os.path.join(HERE, source), encoding="utf-8").read()
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    background = re.search(r'<rect width="880" height="264" fill="([^"]+)"/>', svg)
    shield = re.search(r'<g transform="translate\(48 48\) scale\(2\.625\)"(.*?)</g>', svg, re.S).group(1)
    paths = re.findall(r'<path fill="[^"]+" d="[^"]+"/>', svg)
    letters, x_mark = paths[0], paths[1]  # paths[2] is the tagline, left out
    line = re.search(r'<line [^>]*stroke="([^"]+)" stroke-width="([^"]+)"/>', svg)

    cx = WIDTH / 2
    shield_tx = cx - 32 * SHIELD_SCALE
    shield_ty = 14 - 6 * SHIELD_SCALE + 4
    word_dx = cx - (WORD_X + WORD_W / 2)
    word_dy = 186 - WORD_Y
    underline_y = 186 + 57 + 18

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="Clarix">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{background.group(1)}"/>' if keep_background and background else "",
        f'<g transform="translate({shield_tx:.2f} {shield_ty:.2f}) scale({SHIELD_SCALE})"{shield}</g>',
        f'<g transform="translate({word_dx:.2f} {word_dy:.2f})">{letters}{x_mark}</g>',
        f'<line x1="{cx - UNDERLINE_W / 2}" y1="{underline_y}" x2="{cx + UNDERLINE_W / 2}" '
        f'y2="{underline_y}" stroke="{line.group(1)}" stroke-width="{line.group(2)}"/>',
        "</svg>",
    ]
    open(os.path.join(HERE, target), "w", encoding="utf-8").write("\n".join(p for p in parts if p))
    print("wrote", target)


if __name__ == "__main__":
    build("clarix-logo-navy.svg", "clarix-logo-stacked-navy.svg", keep_background=True)
    build("clarix-logo-transparent.svg", "clarix-logo-stacked-transparent.svg", keep_background=False)
