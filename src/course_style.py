"""Course typeface for every matplotlib figure: Atkinson Hyperlegible Next.

The SE 423 material (slides, handouts, site) is set in Atkinson Hyperlegible Next
(Braille Institute, SIL OFL 1.1) for readability. The OTF files are vendored in
the repo-root ``fonts/`` directory, so figures render the same on any machine.

Call ``apply()`` after any ``plt.style.use(...)`` / ``rcParams.update(...)``
block, since those can reset the font settings::

    import course_style
    course_style.apply()

Text and mathtext letters/digits use Atkinson; glyphs it lacks (Greek, large
operators) fall back to STIX Sans so math stays sans-serif.

The default colour cycle is colour-blind safe (Okabe & Ito palette) and pairs each
colour with its own line style, so series stay distinguishable without colour
(protanopia, deuteranopia, tritanopia, greyscale printing). Scripts that set
explicit colours keep them; pick those from ``CVD_COLORS`` too. The course repo's
``tools/a11y/cvd_check.py`` flags figures whose colours collide.
"""
from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
FAMILY = "Atkinson Hyperlegible Next"

# Okabe & Ito (2008) colour-blind-safe palette, without its yellow (too light on white).
CVD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
CVD_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]


def apply():
    for path in sorted(FONT_DIR.glob("*.otf")):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FAMILY, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FAMILY,
        "mathtext.sf": FAMILY,
        "mathtext.it": f"{FAMILY}:italic",
        "mathtext.bf": f"{FAMILY}:bold",
        "mathtext.cal": f"{FAMILY}:italic",
        "mathtext.fallback": "stixsans",
        "svg.fonttype": "path",
        # Colour plus line style: never colour alone (WCAG 1.4.1).
        "axes.prop_cycle": cycler(color=CVD_COLORS) + cycler(linestyle=CVD_LINESTYLES),
    })
