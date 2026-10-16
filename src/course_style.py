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
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
FAMILY = "Atkinson Hyperlegible Next"


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
    })
