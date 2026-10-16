# UIUC SE 423: Introduction to Mechatronics (Spring 2026)

## This repository is for generating animations for the class to be used in https://github.com/Marius-Juston/SE-423---Class-Material

This library uses the [Manim library](https://docs.manim.community/en/stable/index.html) for animation

# Installation

Need to have LaTeX installed

```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm
```

# Typeface (required)

All SE 423 material uses [Atkinson Hyperlegible Next](https://fonts.google.com/specimen/Atkinson+Hyperlegible+Next) (Braille Institute, SIL Open Font License), and every figure made here must too. The font files are in [`fonts/`](fonts/).

- **matplotlib:** call `course_style.apply()` (from `src/course_style.py`) after any `plt.style.use(...)` / `rcParams.update(...)` block. It registers the vendored fonts, so nothing needs installing. Text and math letters use Atkinson; Greek and large operators fall back to STIX Sans.
- **MATLAB (`other/`):** install `fonts/*.otf`; the scripts set it as the default axes/text font.
- **After regenerating** a figure, replace its copy in the course repo (`Lectures/Figures/svg/`) and in the slide deck that embeds it.

Run figure scripts from a scratch directory (they write to the working directory), for example:

```bash
cd /tmp/figs && uv run --project ~/GitHub/SE423Animations python ~/GitHub/SE423Animations/src/kalman_update_svgs.py
```