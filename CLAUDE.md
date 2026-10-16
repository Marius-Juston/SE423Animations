# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UIUC SE 423: Introduction to Mechatronics (Spring 2026) animation library. Generates educational animations using [Manim](https://docs.manim.community/en/stable/index.html) and static matplotlib figures for the companion course material repo, [SE-423---Class-Material](https://github.com/Marius-Juston/SE-423---Class-Material) (usually checked out at `~/GitHub/SE-423---Class-Material`).

## Setup

**LaTeX (required for Manim text rendering):**
```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm
```

**Python dependencies (Python 3.12):**
```bash
uv sync
```

## Generating Figures (matplotlib)

Figure scripts in `src/*.py` write their outputs to the **current working directory** and some call `plt.show()`. Run them from a scratch directory, headless:

```bash
mkdir -p /tmp/figs && cd /tmp/figs
MPLBACKEND=Agg uv run --project ~/GitHub/SE423Animations python ~/GitHub/SE423Animations/src/kalman_update_svgs.py
```

Gotchas:
- `white_noise.py` draws its three plots into the same figure and relies on an interactive `plt.show()` to clear it between them; headless, patch `plt.show = lambda *a, **k: plt.close('all')` before running it (e.g. via `runpy`), or the later plots accumulate earlier lines.
- `color_space.py` reads `../resource/*.csv`, so run it from a directory whose parent contains `resource/` (or symlink it), and it renders a slow 3-D GIF.
- Every figure script must call `course_style.apply()` (Atkinson Hyperlegible Next) **after** its own `plt.style.use(...)`/`rcParams.update(...)` block, since those reset fonts. Don't set `font.family` or `mathtext.fontset` elsewhere.

Where the outputs are used (copy them into the course repo's `Lectures/Figures/svg/`, then run `python3 tools/sync_deck_figures.py sync` there to update the decks):

| Script | Outputs | Used in |
|---|---|---|
| `white_noise.py` | `white_noise.svg`, `white_noise_mean.svg`, `white_std.svg` | L15 |
| `gaussians.py` | `gaussians.svg` | L20-21 |
| `kalman_update_svgs.py` | `01_initial_belief.svg` … `04_fused_update.svg` | L20-21 |
| `covariance_viz.py` | `05_covariance_matrix.svg` | L20-21 |
| `multivariate_kalman_filter.py` | `MD_01_initial_belief.svg` … `MD_04_fused_update.svg` | L20-21 |
| `gaussian_transform.py` | `gaussian_transform.svg` | L20-21 |
| `nonlinear_covariance.py` | `monte_carlo_only.svg`, `monte_carlo_ekf.svg`, `monte_carlo_ekf_lie.svg` | L20-21 |
| `monte_carlo.py` | `slide1_init.svg` … `slide7_resample2.svg` | L24-25 |
| `occupancy_grid.py` | `lidar_grid.svg` | L24-25 |
| `huber_loss.py` | `huber_loss.svg` | L24-25 |

The course repo's `Lectures/Slides/figures.toml` is the authoritative map from each slide picture to its source file.

`other/FIR_full_filter.m` (MATLAB) sets the course font as the default axes/text/legend font; the font must be installed (`fonts/*.otf`).

## Rendering Animations

Each animation module lives in `src/<module>/` with its own `manim.cfg`. To render:

```bash
# From the module directory (picks up manim.cfg automatically)
cd src/pwm && manim main.py PWMAnimation

# Or from project root with explicit flags
manim -pql src/pwm/main.py PWMAnimation
```

Common flags: `-p` (preview/autoplay), `-ql` (low quality, fast), `-qh` (1080p), `-qk` (4K), `-s` (save last frame only).

Each `manim.cfg` defines the default scene, resolution, and frame rate for that module, so `manim main.py` with no scene name will use those defaults.

Manim `Text(...)` still uses the system default font (not yet switched to Atkinson Hyperlegible Next); switching requires re-rendering the videos.

## Architecture

### Animation Modules (`src/<name>/`)
Each module is a self-contained Manim scene:
- `pwm/` — PWM duty cycle effect on LED brightness
- `blob_detection/` — BFS blob detection with moments computation
- `gpio-led/` — GPIO + NOT gate + LED circuit
- `two-compliment/` — Two's complement arithmetic
- `ti-cpu-timer/` — TI CPU timer counter/prescaler
- `color_spaces/` — 17-scene color science series: EM spectrum → human vision → CIE 1931 → RGB/HSV → CIELAB → OKLab. Has its own `render_all.py` to render and ffmpeg-concatenate all scenes into one video.
- `path_planning/` — 10-scene pathfinding series: `TitleScene` → `GraphBasicsScene` → `MapRepresentationsScene` → `BFSScene` → `EarlyExitScene` → `DijkstraScene` → `GreedyBFSScene` → `AStarScene` → `ComparisonScene` → `SummaryScene`. All algorithm logic (BFS, Dijkstra, A*) is self-contained in `main.py` alongside the scenes; uses `GridWorld`/`WeightedGrid` helpers and a shared color palette of design tokens.

Each scene is a class inheriting from `manim.Scene` with a `construct()` method.

Multi-scene modules with many scenes use a `render_all.py` script (see `color_spaces/`):
```bash
python render_all.py              # 1080p/60fps (default)
python render_all.py --quality l  # low quality preview
python render_all.py --quality k  # 4K
```

### Shared Library (`src/`)
- **`circuit.py`** — Event-driven digital circuit simulator. `Circuit` manages a component graph; components implement `evaluate()` (read inputs) and `commit()` (write outputs) phases per time step. Includes `Clock`, `AND`, `OR`, `DownCounter`, etc.
- **`visual_circuit.py`** — Manim visualization layer for circuits. `CircuitShape` is the base visual element; `VisualGroup` composes them; `LocalCoordinate` tracks coordinate transforms for animations.
- **`ti_timer.py`** — TI-specific circuit component definitions (`Register`, `CPUTimerCounter`, prescaler logic).
- **`router.py`** — A* based PCB trace routing with congestion-aware costs.
- **`svg_animator.py`** — SVG manipulation utilities (e.g., culling elements outside viewbox).
- **`color_space.py`** — CIE color science (XYZ↔Lab conversions using `resource/` CSV data). Note: `color_spaces/main.py` contains a more complete, self-contained color math engine (sRGB gamma, XYZ↔sRGB, CIELAB forward+inverse, OKLab) that does not depend on this file.
- **`icp.py`** — Iterative Closest Point algorithm for point cloud registration (SVD-based).
- **`white_noise.py`** — Standalone matplotlib/seaborn white noise statistical visualization.
- **`course_style.py`** — `apply()` sets the course typeface, Atkinson Hyperlegible Next (vendored in `fonts/`), for matplotlib text and mathtext. Every figure script must call it after its own style/rcParams block.
- **`huber_loss.py`** — Squared vs. Huber loss plot used in the SLAM lecture (L24-25).

### Resource Files
- `resource/CIE_std_illum_D65.csv` and `resource/CIE_xyz_1931_2deg.csv` — CIE color matching data used by `color_space.py`.
- `example/` — Reference SVG files for ePWM timing diagrams.

## Key Patterns

- Manim animations are defined entirely in `Scene.construct()` using mobjects and `self.play()`/`self.add()` calls.
- All figure text uses Atkinson Hyperlegible Next (see `course_style.py`); figure scripts write to the working directory, so run them from a scratch directory and copy the outputs into the course repo (`Lectures/Figures/svg/` and the deck that embeds them).
- Circuit simulation uses two-phase update (evaluate then commit) to avoid race conditions between components.
- Visual circuit components use `LocalCoordinate` to anchor Manim positions to logical circuit positions, enabling smooth animations when circuit state changes.
