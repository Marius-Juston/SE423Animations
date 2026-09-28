# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UIUC SE 423: Introduction to Mechatronics (Spring 2026) animation library. Generates educational animations using [Manim](https://docs.manim.community/en/stable/index.html) for use in the companion course material repo.

## Setup

**LaTeX (required for Manim text rendering):**
```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm
```

**Python dependencies (Python 3.12):**
```bash
uv sync
```

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

### Resource Files
- `resource/CIE_std_illum_D65.csv` and `resource/CIE_xyz_1931_2deg.csv` — CIE color matching data used by `color_space.py`.
- `example/` — Reference SVG files for ePWM timing diagrams.

## Key Patterns

- Manim animations are defined entirely in `Scene.construct()` using mobjects and `self.play()`/`self.add()` calls.
- Circuit simulation uses two-phase update (evaluate then commit) to avoid race conditions between components.
- Visual circuit components use `LocalCoordinate` to anchor Manim positions to logical circuit positions, enabling smooth animations when circuit state changes.
