"""
SE 423 – Blob Detection & Connected-Component Labeling
Comprehensive educational animation covering the full pipeline:
  CMOS → Bayer CFA → Demosaic → Color Space → Threshold
  → BFS/CCL → Image Moments → Shape Filtering
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import colorsys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from manim import *

from visual_circuit import (
    VisualNMOS, VisualPhotodiode, VisualVDD, VisualGND,
    VisualCapacitor, VisualJunction, VisualWire,
    same_x, same_y,
)

# ─── Shared colour palette ────────────────────────────────────────────────────
BAYER_R = ManimColor([0.90, 0.20, 0.21, 1.0])
BAYER_G = ManimColor([0.26, 0.63, 0.28, 1.0])
BAYER_B = ManimColor([0.12, 0.47, 0.71, 1.0])

BLOB_PALETTE = [
    ManimColor([0.13, 0.59, 0.95, 1.0]),  # blue
    ManimColor([0.30, 0.69, 0.31, 1.0]),  # green
    ManimColor([0.00, 0.74, 0.83, 1.0]),  # cyan
    ManimColor([1.00, 0.76, 0.03, 1.0]),  # amber
    ManimColor([0.88, 0.25, 0.98, 1.0]),  # purple
]

SEC_TITLE_COLOR = YELLOW
NOTE_COLOR = ManimColor([0.75, 0.75, 0.75, 1.0])


def _hsv(h: float, s: float = 1.0, v: float = 1.0) -> ManimColor:
    """Convert HSV (all 0–1) to ManimColor."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return ManimColor([r, g, b, 1.0])


# ─── Pixel helper (BFS / threshold sections) ─────────────────────────────────
class Pixel(VGroup):
    def __init__(self, text: int = 0):
        self.s1 = Square()
        self.tracker = ValueTracker(float(text))
        self.value = Integer(int(self.tracker.get_value())).scale(1.5)
        self.value.add_updater(
            lambda v: v.set_value(int(self.tracker.get_value()))
        ).align_to(self.s1)
        super().__init__(self.s1, self.value)

    def color_pixel(self):
        v = max(0.0, min(1.0, self.tracker.get_value() / 255.0))
        return self.s1.animate.set_fill(
            color=ManimColor([v, v, v, 1.0]), opacity=0.85
        )

    def mask_pixel(self):
        if self.tracker.get_value() >= 0.5:
            return self.s1.animate.set_fill(
                color=ManimColor([1.0, 1.0, 1.0, 1.0]), opacity=0.9
            )
        return self.s1.animate.set_fill(
            color=ManimColor([0.08, 0.08, 0.08, 1.0]), opacity=0.9
        )

    def set_value(self, value):
        return self.tracker.animate(run_time=0).set_value(float(value))

    def set_square_color(self, color: ManimColor, run_time: float = 0.12):
        return self.s1.animate(run_time=run_time).set_fill(
            color=color, opacity=0.75
        )


# ─── Main Scene ───────────────────────────────────────────────────────────────
class BlobDetection(Scene):
    # Shared binary image used across multiple sections
    BINARY = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    def construct(self):
        np.random.seed(0)
        for section in [
            self._section_cmos,
            self._section_bayer,
            self._section_demosaic,
            self._section_colorspace,
            self._section_threshold,
            self._section_bfs,
            self._section_moments,
            self._section_shape_filter,
        ]:
            section()
            self._fade_all_out()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _section_title(self, title: str, subtitle: str = "") -> VGroup:
        t = Text(title, font_size=52, color=SEC_TITLE_COLOR, weight=BOLD)
        items = [t]
        if subtitle:
            s = Text(subtitle, font_size=30, color=NOTE_COLOR)
            items.append(s)
        grp = VGroup(*items).arrange(DOWN, buff=0.25).move_to(ORIGIN)
        self.play(FadeIn(grp, shift=UP * 0.2), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(grp), run_time=0.4)

    def _fade_all_out(self):
        for mob in self.mobjects:
            mob.clear_updaters()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.5)
        self.clear()

    def _block(
        self,
        label: str,
        width: float = 2.2,
        height: float = 0.75,
        color=BLUE_D,
        font_size: int = 22,
    ) -> VGroup:
        rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            fill_color=color,
            fill_opacity=0.9,
            stroke_color=WHITE,
            stroke_width=1.5,
        )
        txt = Text(label, font_size=font_size, color=WHITE)
        if txt.width > rect.width - 0.15:
            txt.scale_to_fit_width(rect.width - 0.15)
        txt.move_to(rect.get_center())
        return VGroup(rect, txt)

    def _make_pixel_grid(
        self, array: np.ndarray, scale_frac: float = 0.48
    ) -> tuple[list[Pixel], VGroup]:
        H, W = array.shape
        idx = np.arange(array.size)
        xI, yI = idx // W, idx % W
        value_arr = array.copy().astype(np.uint8)
        mask_bg = value_arr == 0
        value_arr[mask_bg] = np.random.randint(0, 50, int(mask_bg.sum()), dtype=np.uint8)
        value_arr[~mask_bg] = np.random.randint(100, 255, int((~mask_bg).sum()), dtype=np.uint8)

        pixels = [Pixel(int(value_arr[x, y])) for x, y in zip(xI, yI)]
        grid = VGroup(*pixels)
        grid.arrange_in_grid(rows=H, cols=W, buff=0)
        sf = min(
            config.frame_width * scale_frac / grid.width,
            config.frame_height * scale_frac / grid.height,
        )
        grid.scale(sf).move_to(ORIGIN)
        return pixels, grid

    # ── Section 1: CMOS Image Sensor ─────────────────────────────────────────
    def _section_cmos(self):
        self._section_title(
            "Step 1: CMOS Image Sensor",
            "Converting photons to digital pixel values",
        )
        self._cmos_phase_a_block_diagram()
        self._fade_all_out()
        self._cmos_phase_b_pixel_circuit()
        self._fade_all_out()
        self._cmos_phase_c_subarray_readout()
        self.wait(2)

    def _cmos_phase_a_block_diagram(self):
        """System-level block diagram matching TikZ CMOS sensor architecture."""
        BW, BH = 1.9, 0.72

        b_arr     = self._block("Pixel Array\n(M × N pixels)", 2.6, 1.8, BLUE_D, font_size=20)
        b_row_acc = self._block("Row Access\nLogic",            BW,  BH, TEAL_D)
        b_row_drv = self._block("Row Drivers",                  BW,  BH, TEAL_D)
        b_clk     = self._block("Clock / Timing\nOscillator",   BW,  BH, PURPLE_D)
        b_bias    = self._block("Bias\nGenerator",              BW,  BH, TEAL_E)
        b_colbias = self._block("Column Bias &\nSampling",      2.6, BH, BLUE_E)
        b_colmux  = self._block("Column Mux",                   2.6, BH, BLUE_E)
        b_adc     = self._block("Gain / ADC /\nLine Driver",    BW,  1.9, GREEN_D)

        b_arr.move_to(ORIGIN)
        b_row_acc.next_to(b_arr, LEFT, buff=0.65).shift(UP * 0.38)
        b_row_drv.next_to(b_row_acc, DOWN, buff=0.20)
        b_clk.next_to(b_arr, UP, buff=0.45).align_to(b_arr, RIGHT)
        b_bias.next_to(b_row_drv, DOWN, buff=0.22)
        b_colbias.next_to(b_arr, DOWN, buff=0.45)
        b_colmux.next_to(b_colbias, DOWN, buff=0.20)
        b_adc.next_to(b_arr, RIGHT, buff=0.65)

        all_blks = VGroup(b_arr, b_row_acc, b_row_drv, b_clk, b_bias,
                          b_colbias, b_colmux, b_adc)
        # Scale to ~38 % frame width so PCB rect + external labels all fit comfortably
        sf = min(config.frame_width * 0.38 / all_blks.width,
                 config.frame_height * 0.50 / all_blks.height)
        all_blks.scale(sf).move_to(ORIGIN + UP * 0.1)

        pcb_rect = DashedVMobject(
            SurroundingRectangle(all_blks, buff=0.32, color=ORANGE,
                                 corner_radius=0.09, stroke_width=1.5),
            num_dashes=40,
        )
        pcb_lbl = Text("Camera PCB", font_size=18, color=ORANGE)
        pcb_lbl.next_to(pcb_rect, UP, buff=0.10)

        def _arr(a, b, color=WHITE):
            return Arrow(a, b, buff=0.05, stroke_width=2.5, color=color,
                         max_tip_length_to_length_ratio=0.3)

        arrows = VGroup(
            _arr(b_row_acc.get_right(), b_arr.get_left() + UP * 0.25,   BLUE_B),
            _arr(b_row_drv.get_right(), b_arr.get_left() + DOWN * 0.25, GREEN_B),
            _arr(b_clk.get_left(),      b_arr.get_top() + RIGHT * 0.3,  YELLOW_B),
            _arr(b_arr.get_bottom(),    b_colbias.get_top()),
            _arr(b_colbias.get_bottom(), b_colmux.get_top()),
            _arr(b_colmux.get_right(),  b_adc.get_bottom() + LEFT * 0.25),
        )

        # Labels anchored to pcb_rect edges — guaranteed outside the dashed box
        row_bus_lbl = Text("Row Buses\n(RST, SEL)", font_size=14, color=BLUE_B)
        row_bus_lbl.next_to(pcb_rect, LEFT, buff=0.12)
        row_bus_lbl.set_y(VGroup(b_row_acc, b_row_drv).get_center()[1])

        col_bus_lbl = Text("Column\nBuses", font_size=14, color=NOTE_COLOR)
        col_bus_lbl.next_to(pcb_rect, DOWN, buff=0.12)
        col_bus_lbl.set_x(b_colmux.get_center()[0])

        out_lbl = Text("Digital\nOutput", font_size=18, color=GREEN_B)
        out_lbl.next_to(b_adc, RIGHT, buff=0.55)
        out_arr = _arr(b_adc.get_right(), out_lbl.get_left(), GREEN_B)

        blocks_list = [b_arr, b_row_acc, b_row_drv, b_clk, b_bias, b_colbias, b_colmux, b_adc]
        self.play(LaggedStart(*[FadeIn(b) for b in blocks_list], lag_ratio=0.12), run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1))
        self.play(Create(pcb_rect), Write(pcb_lbl))
        self.play(FadeIn(row_bus_lbl), FadeIn(col_bus_lbl), GrowArrow(out_arr), Write(out_lbl))
        self.wait(1.5)

    def _cmos_phase_b_pixel_circuit(self):
        """Transistor-level 3T pixel: PD, RST NMOS, SF NMOS, SEL NMOS with signal animation."""
        title = Text("Single 3T Pixel: Transistor Circuit", font_size=34, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        SCALE = 1.3
        # Pin offsets after scale(SCALE) for VisualNMOS unit geometry
        D_EXT = (0.28 + 0.27) * SCALE   # 0.715  drain/source y offset from body center
        D_X   = 0.18 * SCALE             # 0.234  drain/source x offset (right of body)
        G_X   = 0.42 * SCALE             # 0.546  gate x offset (left of body)
        PD_H  = 0.38 * SCALE             # 0.494  PD cathode/anode y from PD center
        CAP_H = 0.32 * SCALE             # 0.416  capacitor pos/neg terminal y from center

        Y_FD  =  0.5   # Floating Diffusion node y (world units)
        X_RST = -2.5   # RST body center x
        X_SF  =  0.8   # SF / SEL body center x
        X_CAP = -1.4   # C_FD x

        # ── Instantiate ─────────────────────────────────────────────────────
        pd      = VisualPhotodiode(show_photons=True).scale(SCALE)
        rst     = VisualNMOS(label="RST").scale(SCALE)
        sf      = VisualNMOS(label="SF").scale(SCALE)
        sel     = VisualNMOS(label="SEL").scale(SCALE)
        c_fd    = VisualCapacitor(label="C_FD").scale(SCALE)
        vdd_rst = VisualVDD(label="VDD").scale(SCALE)
        vdd_sf  = VisualVDD(label="VDD").scale(SCALE)
        gnd_pd  = VisualGND().scale(SCALE)
        gnd_cap = VisualGND().scale(SCALE)
        fd_junc  = VisualJunction()
        cap_junc = VisualJunction()

        # ── Place: shift each component so key pin lands at target coord ───
        # RST source → FD node  (source is at (D_X, -D_EXT) before shift)
        rst.shift(np.array([X_RST, Y_FD + D_EXT, 0]))
        # PD cathode → FD node  (cathode is at (0, PD_H) before shift)
        pd.shift(np.array([X_RST + D_X, Y_FD - PD_H, 0]))
        # SF: gate y = Y_FD, body center x = X_SF  (gate at (-G_X, 0) before shift)
        sf.shift(np.array([X_SF, Y_FD, 0]))
        # SEL: drain = SF source  (drain at (D_X, D_EXT) before shift)
        sel.shift(np.array([X_SF, Y_FD - 2 * D_EXT, 0]))
        # C_FD: pos_terminal → FD level  (pos at (0, CAP_H) before shift)
        c_fd.shift(np.array([X_CAP, Y_FD - CAP_H, 0]))
        # VDD symbols: terminal at (0,0) before shift
        vdd_rst.shift(np.array([X_RST + D_X, Y_FD + 2 * D_EXT, 0]))
        vdd_sf.shift(np.array([X_SF + D_X,   Y_FD + D_EXT,     0]))
        # GND symbols: terminal at (0,0) before shift
        gnd_pd.shift(np.array([X_RST + D_X, Y_FD - 2 * PD_H,   0]))
        gnd_cap.shift(np.array([X_CAP,       Y_FD - 2 * CAP_H,  0]))
        # Junctions
        fd_junc.move_to([X_RST + D_X, Y_FD, 0])
        cap_junc.move_to([X_CAP,      Y_FD, 0])

        # ── Wires ───────────────────────────────────────────────────────────
        # FD bus: FD node → SF gate  (horizontal at Y_FD)
        w_fd = VisualWire([
            [X_RST + D_X,  Y_FD, 0],
            [X_SF  - G_X,  Y_FD, 0],
        ])
        # RST row line
        x_rst_gate  = X_RST - G_X          # -3.046
        y_rst_ctr   = Y_FD + D_EXT         # 1.215
        w_rst_row = VisualWire([
            [x_rst_gate - 0.9, y_rst_ctr, 0],
            [x_rst_gate,       y_rst_ctr, 0],
        ])
        rst_row_lbl = Text("RST row", font_size=13, color=GREY)
        rst_row_lbl.move_to([x_rst_gate - 1.35, y_rst_ctr, 0])
        # SEL row line
        x_sel_gate  = X_SF - G_X           # 0.254
        y_sel_ctr   = Y_FD - 2 * D_EXT    # -0.93
        w_sel_row = VisualWire([
            [x_sel_gate - 1.3, y_sel_ctr, 0],
            [x_sel_gate,       y_sel_ctr, 0],
        ])
        sel_row_lbl = Text("SEL row", font_size=13, color=GREY)
        sel_row_lbl.move_to([x_sel_gate - 1.75, y_sel_ctr, 0])
        # Column bus
        y_colbus = Y_FD - 3 * D_EXT        # -1.645
        w_colbus = VisualWire([
            [X_SF - 0.4,      y_colbus, 0],
            [X_SF + D_X + 1.0, y_colbus, 0],
        ])
        colbus_lbl = Text("Column Bus", font_size=13, color=GREY)
        colbus_lbl.move_to([X_SF + D_X + 1.65, y_colbus, 0])
        colbus_junc = VisualJunction().move_to([X_SF + D_X, y_colbus, 0])

        # FD label — below-right of junction dot, away from photon arrows (which
        # sit left/above the PD cathode at approximately x=-2.5..−2.2, y=0.4..0.95)
        fd_lbl = Text("FD", font_size=15, color=YELLOW)
        fd_lbl.move_to([X_RST + D_X + 0.45, Y_FD - 0.28, 0])

        # ── Assemble, scale, center ─────────────────────────────────────────
        circuit_grp = VGroup(
            pd, rst, sf, sel, c_fd,
            vdd_rst, vdd_sf, gnd_pd, gnd_cap,
            fd_junc, cap_junc, colbus_junc,
            w_fd, w_rst_row, w_sel_row, w_colbus,
            rst_row_lbl, sel_row_lbl, colbus_lbl, fd_lbl,
        )
        scale_f = min(
            config.frame_width  * 0.68 / circuit_grp.width,
            config.frame_height * 0.70 / circuit_grp.height,
        )
        circuit_grp.scale(scale_f).move_to(ORIGIN + DOWN * 0.4)

        self.play(Create(circuit_grp), run_time=2.5)
        self.wait(0.4)

        # ── Signal animation ─────────────────────────────────────────────────
        def _step(text, *anims):
            lbl = Text(text, font_size=19, color=NOTE_COLOR).to_edge(DOWN, buff=0.3)
            self.play(Write(lbl), run_time=0.4)
            self.play(*anims)
            self.wait(0.5)
            self.play(FadeOut(lbl), run_time=0.25)

        _step("① Photon hits photodiode → charge accumulates at FD",
              Flash(pd, color=YELLOW, flash_radius=0.35 * scale_f),
              *pd.set_active(True),
              w_fd.set_active(True),
              fd_junc.set_active(True),
              cap_junc.set_active(True))

        _step("② RST pulse resets FD node to VDD",
              w_rst_row.set_active(True),
              rst.animate.set_stroke(color=YELLOW))

        _step("③ SF (Source Follower) buffers FD voltage",
              sf.animate.set_stroke(color=YELLOW))

        _step("④ SEL enables row → drives Column Bus → ADC digitizes",
              w_sel_row.set_active(True),
              sel.animate.set_stroke(color=YELLOW),
              w_colbus.set_active(True),
              colbus_junc.set_active(True))

        self.wait(1)

    def _cmos_phase_c_subarray_readout(self):
        """3×3 pixel array showing row-by-row rolling shutter readout."""
        title = Text("Pixel Array: Rolling Shutter Readout", font_size=34, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        ROWS, COLS = 3, 3
        CELL = 1.0
        GAP  = 0.08
        cell_sz = CELL + GAP

        # Pixel cells
        cells = []
        for r in range(ROWS):
            row = []
            for c in range(COLS):
                sq = Square(
                    side_length=CELL,
                    fill_color=ManimColor([0.08, 0.08, 0.18, 1.0]),
                    fill_opacity=0.9,
                    stroke_color=GREY, stroke_width=1.5,
                )
                sq.move_to([c * cell_sz, -r * cell_sz, 0])
                lbl = Text("PD", font_size=14, color=NOTE_COLOR).move_to(sq.get_center())
                row.append(VGroup(sq, lbl))
            cells.append(row)

        all_cells = VGroup(*[cell for row in cells for cell in row])

        # Convenience positions
        col_xs    = [c * cell_sz for c in range(COLS)]
        row_ys    = [-r * cell_sz for r in range(ROWS)]
        g_left    = min(col_xs) - CELL / 2
        g_right   = max(col_xs) + CELL / 2
        g_top     = row_ys[0] + CELL / 2
        g_bottom  = row_ys[-1] - CELL / 2

        # Column buses (vertical below grid)
        col_buses = [
            Line([x, g_bottom, 0], [x, g_bottom - 0.9, 0],
                 stroke_color=GREY, stroke_width=2)
            for x in col_xs
        ]
        adc_lbl = Text("↓ ADC", font_size=16, color=GREEN_B)
        adc_lbl.move_to([col_xs[COLS // 2], g_bottom - 1.25, 0])

        # RST + SEL row buses (horizontal, left of grid)
        rst_buses, sel_buses = [], []
        for r in range(ROWS):
            y = row_ys[r]
            rst_buses.append(
                Line([g_left - 1.4, y + CELL * 0.2, 0],
                     [g_left,       y + CELL * 0.2, 0],
                     stroke_color=GREY, stroke_width=2))
            sel_buses.append(
                Line([g_left - 1.4, y - CELL * 0.2, 0],
                     [g_left,       y - CELL * 0.2, 0],
                     stroke_color=GREY, stroke_width=2))

        rst_bus_lbl = Text("RST", font_size=14, color=GREY)
        rst_bus_lbl.move_to([g_left - 1.8, row_ys[0] + CELL * 0.2, 0])
        sel_bus_lbl = Text("SEL", font_size=14, color=GREY)
        sel_bus_lbl.move_to([g_left - 1.8, row_ys[0] - CELL * 0.2, 0])

        row_labels = VGroup(*[
            Text(f"Row {r}", font_size=15, color=NOTE_COLOR)
            .move_to([g_right + 0.8, row_ys[r], 0])
            for r in range(ROWS)
        ])

        all_grp = VGroup(
            all_cells,
            VGroup(*col_buses), VGroup(*rst_buses), VGroup(*sel_buses),
            adc_lbl, rst_bus_lbl, sel_bus_lbl, row_labels,
        )
        sf = min(
            config.frame_width  * 0.62 / all_grp.width,
            config.frame_height * 0.62 / all_grp.height,
        )
        all_grp.scale(sf).move_to(ORIGIN + DOWN * 0.3)

        self.play(Create(all_cells), run_time=0.8)
        self.play(
            Create(VGroup(*col_buses, *rst_buses, *sel_buses)),
            FadeIn(row_labels), FadeIn(adc_lbl),
            FadeIn(rst_bus_lbl), FadeIn(sel_bus_lbl),
            run_time=0.8,
        )
        self.wait(0.3)

        note = Text(
            "Each row is read sequentially — Rolling Shutter",
            font_size=21, color=NOTE_COLOR,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(note))

        PX_ACTIVE   = BAYER_G
        PX_INACTIVE = ManimColor([0.08, 0.08, 0.18, 1.0])

        for r in range(ROWS):
            # Activate row buses + highlight pixels
            anims = [
                rst_buses[r].animate.set_color(YELLOW).set_stroke(width=4),
                sel_buses[r].animate.set_color(YELLOW).set_stroke(width=4),
            ]
            for c in range(COLS):
                anims.append(cells[r][c][0].animate.set_fill(PX_ACTIVE, opacity=0.9))
            self.play(*anims, run_time=0.5)
            # Column buses light up
            self.play(
                *[cb.animate.set_color(YELLOW).set_stroke(width=4) for cb in col_buses],
                run_time=0.3,
            )
            self.wait(0.35)
            # Deactivate
            deact = [
                rst_buses[r].animate.set_color(GREY).set_stroke(width=2),
                sel_buses[r].animate.set_color(GREY).set_stroke(width=2),
            ]
            for c in range(COLS):
                deact.append(cells[r][c][0].animate.set_fill(PX_INACTIVE, opacity=0.9))
            deact.extend([cb.animate.set_color(GREY).set_stroke(width=2) for cb in col_buses])
            self.play(*deact, run_time=0.3)

        self.wait(1)

    # ── Section 2: Bayer Color Filter Array ──────────────────────────────────
    def _section_bayer(self):
        self._section_title(
            "Step 2: Bayer Color Filter Array",
            "Each pixel captures only ONE color channel",
        )

        ROWS, COLS = 6, 6
        SIDE = 0.85

        def bayer_color(r: int, c: int):
            if r % 2 == 0 and c % 2 == 0:
                return BAYER_R, "R"
            elif r % 2 == 1 and c % 2 == 1:
                return BAYER_B, "B"
            return BAYER_G, "G"

        sq_list, lbl_list = [], []
        for r in range(ROWS):
            for c in range(COLS):
                color, letter = bayer_color(r, c)
                sq = Square(
                    side_length=SIDE,
                    fill_color=color,
                    fill_opacity=0.80,
                    stroke_color=WHITE,
                    stroke_width=1.5,
                )
                sq.move_to(RIGHT * c * SIDE + DOWN * r * SIDE)
                lbl = Text(letter, font_size=22, color=WHITE, weight=BOLD)
                lbl.move_to(sq.get_center())
                sq_list.append(sq)
                lbl_list.append(lbl)

        grid = VGroup(*sq_list, *lbl_list)
        grid.move_to(LEFT * 3.2 + UP * 0.2)

        grid_title = Text("RGGB Bayer Pattern", font_size=30, color=YELLOW)
        grid_title.next_to(grid, UP, buff=0.3)

        self.play(
            LaggedStart(*[FadeIn(s, l) for s, l in zip(sq_list, lbl_list)],
                        lag_ratio=0.04),
        )
        self.play(Write(grid_title))
        self.wait(0.4)

        # Highlight 2×2 unit cell (top-left)
        unit_rect = SurroundingRectangle(
            VGroup(*sq_list[:2], *sq_list[COLS:COLS + 2]),
            color=YELLOW, buff=0.05, stroke_width=3, corner_radius=0.05,
        )
        unit_lbl = Text("2×2 Unit Cell", font_size=22, color=YELLOW)
        unit_lbl.next_to(unit_rect, DOWN, buff=0.15)
        self.play(Create(unit_rect), Write(unit_lbl))
        self.wait(0.6)

        # ── Right: distribution bar chart ─────────────────────────────────────
        dist_title = Text("Pixel Distribution", font_size=28, color=WHITE)
        dist_title.move_to(RIGHT * 3.8 + UP * 2.8)

        baseline_y = 0.5
        bar_data = [
            ("R\n25%", 1.0, BAYER_R),
            ("G\n50%", 2.0, BAYER_G),
            ("B\n25%", 1.0, BAYER_B),
        ]
        bars = VGroup()
        for i, (txt, h, col) in enumerate(bar_data):
            bar = Rectangle(
                width=0.8,
                height=h,
                fill_color=col,
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=1,
            )
            bar.move_to(RIGHT * (2.6 + i * 1.2) + UP * (baseline_y + h / 2))
            bar_lbl = Text(txt, font_size=20, color=WHITE, line_spacing=1.1)
            bar_lbl.next_to(bar, DOWN, buff=0.1)
            bars.add(VGroup(bar, bar_lbl))

        self.play(Write(dist_title), FadeIn(bars))

        why_note = Text(
            "50% green pixels match the human\neye's peak green sensitivity",
            font_size=22,
            color=NOTE_COLOR,
            line_spacing=1.3,
        )
        why_note.next_to(bars, DOWN, buff=0.45)
        self.play(Write(why_note))

        raw_note = Text(
            "Raw sensor output: each pixel has R, G, OR B — not full colour!",
            font_size=24,
            color=ORANGE,
        )
        raw_note.to_edge(DOWN, buff=0.35)
        self.play(Write(raw_note))
        self.wait(2)

    # ── Section 3: Demosaicing ────────────────────────────────────────────────
    def _section_demosaic(self):
        self._section_title(
            "Step 3: Demosaicing",
            "Reconstruct full RGB from sparse Bayer samples",
        )

        SIDE = 1.0
        # 3×3 neighbourhood around a central Blue pixel:
        #   G  R  G
        #   R  B  R   ← centre: only B known
        #   G  R  G
        neighborhood = [
            (BAYER_G, "G"), (BAYER_R, "R"), (BAYER_G, "G"),
            (BAYER_R, "R"), (BAYER_B, "B"), (BAYER_R, "R"),
            (BAYER_G, "G"), (BAYER_R, "R"), (BAYER_G, "G"),
        ]

        cells, cell_lbls = [], []
        for idx, (col, letter) in enumerate(neighborhood):
            r, c = divmod(idx, 3)
            sq = Square(
                side_length=SIDE,
                fill_color=col,
                fill_opacity=0.75,
                stroke_color=WHITE,
                stroke_width=1.5,
            )
            sq.move_to(RIGHT * c * SIDE + DOWN * r * SIDE)
            lbl = Text(letter, font_size=22, color=WHITE, weight=BOLD)
            lbl.move_to(sq.get_center())
            cells.append(sq)
            cell_lbls.append(lbl)

        raw_grid = VGroup(*cells, *cell_lbls)
        raw_grid.move_to(LEFT * 4.2)

        raw_title = Text("Raw Bayer\n(centre = Blue pixel)", font_size=24,
                         color=YELLOW, line_spacing=1.2)
        raw_title.next_to(raw_grid, UP, buff=0.25)

        self.play(FadeIn(raw_grid), Write(raw_title))
        self.wait(0.3)

        centre_ring = SurroundingRectangle(cells[4], color=YELLOW,
                                           stroke_width=3, buff=0.04)
        self.play(Create(centre_ring))

        question_marks = VGroup(
            Text("R = ?", font_size=20, color=BAYER_R),
            Text("G = ?", font_size=20, color=BAYER_G),
            Text("B = known", font_size=20, color=BAYER_B),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        question_marks.next_to(raw_grid, DOWN, buff=0.3)
        self.play(FadeIn(question_marks))
        self.wait(0.4)

        # Centre arrow + label
        interp_arrow = Arrow(
            raw_grid.get_right() + RIGHT * 0.1,
            raw_grid.get_right() + RIGHT * 2.0,
            buff=0.0, color=WHITE, stroke_width=3,
            max_tip_length_to_length_ratio=0.25,
        )
        interp_lbl = Text("Bilinear\nInterpolation", font_size=26, color=YELLOW,
                          line_spacing=1.2)
        interp_lbl.next_to(interp_arrow, UP, buff=0.1)

        self.play(GrowArrow(interp_arrow), Write(interp_lbl))
        self.wait(0.3)

        # After: full-colour centre pixel
        after_sq = Square(
            side_length=SIDE * 1.6,
            stroke_color=WHITE,
            stroke_width=2.5,
            fill_color=PURPLE,
            fill_opacity=0.85,
        )
        after_sq.next_to(interp_arrow, RIGHT, buff=0.2)

        after_vals = VGroup(
            Text("R = avg(neighbours)", font_size=18, color=BAYER_R),
            Text("G = avg(neighbours)", font_size=18, color=BAYER_G),
            Text("B = measured",        font_size=18, color=BAYER_B),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        after_vals.next_to(after_sq, RIGHT, buff=0.25)

        after_title = Text("Full-colour pixel", font_size=22, color=YELLOW)
        after_title.next_to(after_sq, UP, buff=0.15)

        self.play(FadeIn(after_sq), Write(after_title), FadeIn(after_vals))
        self.wait(0.5)

        # Formula
        formula = MathTex(
            r"\hat{R}[i,j] = \frac{1}{|\mathcal{N}|}"
            r"\sum_{(di,dj)\in\mathcal{N}} R[i+di,\,j+dj]",
            font_size=30,
        )
        formula.to_edge(DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait(2)

    # ── Section 4: Colour Space Conversion ───────────────────────────────────
    def _section_colorspace(self):
        self._section_title(
            "Step 4: Colour Space Conversion",
            "RGB → CIE XYZ → L*a*b* → Oklab",
        )
        self._cs_phase_a_rgb_cube()
        self._fade_all_out()
        self._cs_phase_b_cie_cmf()
        self._fade_all_out()
        self._cs_phase_c_transform_pipeline()
        self._fade_all_out()
        self._cs_phase_d_hsv_cylinder()
        self._fade_all_out()
        self._cs_phase_e_cielab_solid()
        self._fade_all_out()
        self._cs_phase_f_oklab()
        self.wait(2)

    # ── helpers shared by colour-space phases ─────────────────────────────────
    @staticmethod
    def _cs_load_cmf():
        """Load CIE 1931 CMF data. Returns (wavelengths, x_bar, y_bar, z_bar)."""
        # __file__ = .../src/blob_detection/main.py  →  project root is 2 levels up
        _PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cmf = np.loadtxt(
            os.path.join(_PROJECT, "resource", "CIE_xyz_1931_2deg.csv"),
            delimiter=",",
        )
        return cmf[:, 0], cmf[:, 1], cmf[:, 2], cmf[:, 3]

    @staticmethod
    def _oklab_from_linear_rgb(r, g, b):
        """Convert linear sRGB → Oklab. Returns (L, a, b) as floats."""
        M1 = np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ])
        M2 = np.array([
            [0.2104542553,  0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050,  0.4505937099],
            [0.0259040371,  0.7827717662, -0.8086757660],
        ])
        lms = M1 @ np.array([r, g, b])
        lms_c = np.cbrt(np.clip(lms, 0, None))
        lab = M2 @ lms_c
        return lab[0], lab[1], lab[2]

    # ── Phase A: RGB Cube (2D isometric + 3D matplotlib) ─────────────────────
    def _cs_phase_a_rgb_cube(self):
        title = Text("RGB Colour Space — The Unit Cube", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        # ── generate matplotlib 3D view ────────────────────────────────────

        n3 = 9
        rv, gv, bv = np.meshgrid(
            np.linspace(0, 1, n3), np.linspace(0, 1, n3), np.linspace(0, 1, n3)
        )
        rv, gv, bv = rv.ravel(), gv.ravel(), bv.ravel()
        rgb_pts = np.stack([rv, gv, bv], axis=1)

        fig = plt.figure(figsize=(4.5, 4.5), facecolor="black")
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.set_facecolor("black")
        ax3.scatter(rv, gv, bv, c=rgb_pts, s=6, alpha=0.85, depthshade=True)
        for spine in ax3.spines.values():
            spine.set_color("white")
        ax3.set_xlabel("R", color="red",   fontsize=10, labelpad=4)
        ax3.set_ylabel("G", color="green", fontsize=10, labelpad=4)
        ax3.set_zlabel("B", color="#4488ff", fontsize=10, labelpad=4)
        ax3.tick_params(colors="white", labelsize=7)
        ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor("white"); ax3.yaxis.pane.set_edgecolor("white")
        ax3.zaxis.pane.set_edgecolor("white")
        ax3.view_init(elev=22, azim=40)
        ax3.set_title("3D Perspective", color="white", fontsize=11, pad=6)
        fig.tight_layout()
        _tmp_cube = "/tmp/rgb_cube_3d.png"
        fig.savefig(_tmp_cube, dpi=110, bbox_inches="tight", facecolor="black")
        plt.close(fig)

        img_cube = ImageMobject(_tmp_cube).set_height(4.8).move_to(RIGHT * 3.4 + DOWN * 0.2)

        # ── 2D isometric projection (Manim Dots) ──────────────────────────
        n2 = 8
        rv2, gv2, bv2 = np.meshgrid(
            np.linspace(0, 1, n2), np.linspace(0, 1, n2), np.linspace(0, 1, n2)
        )
        rv2, gv2, bv2 = rv2.ravel(), gv2.ravel(), bv2.ravel()

        ISO_S = 1.55
        c30 = np.cos(np.radians(30))
        x2d = (rv2 - bv2) * c30 * ISO_S
        y2d = (gv2 - (rv2 + bv2) / 2.0) * ISO_S

        # Sort back-to-front (larger y2d drawn later = on top)
        order = np.argsort(y2d)
        dots_2d = [
            Dot(
                point=[x2d[i] - 3.5, y2d[i] - 0.2, 0],
                radius=0.055,
                color=ManimColor([float(rv2[i]), float(gv2[i]), float(bv2[i]), 1.0]),
            )
            for i in order
        ]

        # Axis arrows from origin of the cube (black corner at (-3.5, -0.2))
        orig = np.array([-3.5, -0.2, 0.0])
        def _axis_arr(dx, dy, col, lbl_str, offset):
            end = orig + np.array([dx * ISO_S * c30, dy * ISO_S, 0])
            arr = Arrow(orig, end, buff=0, stroke_width=2.5, color=col,
                        max_tip_length_to_length_ratio=0.3)
            lbl = Text(lbl_str, font_size=18, color=col).move_to(end + offset)
            return arr, lbl

        # R-axis: (r=1, g=0, b=0) direction → right-down
        arr_r, lbl_r = _axis_arr(1, -0.5, RED,   "R", RIGHT * 0.28 + DOWN * 0.1)
        # G-axis: (r=0, g=1, b=0) direction → up
        arr_g, lbl_g = _axis_arr(0,  1.0, GREEN, "G", UP * 0.25)
        # B-axis: (r=0, g=0, b=1) direction → left-down
        arr_b, lbl_b = _axis_arr(-1, -0.5, ManimColor([0.3, 0.6, 1.0, 1.0]), "B", LEFT * 0.28 + DOWN * 0.1)

        # Corner labels
        def _corner(r, g, b, lbl, offset):
            cx = (r - b) * c30 * ISO_S - 3.5
            cy = (g - (r + b) / 2.0) * ISO_S - 0.2
            return Text(lbl, font_size=12, color=NOTE_COLOR).move_to([cx, cy, 0] + offset)

        corners = VGroup(
            _corner(0, 0, 0, "Black",   LEFT * 0.4 + DOWN * 0.15),
            _corner(1, 1, 1, "White",   RIGHT * 0.4 + UP * 0.15),
            _corner(1, 0, 0, "Red",     RIGHT * 0.35),
            _corner(0, 1, 0, "Green",   UP * 0.2),
            _corner(0, 0, 1, "Blue",    LEFT * 0.35),
            _corner(1, 1, 0, "Yellow",  RIGHT * 0.35 + UP * 0.15),
            _corner(0, 1, 1, "Cyan",    LEFT * 0.35 + UP * 0.15),
            _corner(1, 0, 1, "Magenta", RIGHT * 0.3 + DOWN * 0.15),
        )

        iso_lbl = Text("2D Isometric Projection", font_size=16, color=NOTE_COLOR)
        iso_lbl.move_to([-3.5, -2.85, 0])
        persp_lbl = Text("3D Perspective View", font_size=16, color=NOTE_COLOR)
        persp_lbl.next_to(img_cube, DOWN, buff=0.15)

        dots_grp = VGroup(*dots_2d)
        axes_grp = VGroup(arr_r, lbl_r, arr_g, lbl_g, arr_b, lbl_b)

        self.play(LaggedStart(*[FadeIn(d) for d in dots_2d], lag_ratio=0.003, run_time=2.5))
        self.play(LaggedStart(*[GrowArrow(a) for a in [arr_r, arr_g, arr_b]], lag_ratio=0.2),
                  FadeIn(lbl_r, lbl_g, lbl_b, corners, iso_lbl))
        self.play(FadeIn(img_cube), Write(persp_lbl))
        self.wait(0.5)

        note = Text(
            "All device-independent colour spaces are transformations of this cube.\n"
            "Same orange: RGB(220, 80, 40) indoors ≠ RGB(185, 65, 30) outdoors.",
            font_size=20, color=ORANGE, line_spacing=1.3,
        )
        note.to_edge(DOWN, buff=0.3)
        self.play(Write(note))
        self.wait(2)

    # ── Phase B: CIE 1931 Colour Matching Functions ───────────────────────────
    def _cs_phase_b_cie_cmf(self):
        title = Text("CIE 1931 Colour Matching Functions", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        wavelengths, x_bar, y_bar, z_bar = self._cs_load_cmf()

        ax = Axes(
            x_range=[380, 730, 50],
            y_range=[0, 2.15, 0.5],
            x_length=9.5,
            y_length=4.0,
            axis_config={"stroke_color": WHITE, "include_tip": True,
                         "tip_length": 0.16, "stroke_width": 2},
        )
        ax.move_to(ORIGIN + DOWN * 0.35)
        x_lbl = Text("Wavelength (nm)", font_size=18, color=WHITE)
        x_lbl.next_to(ax.get_x_axis(), DOWN, buff=0.22)
        y_lbl = Text("Sensitivity", font_size=18, color=WHITE)
        y_lbl.next_to(ax.get_y_axis(), LEFT, buff=0.12)

        # Visible spectrum bar along x-axis
        spec_lines = VGroup()
        for nm in range(380, 730, 8):
            hue = np.interp(nm, [380, 450, 500, 560, 620, 700],
                            [0.78, 0.67, 0.50, 0.28, 0.05, 0.0])
            ln = Line(ax.c2p(nm, 0), ax.c2p(nm, 2.0),
                      stroke_color=_hsv(hue, 0.85, 0.9), stroke_width=5,
                      stroke_opacity=0.32)
            spec_lines.add(ln)

        # CMF curves
        vis = (wavelengths >= 380) & (wavelengths <= 730)
        wl_v = wavelengths[vis]

        def _curve(vals, color):
            pts = [ax.c2p(float(w), float(v)) for w, v in zip(wl_v, vals[vis])]
            vm = VMobject(stroke_color=color, stroke_width=2.8)
            vm.set_points_smoothly(pts)
            return vm

        cx = _curve(x_bar, RED)
        cy = _curve(y_bar, GREEN)
        cz = _curve(z_bar, ManimColor([0.3, 0.6, 1.0, 1.0]))

        lx = Text("x̄(λ) — 'red' response",  font_size=17, color=RED)
        ly = Text("ȳ(λ) — luminosity",        font_size=17, color=GREEN)
        lz = Text("z̄(λ) — 'blue' response",  font_size=17, color=ManimColor([0.3, 0.6, 1.0, 1.0]))
        curve_lbls = VGroup(lx, ly, lz).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        curve_lbls.to_edge(RIGHT, buff=0.25).shift(UP * 1.2)

        formula = MathTex(
            r"X = \int S(\lambda)\,\bar{x}(\lambda)\,d\lambda,\quad "
            r"Y = \int S(\lambda)\,\bar{y}(\lambda)\,d\lambda",
            font_size=26, color=NOTE_COLOR,
        )
        formula.to_edge(DOWN, buff=0.28)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(FadeIn(spec_lines))
        self.play(Create(cx), Write(lx))
        self.play(Create(cy), Write(ly))
        self.play(Create(cz), Write(lz))
        self.play(Write(formula))

        note = Text(
            "ȳ(λ) = photopic luminosity.  Every modern colour space is defined via CIE XYZ.",
            font_size=20, color=ORANGE,
        )
        note.to_edge(DOWN, buff=0.05)
        self.play(Write(note))
        self.wait(2)

    # ── Phase C: Transformation Pipeline ─────────────────────────────────────
    def _cs_phase_c_transform_pipeline(self):
        title = Text("Colour Space Transformation Pipeline", font_size=30, color=YELLOW)
        title.to_edge(UP, buff=0.28)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        # Stages: (label, color)
        stages = [
            ("sRGB\n(device)", ORANGE),
            ("Linear RGB", ManimColor([0.85, 0.4, 0.4, 1.0])),
            ("CIE XYZ", PURPLE_B),
            ("CIE L*a*b*", BLUE_B),
            ("Oklab", GREEN_B),
            ("OKLCH", TEAL_B),
        ]
        blks = [self._block(lbl, 2.0, 0.62, col, font_size=16) for lbl, col in stages]

        # Two rows of 3, S-shaped flow: row1 L→R, row2 R←L (so XYZ connects to Lab)
        row1 = VGroup(*blks[:3]).arrange(RIGHT, buff=1.0)
        row2 = VGroup(*blks[5], *blks[4], *blks[3]).arrange(RIGHT, buff=1.0)
        pipeline = VGroup(row1, row2).arrange(DOWN, buff=1.2)
        pipeline.move_to(ORIGIN + DOWN * 0.25)

        # Arrows and formula labels
        def _pipe_arrow(src, dst, label_str):
            arr = Arrow(src.get_right(), dst.get_left(), buff=0.08, stroke_width=2.0,
                        color=WHITE, max_tip_length_to_length_ratio=0.25)
            lbl = Text(label_str, font_size=13, color=NOTE_COLOR)
            lbl.next_to(arr, UP, buff=0.06)
            return arr, lbl

        arr1, l1 = _pipe_arrow(blks[0], blks[1], "γ decode")
        arr2, l2 = _pipe_arrow(blks[1], blks[2], "M·c")

        # Vertical arrow: XYZ (row1[2]) → Lab (row2[2] = blks[3])
        arr_v = Arrow(blks[2].get_bottom(), blks[3].get_top(), buff=0.08,
                      stroke_width=2.0, color=WHITE, max_tip_length_to_length_ratio=0.25)
        lv = Text("f(Y/Yₙ)", font_size=13, color=NOTE_COLOR)
        lv.next_to(arr_v, RIGHT, buff=0.08)

        arr3, l3 = _pipe_arrow(blks[3], blks[4], "M + ∛")
        arr4, l4 = _pipe_arrow(blks[4], blks[5], "polar")

        # Key formulas (shown below pipeline)
        f_gamma = MathTex(
            r"C_\text{lin} = \left(\frac{C+0.055}{1.055}\right)^{2.4}",
            font_size=22, color=NOTE_COLOR,
        )
        f_lab = MathTex(
            r"L^* = 116\,f\!\left(\tfrac{Y}{Y_n}\right)-16,\quad "
            r"a^* = 500\!\left[f\!\left(\tfrac{X}{X_n}\right)-f\!\left(\tfrac{Y}{Y_n}\right)\right]",
            font_size=20, color=NOTE_COLOR,
        )
        f_oklch = MathTex(
            r"C = \sqrt{a^2+b^2},\quad H = \operatorname{atan2}(b,a)",
            font_size=22, color=NOTE_COLOR,
        )
        formulas = VGroup(f_gamma, f_lab, f_oklch).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        formulas.to_edge(DOWN, buff=0.18)

        self.play(LaggedStart(*[FadeIn(b) for b in blks], lag_ratio=0.12, run_time=1.5))
        self.play(
            LaggedStart(
                GrowArrow(arr1), Write(l1),
                GrowArrow(arr2), Write(l2),
                GrowArrow(arr_v), Write(lv),
                GrowArrow(arr3), Write(l3),
                GrowArrow(arr4), Write(l4),
                lag_ratio=0.15,
            )
        )
        self.play(Write(formulas))
        self.wait(2)

    # ── Phase D: HSV Cylinder ─────────────────────────────────────────────────
    def _cs_phase_d_hsv_cylinder(self):
        title = Text("HSV Colour Space — The Cylinder", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        NUM_H = 36       # hue slices
        NUM_S = 6        # saturation rings
        R_MAX = 1.55     # outer radius

        radii = np.linspace(0, R_MAX, NUM_S + 1)

        # Build hue wheel with saturation rings (AnnularSector)
        wheel = VGroup()
        for si in range(NUM_S):
            s_val = (si + 0.5) / NUM_S
            r_in  = radii[si]
            r_out = radii[si + 1]
            for hi in range(NUM_H):
                wheel.add(
                    AnnularSector(
                        inner_radius=r_in,
                        outer_radius=r_out,
                        angle=TAU / NUM_H,
                        start_angle=hi * TAU / NUM_H,
                        fill_color=_hsv(hi / NUM_H, s_val, 0.95),
                        fill_opacity=0.93,
                        stroke_width=0,
                    )
                )

        wheel.move_to(LEFT * 3.2 + DOWN * 0.3)

        # Hue angle label
        h_arc_lbl = Text("H = hue\n0°→360°", font_size=16, color=YELLOW)
        h_arc_lbl.move_to(wheel.get_center() + UP * (R_MAX + 0.45))

        # Saturation arrow (from center outward)
        sat_arr = Arrow(
            wheel.get_center(),
            wheel.get_center() + RIGHT * (R_MAX + 0.1),
            buff=0, stroke_width=2.0, color=WHITE, max_tip_length_to_length_ratio=0.25,
        )
        s_lbl = Text("S = saturation →", font_size=15, color=WHITE)
        s_lbl.next_to(sat_arr, DOWN, buff=0.1)

        # Value strip (right side): V from 1 (top) to 0 (bottom)
        N_V = 20
        v_strip = VGroup()
        strip_h = 3.4
        strip_w = 0.55
        for vi in range(N_V):
            v_val = 1.0 - vi / N_V
            rect = Rectangle(
                width=strip_w, height=strip_h / N_V,
                fill_color=_hsv(0.08, 1.0, v_val),
                fill_opacity=0.95, stroke_width=0,
            )
            rect.move_to([0.3, (strip_h / 2) - (vi + 0.5) * (strip_h / N_V), 0])
            v_strip.add(rect)
        v_strip.move_to(RIGHT * 0.3 + DOWN * 0.3)

        v_top_lbl = Text("V = 1\n(bright)", font_size=15, color=WHITE)
        v_top_lbl.next_to(v_strip, UP, buff=0.12)
        v_bot_lbl = Text("V = 0\n(black)", font_size=15, color=NOTE_COLOR)
        v_bot_lbl.next_to(v_strip, DOWN, buff=0.12)

        # HSV component summary
        hsv_sum = VGroup(
            Text("H = Hue (which colour)",     font_size=18, color=YELLOW),
            Text("S = Saturation (how vivid)",  font_size=18, color=WHITE),
            Text("V = Value (how bright)",      font_size=18, color=NOTE_COLOR),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        hsv_sum.move_to(RIGHT * 3.6 + DOWN * 0.5)

        self.play(LaggedStart(*[FadeIn(s) for s in wheel], lag_ratio=0.005, run_time=2.0))
        self.play(FadeIn(h_arc_lbl), GrowArrow(sat_arr), Write(s_lbl))
        self.play(FadeIn(v_strip), Write(v_top_lbl), Write(v_bot_lbl))
        self.play(FadeIn(hsv_sum))
        self.wait(0.4)

        # Hue threshold demo
        thresh_lo, thresh_hi = 15.0, 48.0  # orange range in degrees
        lo_rad = np.radians(thresh_lo)
        hi_rad = np.radians(thresh_hi)
        thresh_arc = Arc(
            radius=R_MAX + 0.18,
            start_angle=lo_rad,
            angle=hi_rad - lo_rad,
            stroke_color=YELLOW,
            stroke_width=4.5,
        )
        thresh_arc.move_to(wheel.get_center())
        thresh_text = Text("Orange:\nH ∈ [15°, 48°]", font_size=16, color=YELLOW)
        thresh_text.next_to(wheel.get_center() + RIGHT * (R_MAX + 0.3), RIGHT, buff=0.05)
        thresh_text.shift(DOWN * 0.3)

        # Highlight orange sectors
        orange_sectors = VGroup(*[
            AnnularSector(
                inner_radius=0, outer_radius=R_MAX,
                angle=TAU / NUM_H,
                start_angle=hi * TAU / NUM_H,
                fill_color=YELLOW,
                fill_opacity=0.35,
                stroke_width=0,
            )
            for hi in range(NUM_H)
            if 15 <= (hi * 360 / NUM_H) <= 48
        ])
        orange_sectors.move_to(wheel.get_center())

        self.play(Create(thresh_arc), Write(thresh_text))
        self.play(FadeIn(orange_sectors))

        note = Text(
            "Intuitive — but HSV threshold fails under changing illumination!",
            font_size=20, color=ORANGE,
        )
        note.to_edge(DOWN, buff=0.3)
        self.play(Write(note))
        self.wait(2)

    # ── Phase E: CIELab Colour Solid ─────────────────────────────────────────
    def _cs_phase_e_cielab_solid(self):
        title = Text("CIE L*a*b* — The Optimal Colour Solid", font_size=30, color=YELLOW)
        title.to_edge(UP, buff=0.28)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        _PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _SRC  = os.path.join(_PROJECT, "src")

        img_top  = ImageMobject(os.path.join(_SRC, "optimal_colors_3d_top.png"))
        img_side = ImageMobject(os.path.join(_SRC, "optimal_colors_3d_left.png"))
        img_top.set_height(3.8).move_to(LEFT * 3.1 + DOWN * 0.35)
        img_side.set_height(3.8).move_to(RIGHT * 3.1 + DOWN * 0.35)

        top_title  = Text("Top view  (a* vs b*)", font_size=18, color=NOTE_COLOR)
        top_title.next_to(img_top, UP, buff=0.1)
        side_title = Text("Side view  (a* vs L*)", font_size=18, color=NOTE_COLOR)
        side_title.next_to(img_side, UP, buff=0.1)

        # Axis annotations on top view
        a_lbl_r = Text("a* →  red",   font_size=14, color=BAYER_R)
        a_lbl_r.next_to(img_top, RIGHT, buff=0.12).shift(DOWN * 0.4)
        a_lbl_l = Text("←  green",    font_size=14, color=BAYER_G)
        a_lbl_l.next_to(img_top, LEFT, buff=0.12).shift(DOWN * 0.4)
        b_lbl_u = Text("↑  yellow", font_size=14, color=YELLOW)
        b_lbl_u.next_to(img_top, UP, buff=0.35).shift(RIGHT * 0.5)
        b_lbl_d = Text("↓  blue",   font_size=14, color=BAYER_B)
        b_lbl_d.next_to(img_top, DOWN, buff=0.22).shift(RIGHT * 0.5)

        # Side-view annotations
        white_dot = Dot(img_side.get_top()   + DOWN * 0.3, radius=0.09, color=WHITE)
        black_dot = Dot(img_side.get_bottom() + UP   * 0.3, radius=0.09, color=GREY)
        wl = Text("L*=100 white", font_size=14, color=WHITE).next_to(white_dot, RIGHT, buff=0.1)
        bl = Text("L*=0   black", font_size=14, color=NOTE_COLOR).next_to(black_dot, RIGHT, buff=0.1)
        mid_brace = Brace(
            Line(img_side.get_center() + UP * 0.3, img_side.get_center() + DOWN * 0.3),
            direction=RIGHT, color=TEAL_B,
        )
        mid_lbl = Text("Max\nchroma", font_size=13, color=TEAL_B, line_spacing=1.1)
        mid_lbl.next_to(mid_brace, RIGHT, buff=0.08)

        shape_note = Text(
            "Shape: pointed at black & white, bulging at mid-lightness\n"
            "— like Munsell's COLOR TREE (1905). Boundary = 'optimal colors'.",
            font_size=18, color=NOTE_COLOR, line_spacing=1.2,
        )
        shape_note.to_edge(DOWN, buff=0.38)

        de_formula = MathTex(
            r"\Delta E_{ab}^* = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}",
            font_size=26, color=WHITE,
        )
        de_formula.to_edge(DOWN, buff=0.12)

        self.play(FadeIn(img_top), Write(top_title))
        self.play(FadeIn(a_lbl_r, a_lbl_l, b_lbl_u, b_lbl_d))
        self.play(FadeIn(img_side), Write(side_title))
        self.play(FadeIn(white_dot), Write(wl), FadeIn(black_dot), Write(bl))
        self.play(FadeIn(mid_brace), Write(mid_lbl))
        self.play(Write(shape_note))
        self.wait(0.5)
        self.play(FadeOut(shape_note))
        self.play(Write(de_formula))
        note = Text("Equal Euclidean distance ≈ equal perceived colour difference (perceptual uniformity)",
                    font_size=18, color=ORANGE)
        note.to_edge(DOWN, buff=0.38)
        self.play(Write(note))
        self.wait(2)

    # ── Phase F: Oklab & OKLCH ────────────────────────────────────────────────
    def _cs_phase_f_oklab(self):
        title = Text("Oklab & OKLCH — Modern Perceptual Colour Spaces", font_size=28, color=YELLOW)
        title.to_edge(UP, buff=0.28)
        self.play(FadeIn(title, shift=DOWN * 0.1))

        # ── Oklab gamut scatter (sRGB cube → Oklab a*/b*) ─────────────────
        np.random.seed(7)
        n_samp = 600
        rs = np.random.rand(n_samp)
        gs = np.random.rand(n_samp)
        bs = np.random.rand(n_samp)
        # gamma decode
        def _gamma_dec(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        rl, gl, bl = _gamma_dec(rs), _gamma_dec(gs), _gamma_dec(bs)

        ok_L_arr, ok_a_arr, ok_b_arr = [], [], []
        for i in range(n_samp):
            L, a, b = self._oklab_from_linear_rgb(float(rl[i]), float(gl[i]), float(bl[i]))
            ok_L_arr.append(L); ok_a_arr.append(a); ok_b_arr.append(b)
        ok_L_arr = np.array(ok_L_arr)
        ok_a_arr = np.array(ok_a_arr)
        ok_b_arr = np.array(ok_b_arr)

        # Left: Oklab a*/b* scatter
        ax_ok = Axes(
            x_range=[-0.40, 0.40, 0.2],
            y_range=[-0.40, 0.40, 0.2],
            x_length=3.6,
            y_length=3.6,
            axis_config={"stroke_color": WHITE, "include_tip": True,
                         "tip_length": 0.14, "stroke_width": 1.8},
        )
        ax_ok.move_to(LEFT * 3.4 + DOWN * 0.35)
        ok_lbl_x = Text("a*", font_size=16, color=BAYER_R)
        ok_lbl_x.next_to(ax_ok.get_x_axis().get_right(), RIGHT, buff=0.06)
        ok_lbl_y = Text("b*", font_size=16, color=YELLOW)
        ok_lbl_y.next_to(ax_ok.get_y_axis().get_top(), UP, buff=0.06)
        ok_title = Text("Oklab  (a* vs b*)", font_size=17, color=NOTE_COLOR)
        ok_title.next_to(ax_ok, UP, buff=0.12)

        ok_dots = VGroup(*[
            Dot(
                ax_ok.c2p(float(ok_a_arr[i]), float(ok_b_arr[i])),
                radius=0.035,
                color=ManimColor([float(rs[i]), float(gs[i]), float(bs[i]), 1.0]),
            )
            for i in range(n_samp)
        ])

        shape_lbl = Text(
            '"Coffee cup" shape:\nflat top/bottom,\nconvex right,\nconcave left',
            font_size=14, color=NOTE_COLOR, line_spacing=1.15,
        )
        shape_lbl.next_to(ax_ok, RIGHT, buff=0.3).shift(UP * 0.4)

        # Right: OKLCH polar view
        ax_pol_title = Text("OKLCH  (polar: C vs H)", font_size=17, color=NOTE_COLOR)
        ax_pol_center = RIGHT * 3.4 + DOWN * 0.35

        polar_circles = VGroup(*[
            Circle(radius=r, stroke_color=GREY, stroke_width=0.8, stroke_opacity=0.5)
            .move_to(ax_pol_center)
            for r in [0.45, 0.90, 1.35, 1.80]
        ])
        polar_spokes = VGroup(*[
            Line(
                ax_pol_center,
                ax_pol_center + np.array([np.cos(a) * 1.85, np.sin(a) * 1.85, 0]),
                stroke_color=GREY, stroke_width=0.8, stroke_opacity=0.5,
            )
            for a in np.linspace(0, TAU, 12, endpoint=False)
        ])
        pol_c_lbl = Text("C →", font_size=14, color=NOTE_COLOR)
        pol_c_lbl.move_to(ax_pol_center + RIGHT * 2.1)
        pol_h_lbl = Text("H ↺", font_size=14, color=NOTE_COLOR)
        pol_h_lbl.move_to(ax_pol_center + UP * 2.05)

        # Same Oklab data plotted in polar
        C_arr = np.sqrt(ok_a_arr ** 2 + ok_b_arr ** 2)
        H_arr = np.arctan2(ok_b_arr, ok_a_arr)
        pol_scale = 1.80 / C_arr.max()
        polar_dots = VGroup(*[
            Dot(
                ax_pol_center + np.array([C_arr[i] * pol_scale * np.cos(H_arr[i]),
                                          C_arr[i] * pol_scale * np.sin(H_arr[i]), 0]),
                radius=0.035,
                color=ManimColor([float(rs[i]), float(gs[i]), float(bs[i]), 1.0]),
            )
            for i in range(n_samp)
        ])
        ax_pol_title.move_to(ax_pol_center + UP * 2.3)

        self.play(Create(ax_ok), Write(ok_lbl_x), Write(ok_lbl_y), Write(ok_title))
        self.play(LaggedStart(*[FadeIn(d) for d in ok_dots], lag_ratio=0.003, run_time=1.5))
        self.play(Write(shape_lbl))
        self.play(FadeIn(polar_circles, polar_spokes), Write(pol_c_lbl), Write(pol_h_lbl),
                  Write(ax_pol_title))
        self.play(LaggedStart(*[FadeIn(d) for d in polar_dots], lag_ratio=0.003, run_time=1.5))
        self.wait(0.4)

        # Interpolation comparison
        self.play(FadeOut(VGroup(ax_ok, ok_lbl_x, ok_lbl_y, ok_title, ok_dots, shape_lbl,
                                  polar_circles, polar_spokes, pol_c_lbl, pol_h_lbl,
                                  ax_pol_title, polar_dots)))

        interp_title = Text("Interpolation: Rectangular (Oklab) vs Polar (OKLCH)", font_size=24, color=WHITE)
        interp_title.move_to(UP * 2.8)
        self.play(Write(interp_title))

        # Blue (0,0,1) → Orange (1,0.5,0) via Oklab and OKLCH
        blue_lin  = _gamma_dec(np.array([0.0, 0.0, 1.0]))
        orange_lin = _gamma_dec(np.array([1.0, 0.5, 0.0]))
        bL, ba, bb = self._oklab_from_linear_rgb(*blue_lin)
        oL, oa, ob = self._oklab_from_linear_rgb(*orange_lin)

        N_STEPS = 14
        strip_w, strip_h = 0.72, 0.62

        # Oklab strip (rectangular lerp)
        ok_strip = VGroup()
        for k in range(N_STEPS):
            t = k / (N_STEPS - 1)
            iL = bL + t * (oL - bL)
            ia = ba + t * (oa - ba)
            ib = bb + t * (ob - bb)
            # Oklab → LMS
            M2_inv = np.linalg.inv(np.array([
                [0.2104542553,  0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050,  0.4505937099],
                [0.0259040371,  0.7827717662, -0.8086757660],
            ]))
            M1_inv = np.linalg.inv(np.array([
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]))
            lms_c = M2_inv @ np.array([iL, ia, ib])
            lms = lms_c ** 3
            lin_rgb = M1_inv @ lms
            lin_rgb = np.clip(lin_rgb, 0, 1)
            # gamma encode
            srgb = np.where(lin_rgb <= 0.0031308,
                            12.92 * lin_rgb,
                            1.055 * lin_rgb ** (1 / 2.4) - 0.055)
            srgb = np.clip(srgb, 0, 1)
            r_sq = Rectangle(width=strip_w, height=strip_h,
                              fill_color=ManimColor([float(srgb[0]), float(srgb[1]), float(srgb[2]), 1.0]),
                              fill_opacity=1.0, stroke_width=0)
            r_sq.move_to([(k - N_STEPS / 2 + 0.5) * strip_w, 0.5, 0])
            ok_strip.add(r_sq)

        # OKLCH strip (polar lerp — constant C, lerp H)
        bC = np.sqrt(ba ** 2 + bb ** 2); bH = np.arctan2(bb, ba)
        oC = np.sqrt(oa ** 2 + ob ** 2); oH = np.arctan2(ob, oa)
        if oH - bH > np.pi:  oH -= TAU
        if bH - oH > np.pi:  bH -= TAU

        lch_strip = VGroup()
        M2_inv = np.linalg.inv(np.array([
            [0.2104542553,  0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050,  0.4505937099],
            [0.0259040371,  0.7827717662, -0.8086757660],
        ]))
        M1_inv = np.linalg.inv(np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]))
        for k in range(N_STEPS):
            t = k / (N_STEPS - 1)
            iL = bL + t * (oL - bL)
            iC = bC + t * (oC - bC)
            iH = bH + t * (oH - bH)
            ia = iC * np.cos(iH); ib = iC * np.sin(iH)
            lms_c = M2_inv @ np.array([iL, ia, ib])
            lms = lms_c ** 3
            lin_rgb = np.clip(M1_inv @ lms, 0, 1)
            srgb = np.clip(np.where(lin_rgb <= 0.0031308,
                                    12.92 * lin_rgb,
                                    1.055 * lin_rgb ** (1 / 2.4) - 0.055), 0, 1)
            r_sq = Rectangle(width=strip_w, height=strip_h,
                              fill_color=ManimColor([float(srgb[0]), float(srgb[1]), float(srgb[2]), 1.0]),
                              fill_opacity=1.0, stroke_width=0)
            r_sq.move_to([(k - N_STEPS / 2 + 0.5) * strip_w, -0.7, 0])
            lch_strip.add(r_sq)

        ok_lbl  = Text("Oklab  (rectangular — passes through gray)", font_size=17, color=GREEN_B)
        ok_lbl.next_to(ok_strip,  LEFT, buff=0.2)
        lch_lbl = Text("OKLCH  (polar — keeps saturation)", font_size=17, color=TEAL_B)
        lch_lbl.next_to(lch_strip, LEFT, buff=0.2)

        self.play(FadeIn(ok_strip), Write(ok_lbl))
        self.play(FadeIn(lch_strip), Write(lch_lbl))

        closing = Text(
            "Oklab (2020): simple math + perceptually superior to L*a*b*.\n"
            "OKLCH = same data in polar coords → intuitive H/C/L editing.\n"
            "For blob detection: CIELab is standard; Oklab is the modern choice.",
            font_size=18, color=ORANGE, line_spacing=1.3,
        )
        closing.to_edge(DOWN, buff=0.22)
        self.play(Write(closing))
        self.wait(2)

    # ── Section 5: Thresholding → Binary Image ────────────────────────────────
    def _section_threshold(self):
        self._section_title(
            "Step 5: Colour Thresholding",
            "Segment objects by a range in colour space",
        )

        H, W = self.BINARY.shape
        SIDE = min(
            config.frame_width * 0.28 / W,
            config.frame_height * 0.60 / H,
        )
        value_arr = self.BINARY.copy().astype(float)
        mask_bg = self.BINARY == 0
        np.random.seed(1)
        value_arr[mask_bg]  = np.random.randint(0, 50, mask_bg.sum())
        value_arr[~mask_bg] = np.random.randint(120, 255, (~mask_bg).sum())

        def make_grid(filler, x_center):
            """Build a VGroup of coloured squares."""
            sqs = []
            for r in range(H):
                for c in range(W):
                    v = filler[r, c]
                    sq = Square(
                        side_length=SIDE,
                        fill_color=ManimColor([v / 255.0, v / 255.0, v / 255.0, 1.0]),
                        fill_opacity=0.90,
                        stroke_color=DARK_GRAY,
                        stroke_width=0.8,
                    )
                    sq.move_to(
                        np.array([x_center + c * SIDE - (W - 1) * SIDE / 2,
                                  -(r * SIDE - (H - 1) * SIDE / 2),
                                  0.0])
                    )
                    sqs.append(sq)
            return VGroup(*sqs)

        gray_grid = make_grid(value_arr, -4.8)
        gray_lbl  = Text("Grayscale Image", font_size=24, color=YELLOW)
        gray_lbl.next_to(gray_grid, UP, buff=0.2)

        self.play(FadeIn(gray_grid), Write(gray_lbl))
        self.wait(0.3)

        # Threshold function plot (centre)
        axes = Axes(
            x_range=[0, 255, 50],
            y_range=[-0.1, 1.2, 0.5],
            x_length=3.5,
            y_length=2.6,
            axis_config={"stroke_color": WHITE, "include_tip": True,
                         "tip_length": 0.18, "stroke_width": 2},
        )
        axes.move_to(ORIGIN + DOWN * 0.2)
        x_ax_lbl = Text("Pixel value", font_size=18, color=WHITE)
        x_ax_lbl.next_to(axes.get_x_axis(), DOWN, buff=0.12)
        y_ax_lbl = Text("Mask", font_size=18, color=WHITE)
        y_ax_lbl.next_to(axes.get_y_axis(), UP, buff=0.08)

        T = 75
        step = VMobject(stroke_color=GREEN, stroke_width=3)
        step.set_points_as_corners([
            axes.c2p(0, 0),
            axes.c2p(T, 0),
            axes.c2p(T, 1),
            axes.c2p(255, 1),
        ])
        thresh_line = DashedLine(
            axes.c2p(T, -0.1), axes.c2p(T, 1.2),
            stroke_color=YELLOW, stroke_width=2.5, dash_length=0.15,
        )
        thresh_lbl = MathTex(r"T=75", font_size=28, color=YELLOW)
        thresh_lbl.next_to(thresh_line, UP, buff=0.1)

        plot_title = Text("Threshold Function", font_size=24, color=YELLOW)
        plot_title.next_to(axes, UP, buff=0.12)

        self.play(Create(axes), Write(x_ax_lbl), Write(y_ax_lbl), Write(plot_title))
        self.play(Create(step), Create(thresh_line), Write(thresh_lbl))
        self.wait(0.4)

        # Binary grid (right)
        bin_filler = self.BINARY.copy().astype(float) * 255.0
        bin_grid = make_grid(bin_filler, 4.8)
        bin_lbl = Text("Binary Mask", font_size=24, color=YELLOW)
        bin_lbl.next_to(bin_grid, UP, buff=0.2)

        t_arrow = Arrow(
            axes.get_right() + RIGHT * 0.05,
            bin_grid.get_left() + LEFT * 0.05,
            buff=0.1, stroke_width=2.5, color=WHITE,
            max_tip_length_to_length_ratio=0.2,
        )
        self.play(GrowArrow(t_arrow))
        self.play(FadeIn(bin_grid), Write(bin_lbl))
        self.wait(2)

    # ── Section 6: Connected-Component Labeling (BFS) ────────────────────────
    def _section_bfs(self):
        self._section_title(
            "Step 6: Connected-Component Labeling",
            "BFS identifies every distinct blob (4-connectivity)",
        )

        binary_array = self.BINARY
        H, W = binary_array.shape
        np.random.seed(0)

        pixels, pixel_grid = self._make_pixel_grid(binary_array, scale_frac=0.52)
        pixel_grid.move_to(ORIGIN)

        idx    = np.arange(binary_array.size)
        xI, yI = idx // W, idx % W
        pose_to_index = idx.reshape(H, W)

        self.play(Write(pixel_grid))
        self.wait(0.3)

        # Show grayscale intensity
        self.play(LaggedStart(*[p.color_pixel() for p in pixels], lag_ratio=0.02))
        self.wait(0.3)

        thresh_txt = Text("Threshold → binary mask", font_size=26, color=YELLOW)
        thresh_txt.to_edge(UP, buff=0.35)
        self.play(Write(thresh_txt))

        for i, x, y in zip(idx, xI, yI):
            pixels[i].tracker.set_value(float(binary_array[x, y]))
        self.play(LaggedStart(*[p.mask_pixel() for p in pixels], lag_ratio=0.01))
        self.wait(0.3)

        bfs_txt = Text("BFS — 4-connectivity", font_size=26, color=YELLOW)
        bfs_txt.to_edge(UP, buff=0.35)
        self.play(ReplacementTransform(thresh_txt, bfs_txt))

        # Group counter panel (right edge)
        counter_title = Text("Blobs found:", font_size=24, color=ORANGE)
        counter_val   = Integer(0, font_size=36, color=WHITE)
        counter_grp   = VGroup(counter_title, counter_val).arrange(RIGHT, buff=0.2)
        counter_grp.to_edge(RIGHT, buff=0.4).to_edge(UP, buff=0.5)
        self.play(FadeIn(counter_grp))

        # Algorithm
        val       = set()
        group_count = 0
        diffX     = [-1, 0, 1, 0]
        diffY     = [0, -1, 0, 1]

        blob_labels = VGroup()

        for i, x, y in zip(idx, xI, yI):
            current_pose = (x, y)
            if current_pose in val or binary_array[x, y] != 1:
                val.add(current_pose)
                continue

            group_count += 1
            queue       = deque([current_pose])
            val.add(current_pose)
            g_color     = BLOB_PALETTE[group_count % len(BLOB_PALETTE)]

            # Update counter
            self.play(counter_val.animate.set_value(group_count), run_time=0.3)

            # Blob label on grid
            first_px = pixels[int(pose_to_index[x, y])]
            blob_lbl = Text(str(group_count), font_size=18, color=g_color, weight=BOLD)
            blob_lbl.set_z_index(5)

            while queue:
                nx, ny = queue.popleft()
                pi = int(pose_to_index[nx, ny])
                self.play(pixels[pi].set_square_color(g_color, run_time=0.12),
                          run_time=0.12)

                for dx, dy in zip(diffX, diffY):
                    nnx, nny = nx + dx, ny + dy
                    pose = (nnx, nny)
                    if (0 <= nnx < H and 0 <= nny < W
                            and binary_array[nnx, nny] == 1
                            and pose not in val):
                        queue.append(pose)
                        val.add(pose)

        self.wait(1.5)

        # Annotate: each colour = one blob
        annot = Text(
            f"{group_count} distinct blobs detected",
            font_size=28, color=WHITE,
        )
        annot.to_edge(DOWN, buff=0.4)
        self.play(Write(annot))
        self.wait(2)

    # ── Section 7: Image Moments ──────────────────────────────────────────────
    def _section_moments(self):
        self._section_title(
            "Step 7: Image Moments",
            "Describe blob geometry mathematically",
        )

        # ── Formulas (left panel) ─────────────────────────────────────────────
        f_general = MathTex(
            r"M_{pq} = \sum_{x}\sum_{y} x^p\, y^q\, I(x,y)",
            font_size=36,
        )
        f_area = MathTex(r"M_{00} = \text{Area (pixel count)}", font_size=32)
        f_m10  = MathTex(r"M_{10} = \sum x \cdot I(x,y)",       font_size=32)
        f_m01  = MathTex(r"M_{01} = \sum y \cdot I(x,y)",       font_size=32)
        f_cx   = MathTex(r"\bar{x} = \frac{M_{10}}{M_{00}}",    font_size=34)
        f_cy   = MathTex(r"\bar{y} = \frac{M_{01}}{M_{00}}",    font_size=34)
        f_mu   = MathTex(
            r"\mu_{pq} = \sum_x\sum_y (x-\bar{x})^p(y-\bar{y})^q I(x,y)",
            font_size=28,
        )

        formulas = VGroup(f_general, f_area, f_m10, f_m01).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT
        )
        centroid_row = VGroup(f_cx, f_cy).arrange(RIGHT, buff=0.9)
        left_col = VGroup(formulas, centroid_row, f_mu).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT
        )
        left_col.to_edge(LEFT, buff=0.55).shift(UP * 0.2)

        self.play(Write(f_general))
        self.wait(0.3)
        self.play(LaggedStart(Write(f_area), Write(f_m10), Write(f_m01), lag_ratio=0.3))
        self.wait(0.2)
        self.play(Write(centroid_row))
        self.wait(0.2)
        self.play(Write(f_mu))
        self.wait(0.4)

        # ── Live demo on small blob (right panel) ─────────────────────────────
        demo_blob = np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            dtype=int,
        )
        BH, BW = demo_blob.shape
        SIDE = 0.85

        demo_cells = []
        for r in range(BH):
            for c in range(BW):
                sq = Square(
                    side_length=SIDE,
                    fill_color=WHITE if demo_blob[r, c] else BLACK,
                    fill_opacity=0.85 if demo_blob[r, c] else 0.4,
                    stroke_color=GRAY,
                    stroke_width=1.5,
                )
                sq.move_to(
                    np.array([c * SIDE - (BW - 1) * SIDE / 2,
                               -(r * SIDE - (BH - 1) * SIDE / 2),
                               0.0])
                )
                demo_cells.append(sq)

        demo_grid = VGroup(*demo_cells)
        demo_grid.to_edge(RIGHT, buff=0.6).shift(UP * 1.2)

        demo_lbl = Text("Live Demo Blob", font_size=22, color=YELLOW)
        demo_lbl.next_to(demo_grid, UP, buff=0.2)

        self.play(FadeIn(demo_grid), Write(demo_lbl))

        # Live value trackers
        m00_t = ValueTracker(0.0)
        m10_t = ValueTracker(0.0)
        m01_t = ValueTracker(0.0)

        def dyn_row(tex_str, tracker, is_int=True):
            label = MathTex(tex_str, font_size=26)
            val   = Integer(0, font_size=26) if is_int else DecimalNumber(
                0, num_decimal_places=2, font_size=26
            )
            row   = VGroup(label, val).arrange(RIGHT, buff=0.15)
            if is_int:
                val.add_updater(
                    lambda d, t=tracker: d.set_value(max(0, int(t.get_value())))
                )
            else:
                val.add_updater(lambda d, t=tracker: d.set_value(t.get_value()))
            return row

        m00_row = dyn_row(r"M_{00} =", m00_t)
        m10_row = dyn_row(r"M_{10} =", m10_t)
        m01_row = dyn_row(r"M_{01} =", m01_t)

        cx_val = DecimalNumber(0, num_decimal_places=2, font_size=26)
        cy_val = DecimalNumber(0, num_decimal_places=2, font_size=26)
        cx_val.add_updater(lambda d: d.set_value(
            m10_t.get_value() / m00_t.get_value() if m00_t.get_value() >= 1 else 0.0
        ))
        cy_val.add_updater(lambda d: d.set_value(
            m01_t.get_value() / m00_t.get_value() if m00_t.get_value() >= 1 else 0.0
        ))
        cx_row = VGroup(MathTex(r"\bar{x} =", font_size=26), cx_val).arrange(RIGHT, buff=0.15)
        cy_row = VGroup(MathTex(r"\bar{y} =", font_size=26), cy_val).arrange(RIGHT, buff=0.15)

        live_panel = VGroup(m00_row, m10_row, m01_row, cx_row, cy_row).arrange(
            DOWN, buff=0.2, aligned_edge=LEFT
        )
        live_panel.next_to(demo_grid, DOWN, buff=0.4)

        self.play(FadeIn(live_panel))

        # Centroid marker
        p00     = demo_cells[0].get_center()
        vec_col = demo_cells[1].get_center() - p00
        vec_row = demo_cells[BW].get_center() - p00

        c_dot  = Dot(color=RED, radius=0.10)
        c_ring = Circle(radius=0.30, color=RED, stroke_width=3)
        c_grp  = VGroup(c_dot, c_ring)
        c_grp.set_z_index(10)
        c_grp.move_to(p00)

        def upd_centroid(mob):
            m00 = m00_t.get_value()
            if m00 >= 1.0:
                mob.move_to(
                    p00
                    + (m10_t.get_value() / m00) * vec_row
                    + (m01_t.get_value() / m00) * vec_col
                )

        c_grp.add_updater(upd_centroid)
        self.play(FadeIn(c_grp))

        # Traverse pixels
        for r in range(BH):
            for c in range(BW):
                if demo_blob[r, c] == 1:
                    ci = r * BW + c
                    self.play(
                        AnimationGroup(
                            Indicate(demo_cells[ci], color=YELLOW,
                                     scale_factor=1.15, run_time=0.22),
                            m00_t.animate(run_time=0.22).set_value(
                                m00_t.get_value() + 1),
                            m10_t.animate(run_time=0.22).set_value(
                                m10_t.get_value() + r),
                            m01_t.animate(run_time=0.22).set_value(
                                m01_t.get_value() + c),
                        )
                    )

        c_grp.remove_updater(upd_centroid)
        self.wait(1)

        mu_note = Text(
            "Central moments μ_pq describe shape independently of blob position",
            font_size=22, color=NOTE_COLOR,
        )
        mu_note.to_edge(DOWN, buff=0.35)
        self.play(Write(mu_note))
        self.wait(2)

    # ── Section 8: Shape & Area Filtering ────────────────────────────────────
    def _section_shape_filter(self):
        self._section_title(
            "Step 8: Shape & Area Filtering",
            "Keep only blobs matching expected geometry",
        )

        # ── Three example blobs ───────────────────────────────────────────────
        def make_blob(verts, color, label_str):
            poly = Polygon(
                *verts,
                fill_color=color,
                fill_opacity=0.72,
                stroke_color=WHITE,
                stroke_width=2,
            )
            lbl = Text(label_str, font_size=19, color=WHITE, line_spacing=1.1)
            lbl.move_to(poly.get_center())
            return VGroup(poly, lbl)

        # Blob A: tiny (noise)
        small_verts = [
            np.array([np.cos(a) * 0.42, np.sin(a) * 0.42, 0.0])
            for a in np.linspace(0, TAU, 10, endpoint=False)
        ]
        blobA = make_blob(small_verts, RED_D, "Noise\nBlob")
        areaA = 0.55
        circA = 0.88

        # Blob B: large but irregular
        irr_verts = [
            np.array([ 1.20,  0.10, 0.0]),
            np.array([ 0.75,  0.85, 0.0]),
            np.array([ 0.00,  1.05, 0.0]),
            np.array([-0.55,  0.40, 0.0]),
            np.array([-0.95, -0.35, 0.0]),
            np.array([-0.20, -0.90, 0.0]),
            np.array([ 0.70, -0.70, 0.0]),
        ]
        blobB = make_blob(irr_verts, ManimColor([0.85, 0.40, 0.05, 1.0]), "Irregular\nBlob")
        areaB = 4.20
        circB = 0.32

        # Blob C: large circle (target)
        circ_verts = [
            np.array([np.cos(a) * 0.95, np.sin(a) * 0.95, 0.0])
            for a in np.linspace(0, TAU, 32, endpoint=False)
        ]
        blobC = make_blob(circ_verts, GREEN_D, "Target\nBlob")
        areaC = 2.84
        circC = 0.97

        blobA.move_to(LEFT * 4.5)
        blobB.move_to(ORIGIN)
        blobC.move_to(RIGHT * 4.5)

        stats = [
            (blobA, f"Area ≈ {areaA:.1f} u²\nCirc ≈ {circA:.2f}", LEFT * 4.5),
            (blobB, f"Area ≈ {areaB:.1f} u²\nCirc ≈ {circB:.2f}", ORIGIN),
            (blobC, f"Area ≈ {areaC:.1f} u²\nCirc ≈ {circC:.2f}", RIGHT * 4.5),
        ]

        stat_labels = []
        for blob, stat_txt, pos in stats:
            lbl = Text(stat_txt, font_size=19, color=NOTE_COLOR, line_spacing=1.2)
            lbl.next_to(blob, DOWN, buff=0.3)
            stat_labels.append(lbl)

        self.play(FadeIn(blobA, blobB, blobC))
        self.play(LaggedStart(*[FadeIn(l) for l in stat_labels], lag_ratio=0.2))
        self.wait(0.5)

        # ── Circularity formula ───────────────────────────────────────────────
        circ_formula = MathTex(
            r"C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}"
            r"\quad(C \to 1 \text{ for a perfect circle})",
            font_size=30,
        )
        circ_formula.to_edge(UP, buff=0.4)
        self.play(Write(circ_formula))
        self.wait(0.5)

        # ── Step 1: Area filter ───────────────────────────────────────────────
        area_rule = Text(
            "Area Filter:  A_min < Area < A_max",
            font_size=26, color=YELLOW,
        )
        area_rule.next_to(circ_formula, DOWN, buff=0.3)
        self.play(Write(area_rule))
        self.wait(0.3)

        cross_a = Cross(blobA, color=RED, stroke_width=4)
        fail_a  = Text("FAIL\n(too small)", font_size=20, color=RED)
        fail_a.next_to(blobA, UP, buff=0.15)
        self.play(Create(cross_a), Write(fail_a))
        self.wait(0.3)
        self.play(FadeOut(blobA, stat_labels[0], cross_a, fail_a))

        pass_b = Text("PASS", font_size=22, color=GREEN)
        pass_b.next_to(blobB, UP, buff=0.15)
        pass_c = Text("PASS", font_size=22, color=GREEN)
        pass_c.next_to(blobC, UP, buff=0.15)
        self.play(Write(pass_b), Write(pass_c))
        self.wait(0.5)
        self.play(FadeOut(pass_b, pass_c))

        # ── Step 2: Circularity filter ────────────────────────────────────────
        circ_rule = Text(
            "Circularity Filter:  C > 0.7",
            font_size=26, color=YELLOW,
        )
        circ_rule.next_to(circ_formula, DOWN, buff=0.3)
        self.play(ReplacementTransform(area_rule, circ_rule))
        self.wait(0.3)

        cross_b = Cross(blobB, color=RED, stroke_width=4)
        fail_b  = Text("FAIL\n(irregular)", font_size=20, color=RED)
        fail_b.next_to(blobB, UP, buff=0.15)
        self.play(Create(cross_b), Write(fail_b))
        self.wait(0.3)
        self.play(FadeOut(blobB, stat_labels[1], cross_b, fail_b))

        # ── Surviving blob ────────────────────────────────────────────────────
        ring = Circle(
            radius=blobC[0].width / 2 + 0.22,
            color=YELLOW,
            stroke_width=3.5,
        )
        ring.move_to(blobC.get_center())
        keep_lbl = Text("✓  Detected Blob", font_size=28, color=GREEN)
        keep_lbl.next_to(blobC, UP, buff=0.25)

        self.play(Create(ring), Write(keep_lbl))
        self.wait(0.8)

        # ── Full pipeline summary ──────────────────────────────────────────────
        summary = Text(
            "Full Pipeline:\n"
            "CMOS Sensor  →  Bayer CFA  →  Demosaic  →  CIE Lab\n"
            "→  Threshold  →  CCL (BFS)  →  Moments  →  Shape Filter  →  Result",
            font_size=22,
            color=WHITE,
            line_spacing=1.35,
        )
        summary.to_edge(DOWN, buff=0.35)
        self.play(Write(summary))
        self.wait(3)


if __name__ == "__main__":
    from manim import config

    config.pixel_height = 1920
    config.pixel_width  = 1920
    config.frame_rate   = 30

    scene = BlobDetection()
    scene.render()
