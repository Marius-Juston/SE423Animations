from functools import partial
from typing import Sequence, Self, Callable

from manim import *
from manim.typing import Vector3DLike, Point3D

ANIMATE = True


class CircuitShape(VGroup):
    def set_active(self, is_active):
        target_color = YELLOW if is_active else WHITE

        stroke = self.animate.set_stroke(color=target_color) if ANIMATE else self.set_stroke(color=target_color)

        if hasattr(self, "fill_shape"):
            self.fill_shape: VGroup

            shape = self.fill_shape.animate if ANIMATE else self.fill_shape

            return stroke, shape.set_fill(color=target_color, opacity=0.8 if is_active else 0.2), shape.set_stroke(
                color=target_color)


class VisualGroup(CircuitShape):
    def __init__(self, *components):
        super().__init__(*components)
        self.components = components

    def set_active(self, is_active):
        animations = []

        for c in self.components:
            out = c.set_active(is_active)

            if isinstance(out, Sequence):
                animations.extend(out)
            else:
                animations.append(out)

        return animations


def center_return(com: VectorizedPoint):
    return com.get_location()


class LocalCoordinate(VGroup):
    get_top_left: Callable[[], np.ndarray]
    get_top_right: Callable[[], np.ndarray]
    get_bottom_left: Callable[[], np.ndarray]
    get_bottom_right: Callable[[], np.ndarray]

    def __init__(self, component, debug=False, **kwargs):
        super().__init__(**kwargs)

        for fun in (component.get_right,
                    component.get_left,
                    component.get_top,
                    component.get_bottom,
                    component.get_center):
            variable_name = f"p_{fun.__name__}"

            comp = VectorizedPoint(location=fun())
            setattr(self, variable_name, comp)

            self.add(comp)

            setattr(self, fun.__name__, partial(center_return, comp))

        # Corner points
        for tb in ['top', 'bottom']:
            for rl in ['right', 'left']:
                name = f"{tb}_{rl}"

                tb_v = getattr(self, f"get_{tb}")()
                rl_v = getattr(self, f"get_{rl}")()

                loc = same_x(rl_v, tb_v)

                comp = VectorizedPoint(location=loc)
                setattr(self, f"p_{name}", comp)

                self.add(comp)

                setattr(self, f"get_{name}", partial(center_return, comp))

        if debug:
            r = 0.1
            for comp in self.submobjects[:]:
                self.add(Circle(radius=r).move_to(comp.get_location()))


class Pins(VGroup):
    def __init__(self, *locations, **kwargs):
        super().__init__(**kwargs)

        for v in locations:
            point = VectorizedPoint(location=v)

            self.add(point)

    def __getitem__(self, item):
        return self.submobjects[item].get_location()


class VisualGate(CircuitShape):
    margin = 0.2

    def __init__(self, gate_type, **kwargs):
        super().__init__(**kwargs)
        self.fill_shape = VGroup()
        self.num_inputs = 2
        self.num_outputs = 1

        self.center_shape = None

        # TODO should probably just make VisualGate be an abstract class or something
        if gate_type == "OR":
            self.fill_shape = ArcPolygon(
                [-1, 0.5, 0], [-1, -0.5, 0], [1, 0, 0],
                arc_config=[{"angle": -1.5}, {"angle": PI / 4}, {"angle": PI / 4}]
            )
        elif gate_type == "AND":
            self.fill_shape = Union(
                Rectangle(height=1.0, width=1.0).shift(LEFT * 0.5),
                Circle(radius=0.5).shift(RIGHT * 0, UP * 0)
            )

            # Cleanup union artifacts
            self.fill_shape = VGroup(self.fill_shape.add_points_as_corners([[-1, 0.5, 0], [-1, -0.5, 0], [0, -0.5, 0]]),
                                     Arc(start_angle=-PI / 2, angle=PI, radius=0.5).shift(RIGHT * 0, UP * 0),
                                     Line([-1, 0.5, 0], [0, 0.5, 0]))
        elif gate_type == "NOT":
            # OR
            t = Triangle().scale(0.5).rotate(-90 * DEGREES)

            self.fill_shape = VGroup(
                Union(t,
                      Circle(radius=0.15).next_to(t.get_right()).shift(LEFT * 0.2, UP * 0),
                      ))

            self.num_inputs = 1

        elif gate_type == "LED":
            self.num_inputs = 1

            # Diode triangle
            diode = Triangle(color=WHITE)
            diode.scale(0.5)
            diode.rotate(-90 * DEGREES)

            # Cathode bar
            cathode = Line(
                start=diode.get_top(),
                end=diode.get_bottom()
            ).next_to(diode.get_right(), buff=0.05)

            # Light emission arrows
            angle = 45
            radius = 0.75
            params = {
                "buff": 0,
                "stroke_width": 4,
                "max_tip_length_to_length_ratio": 0.25
            }
            arrow = np.array([np.cos(angle), np.sin(angle), 0]) * radius

            start = diode.get_top() + DOWN * 0.1

            arrow_1 = Arrow(
                start=start,
                end=start + arrow,
                **params
            )

            start = diode.get_top() + RIGHT * 0.2 + DOWN * 0.2

            arrow_2 = Arrow(
                start=start,
                end=start + arrow,
                **params
            )

            shape = VGroup(diode, cathode)

            self.fill_shape = VGroup(
                shape,
                arrow_1,
                arrow_2,
            )

            self.center_shape = shape

        else:
            raise ValueError("Unknown gate_type")

        if self.center_shape is None:
            self.center_shape = self.fill_shape

        self.local_transform = LocalCoordinate(self.center_shape)

        self.add(self.fill_shape)
        self.add(self.local_transform)

        self.set_active(False)

        self.actual = False
        self.prev_actual = self.actual

        self.input_pins = Pins(self.compute_entry_points(is_input=True))
        self.output_pins = Pins(self.compute_entry_points(is_input=False))

        self.pins = VGroup(self.input_pins, self.output_pins)

        self.add(self.pins)

    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        if self.actual:
            return super().get_critical_point(direction)

        return self.center_shape.get_critical_point(direction)

    def compute_entry_points(self, is_input=True) -> np.ndarray:
        if is_input:
            top = self.local_transform.get_top_left()
            bottom = self.local_transform.get_bottom_left()
            num_vals = self.num_inputs
        else:
            top = self.local_transform.get_top_right()
            bottom = self.local_transform.get_bottom_right()
            num_vals = self.num_outputs

        if num_vals == 1:
            return (top + bottom) / 2

        range_ = top - bottom

        margined_range = range_ * VisualGate.margin

        res = margined_range / 2

        width = np.linspace(top - res, bottom + res, num=self.num_inputs)

        return width

    def get_out(self, n=0):
        return self.output_pins[n]

    def get_in(self, n=0):
        return self.input_pins[n]

    def toggle_actual(self) -> Self:
        self.prev_actual = self.actual
        self.actual = not self.actual

        return self

    def __enter__(self):
        self.prev_actual = self.actual
        self.actual = True

        return self

    def __exit__(self, *exc):
        self.actual = self.prev_actual


class VisualBlock(CircuitShape):
    def __init__(self, label, sub_label: int = 0, width=2.5, height=1.0, **kwargs):
        super().__init__(**kwargs)

        self.rect = Rectangle(width=width, height=height, color=WHITE, fill_color=GREY)
        self.fill_shape = self.rect

        self.text = Text(label, font_size=20).move_to(self.rect.get_center() + UP * 0.2)

        self.current_val = sub_label
        self.val_text = Integer(sub_label, font_size=32, color=BLUE)
        self.val_text.move_to(self.rect.get_center() + DOWN * 0.2)

        if ANIMATE:
            self.val_text.add_updater(lambda m: m.set_value(self.current_val))

        self.add(self.rect, self.text, self.val_text)
        self.set_active(False)

    def update_val(self, val):
        self.current_val = val
        if not ANIMATE:
            self.val_text.set_value(val)
        return None


class VisualWire(CircuitShape):
    def __init__(self, points):
        super().__init__()
        self.line = VMobject()
        self.line.set_points_as_corners(points)
        self.add(self.line)

        self.set_active(False)

    def set_active(self, is_active):
        shape = self.line.animate if ANIMATE else self.line

        return shape.set_color(YELLOW if is_active else GREY).set_stroke(width=6 if is_active else 2)


class VisualResistor(VisualWire):
    def __init__(self):
        x_start = 0

        x_delta = 0.1

        y_switch = 0.1

        points = []

        dir = 1

        steps = 6 + 2

        for n in range(steps):

            last_point = n == steps - 1

            mask = 1 - (n == 0 or last_point)

            points.append([x_start + x_delta * (n - 1 / 2), y_switch * dir * mask, 0])

            if mask:
                dir = -dir

        super().__init__(points)

        self.scale(2.5)


# ─────────────────────────────────────────────────────────────────────────────
# CMOS / Analog circuit primitives
# ─────────────────────────────────────────────────────────────────────────────

class VisualJunction(CircuitShape):
    """Filled dot marking a wire T-junction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dot = Dot(radius=0.08, color=WHITE, fill_opacity=1.0)
        self.add(self.dot)
        self.set_active(False)

    def set_active(self, is_active):
        target = YELLOW if is_active else WHITE
        if ANIMATE:
            return self.dot.animate.set_color(target)
        self.dot.set_color(target)


class VisualVDD(CircuitShape):
    """
    VDD power supply symbol.
    Terminal is at the origin (bottom); symbol extends upward.
    """

    def __init__(self, label: str = "VDD", **kwargs):
        super().__init__(**kwargs)
        stub = Line([0, 0, 0], [0, 0.18, 0], stroke_color=WHITE, stroke_width=2)
        arrow = Arrow(
            [0, 0.18, 0], [0, 0.52, 0],
            buff=0, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.45,
            color=WHITE,
        )
        lbl = Text(label, font_size=13, color=YELLOW)
        lbl.move_to([0, 0.72, 0])
        self.add(stub, arrow, lbl)

        self._terminal = VectorizedPoint([0.0, 0.0, 0.0])
        self.add(self._terminal)
        self.set_active(False)

    def get_terminal(self) -> np.ndarray:
        return self._terminal.get_location()


class VisualGND(CircuitShape):
    """
    Ground symbol.
    Terminal is at the origin (top); symbol extends downward.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stub = Line([0, 0, 0], [0, -0.08, 0], stroke_color=WHITE, stroke_width=2)
        b1 = Line([-0.22, -0.08, 0], [0.22, -0.08, 0], stroke_color=WHITE, stroke_width=2.5)
        b2 = Line([-0.15, -0.18, 0], [0.15, -0.18, 0], stroke_color=WHITE, stroke_width=2.5)
        b3 = Line([-0.08, -0.28, 0], [0.08, -0.28, 0], stroke_color=WHITE, stroke_width=2.5)
        self.add(stub, b1, b2, b3)

        self._terminal = VectorizedPoint([0.0, 0.0, 0.0])
        self.add(self._terminal)
        self.set_active(False)

    def get_terminal(self) -> np.ndarray:
        return self._terminal.get_location()


class VisualCapacitor(CircuitShape):
    """
    Capacitor symbol (two parallel plates), oriented vertically.
    Positive terminal at top, negative at bottom.
    """

    def __init__(self, label: str = "", **kwargs):
        super().__init__(**kwargs)
        top_wire = Line([0, 0.32, 0], [0, 0.07, 0], stroke_color=WHITE, stroke_width=2)
        top_plate = Line([-0.22, 0.07, 0], [0.22, 0.07, 0], stroke_color=WHITE, stroke_width=3.5)
        bot_plate = Line([-0.22, -0.07, 0], [0.22, -0.07, 0], stroke_color=WHITE, stroke_width=3.5)
        bot_wire = Line([0, -0.07, 0], [0, -0.32, 0], stroke_color=WHITE, stroke_width=2)
        self.add(top_wire, top_plate, bot_plate, bot_wire)

        if label:
            lbl = Text(label, font_size=12, color=WHITE)
            lbl.move_to([0.38, 0, 0])
            self.add(lbl)

        self._pos_terminal = VectorizedPoint([0, 0.32, 0])
        self._neg_terminal = VectorizedPoint([0, -0.32, 0])
        self.add(self._pos_terminal, self._neg_terminal)
        self.set_active(False)

    def get_pos_terminal(self) -> np.ndarray:
        return self._pos_terminal.get_location()

    def get_neg_terminal(self) -> np.ndarray:
        return self._neg_terminal.get_location()


class VisualPhotodiode(CircuitShape):
    """
    Photodiode symbol, oriented vertically with cathode at top, anode at bottom.
    Includes two diagonal photon arrows indicating light absorption.

    Pin access:
      get_cathode() → top terminal
      get_anode()   → bottom terminal
    """

    def __init__(self, show_photons: bool = True, **kwargs):
        super().__init__(**kwargs)

        # Triangle (anode at base/bottom, cathode at tip/top)
        tri = Polygon(
            [-0.18, -0.15, 0],
            [0.18, -0.15, 0],
            [0, 0.15, 0],
            stroke_color=WHITE, stroke_width=2,
            fill_color=WHITE, fill_opacity=0.15,
        )
        # Cathode bar (horizontal line at top of triangle)
        bar = Line([-0.22, 0.15, 0], [0.22, 0.15, 0], stroke_color=WHITE, stroke_width=2.5)
        # Terminal wires
        top_wire = Line([0, 0.15, 0], [0, 0.38, 0], stroke_color=WHITE, stroke_width=2)
        bot_wire = Line([0, -0.15, 0], [0, -0.38, 0], stroke_color=WHITE, stroke_width=2)
        self.fill_shape = tri
        self.add(tri, bar, top_wire, bot_wire)

        if show_photons:
            ph_params = dict(buff=0, stroke_width=2.5,
                             max_tip_length_to_length_ratio=0.4, color=YELLOW)
            ph1 = Arrow([-0.38, 0.60, 0], [-0.16, 0.32, 0], **ph_params)
            ph2 = Arrow([-0.20, 0.72, 0], [0.02, 0.44, 0], **ph_params)
            self.add(ph1, ph2)

        self._cathode = VectorizedPoint([0, 0.38, 0])
        self._anode   = VectorizedPoint([0, -0.38, 0])
        self.add(self._cathode, self._anode)
        self.set_active(False)

    def get_cathode(self) -> np.ndarray:
        return self._cathode.get_location()

    def get_anode(self) -> np.ndarray:
        return self._anode.get_location()


class VisualNMOS(CircuitShape):
    """
    Standard NMOS enhancement-mode transistor symbol.

    Local coordinate layout (before scale / move_to):

         D  [0.18, 0.55, 0]
         |
    G ───||  (gate line left, channel line right, ~0.08 gap)
    [-0.42, 0, 0]
         |
         ↑  (N-channel arrow on source stub)
         S  [0.18, -0.55, 0]

    Pin access:
      get_gate()   → leftmost gate terminal
      get_drain()  → top terminal
      get_source() → bottom terminal
    """

    # Geometry constants (local coords before any scaling)
    _BODY_X   = 0.00   # x of body (channel) line
    _BODY_HH  = 0.28   # half-height of body line
    _GATE_X   = -0.10  # x of gate electrode line (0.10 gap from body)
    _GATE_EXT = -0.42  # x of gate terminal (left end of gate stub)
    _STUB_X   = 0.18   # x of drain/source terminal wires
    _EXT_H    = 0.27   # additional height above/below stubs for terminal wires

    def __init__(self, label: str = "", **kwargs):
        super().__init__(**kwargs)

        bx, bhh = self._BODY_X, self._BODY_HH
        gx, ge  = self._GATE_X, self._GATE_EXT
        sx, eh  = self._STUB_X, self._EXT_H

        # Channel (body) — vertical bar
        body = Line([bx, -bhh, 0], [bx, bhh, 0], stroke_color=WHITE, stroke_width=3)

        # Gate electrode — parallel to body, separated by gap
        gate_line = Line([gx, -bhh * 0.85, 0], [gx, bhh * 0.85, 0],
                         stroke_color=WHITE, stroke_width=2.5)

        # Gate horizontal stub → external terminal
        gate_stub = Line([ge, 0, 0], [gx, 0, 0], stroke_color=WHITE, stroke_width=2)

        # Drain stub (horizontal from body to right)
        drain_stub = Line([bx, bhh, 0], [sx, bhh, 0], stroke_color=WHITE, stroke_width=2)
        # Drain terminal (vertical up)
        drain_wire = Line([sx, bhh, 0], [sx, bhh + eh, 0], stroke_color=WHITE, stroke_width=2)

        # Source stub (horizontal from body to right)
        src_stub = Line([bx, -bhh, 0], [sx, -bhh, 0], stroke_color=WHITE, stroke_width=2)
        # Source terminal (vertical down)
        src_wire = Line([sx, -bhh, 0], [sx, -(bhh + eh), 0], stroke_color=WHITE, stroke_width=2)

        # N-channel arrow on source stub (small arrow pointing LEFT → toward body)
        mid_src_x = (bx + sx) / 2 + 0.02
        n_arrow = Arrow(
            [mid_src_x + 0.10, -bhh, 0], [mid_src_x - 0.06, -bhh, 0],
            buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.7,
            color=WHITE,
        )

        self.add(body, gate_line, gate_stub, drain_stub, drain_wire,
                 src_stub, src_wire, n_arrow)

        # Optional identifier label (placed left of gate)
        if label:
            lbl = Text(label, font_size=13, color=WHITE)
            lbl.move_to([ge - 0.22, 0.22, 0])
            self.add(lbl)

        # Pin VectorizedPoints (move with the VGroup on scale/move_to)
        self._gate_pt   = VectorizedPoint([ge, 0, 0])
        self._drain_pt  = VectorizedPoint([sx, bhh + eh, 0])
        self._source_pt = VectorizedPoint([sx, -(bhh + eh), 0])
        self.add(self._gate_pt, self._drain_pt, self._source_pt)

        self.set_active(False)

    def get_gate(self) -> np.ndarray:
        return self._gate_pt.get_location()

    def get_drain(self) -> np.ndarray:
        return self._drain_pt.get_location()

    def get_source(self) -> np.ndarray:
        return self._source_pt.get_location()


def same_y(target, val):
    val = val[:]
    val[1] = target[1]

    return val


def same_x(target, val):
    val = val[:]
    val[0] = target[0]

    return val


def set_animate(animate: bool):
    global ANIMATE
    ANIMATE = animate


def get_animate():
    return ANIMATE
