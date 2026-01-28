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
