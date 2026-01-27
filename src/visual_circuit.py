from manim import *

ANIMATE = True


class CircuitShape(VGroup):
    def set_active(self, is_active):
        target_color = YELLOW if is_active else WHITE

        stroke = self.animate.set_stroke(color=target_color) if ANIMATE else self.set_stroke(color=target_color)

        if hasattr(self, "fill_shape"):
            self.fill_shape: VGroup

            shape = self.fill_shape.animate if ANIMATE else self.fill_shape

            return stroke, shape.set_fill(color=target_color, opacity=0.8 if is_active else 0.2)


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


class VisualGate(CircuitShape):
    margin = 0.2

    def __init__(self, gate_type, **kwargs):
        super().__init__(**kwargs)
        self.fill_shape = VGroup()
        self.num_inputs = 2

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
        else:
            # OR
            t = Triangle().scale(0.5).rotate(-90 * DEGREES)

            self.fill_shape = VGroup(
                Union(t,
                      Circle(radius=0.15).next_to(t.get_right()).shift(LEFT * 0.2, UP * 0),
                      ))

            self.num_inputs = 1

        self.add(self.fill_shape)
        self.set_active(False)

    def get_out(self):
        return self.get_right()

    def get_in(self, n=0):
        top_left = same_x(self.get_left(), self.get_top())
        bottom_left = same_x(self.get_left(), self.get_bottom())

        if self.num_inputs == 1:
            return (top_left + bottom_left) / 2

        range_ = top_left - bottom_left

        margined_range = range_ * VisualGate.margin

        res = margined_range / 2

        width = np.linspace(top_left - res, bottom_left + res, num=self.num_inputs)

        return width[n]


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


class VisualWire(VGroup):
    def __init__(self, points):
        super().__init__()
        self.line = VMobject()
        self.line.set_points_as_corners(points)
        self.add(self.line)

        self.set_active(False)

    def set_active(self, is_active):
        shape = self.line.animate if ANIMATE else self.line

        return shape.set_color(YELLOW if is_active else GREY).set_stroke(width=6 if is_active else 2)


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
