from manim import *

from circuit import Circuit, Clock
from ti_timer import LogicGate, NOT, Register, CPUTimerCounter


class CircuitShape(VGroup):
    def set_active(self, is_active):
        target_color = YELLOW if is_active else WHITE
        self.set_stroke(color=target_color)
        if hasattr(self, "fill_shape"):
            self.fill_shape: VGroup

            self.fill_shape.set_fill(color=target_color, opacity=0.8 if is_active else 0.2)
            # self.fill_shape.set_color(color=target_color)


class VisualGroup(CircuitShape):
    def __init__(self, *components):
        super().__init__(*components)
        self.components = components

    def set_active(self, is_active):
        for c in self.components:
            c.set_active(is_active)

class VisualGate(CircuitShape):
    margin = 0.2

    def __init__(self, gate_type, **kwargs):
        super().__init__(**kwargs)
        self.fill_shape = VGroup()
        self.num_inputs = 2

        if gate_type == "OR":
            self.fill_shape = ArcPolygon(
                [-1, 0.5, 0], [-1, -0.5, 0], [1, 0, 0], arc_config=[{"angle": -1.5}, {"angle": PI / 4}, {"angle": PI / 4}]
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
                t,
                Circle(radius=0.15).next_to(t.get_right()).shift(LEFT * 0.2, UP * 0),
            )

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
    def __init__(self, label, sub_label="", width=2.5, height=1.0, **kwargs):
        super().__init__(**kwargs)
        self.rect = Rectangle(width=width, height=height, color=WHITE, fill_color=GREY)
        self.fill_shape = self.rect  # For lighting up
        self.text = Text(label, font_size=20).move_to(self.rect.get_center() + UP * 0.2)
        self.val_text = Text(sub_label, font_size=18, color=BLUE).move_to(self.rect.get_center() + DOWN * 0.2)
        self.add(self.rect, self.text, self.val_text)

        self.set_active(False)

    def update_val(self, val):
        self.val_text.become(Text(str(val), font_size=18, color=BLUE).move_to(self.val_text.get_center()))


class VisualWire(VGroup):
    def __init__(self, points):
        super().__init__()
        self.line = VMobject()
        self.line.set_points_as_corners(points)
        self.add(self.line)

        self.set_active(False)

    def set_active(self, is_active):
        self.line.set_color(YELLOW if is_active else GREY).set_stroke(width=6 if is_active else 2)


def same_y(target, val):
    val = val[:]
    val[1] = target[1]

    return val


def same_x(target, val):
    val = val[:]
    val[0] = target[0]

    return val


class CPUTimerAnimation(Scene):
    def construct(self):
        # --- A. Setup Simulation ---
        c = Circuit()

        period = 2

        # Components
        clk = Clock("SYSCLK", period=period)
        reset_or = LogicGate("Reset_OR", lambda a, b: a or b)
        pre_or = LogicGate("Pre_OR", lambda a, b: a or b)
        main_or = LogicGate("Main_OR", lambda a, b: a or b)
        gate_and = LogicGate("Gate_AND", lambda a, b: a and b)
        inv = NOT("INV")

        tddr = Register("TDDR", 2)
        psc = CPUTimerCounter("PSC", 2)
        prd = Register("PRD", 5)
        tim = CPUTimerCounter("TIM", 5)

        for comp in [clk, reset_or, pre_or, main_or, gate_and, inv, tddr, psc, prd, tim]:
            c.add(comp)

        # Wiring (Topology)
        c.connect("SYSCLK", "clk", "Gate_AND", "A")
        c.connect("INV", "out", "Gate_AND", "B")
        c.connect("Gate_AND", "out", "PSC", "clk")

        c.connect("Reset_OR", "out", "Pre_OR", "A")
        c.connect("Reset_OR", "out", "Main_OR", "A")

        c.connect("PSC", "borrow", "Pre_OR", "B")
        c.connect("Pre_OR", "out", "PSC", "load")
        c.connect("PSC", "borrow", "TIM", "clk")  # Cascade

        c.connect("TIM", "borrow", "Main_OR", "B")
        c.connect("Main_OR", "out", "TIM", "load")

        c.connect("TDDR", "out", "PSC", "din")
        c.connect("PRD", "out", "TIM", "din")

        # Initial Pokes
        c.poke("INV", "in", 0)  # TCR.4 = 0 (Enabled)
        c.poke("Reset_OR", "A", 0)
        c.poke("Reset_OR", "B", 0)
        c.schedule("SYSCLK", 0)

        # --- B. Layout Visuals ---
        # Coordinates map roughly to the diagram provided

        # Blocks
        v_tddr = VisualBlock("TDDR", "2").move_to([-1, 0.5, 0])
        v_psc = VisualBlock("PSC", "2").move_to([-1, -1, 0])
        v_prd = VisualBlock("PRD", "5").move_to([4, 0.5, 0])
        v_tim = VisualBlock("TIM", "5").move_to([4, -1, 0])

        # Gates
        v_pre_or = VisualGate("OR").scale(0.5).move_to([-1.75, 2, 0])
        v_reset_or = VisualGate("OR").scale(0.5).move_to(v_pre_or.get_in(0) + LEFT * 1.5)
        v_main_or = VisualGate("OR").scale(0.5).move_to([3.25, 2, 0])
        v_and = VisualGate("AND").scale(0.5).next_to(v_psc, LEFT, buff=1)

        v_not = VisualGate("NOT").scale(0.5).move_to(v_and.get_in(1) + LEFT)

        t_sysclk = Text("SYSCLK", font_size=16).move_to(v_and.get_in(0) + LEFT)
        t_tcr = Text("TCR.4", font_size=16).move_to(v_not.get_in(0) + LEFT)

        t_reset = Text("Reset", font_size=16).move_to(v_reset_or.get_in(0) + LEFT)
        t_timer_reload = Text("Timer reload", font_size=16).move_to(v_reset_or.get_in(1) + LEFT)
        t_tint = Text("TINT", font_size=16).next_to(v_tim, RIGHT, buff=1)

        # Labels
        labels = VGroup(
            Text("CPU-Timer", font_size=36).to_edge(UP),
            t_sysclk,
            t_reset,
            t_tint,
            t_timer_reload,
            t_tcr
        )

        # Wires (Hardcoded paths to look like diagram)
        # Using [Start, Corner1, Corner2, End] format
        w_sysclk = VisualWire([t_sysclk.get_right(),
                               v_and.get_in(0)])
        w_tcr_not = VisualWire([t_tcr.get_right(),
                               v_not.get_in(0)])

        w_tcr_and = VisualWire([v_not.get_out(),
                            v_and.get_in(1)])

        w_reset = VisualWire([t_reset.get_right(),
                              v_reset_or.get_in(0)])
        w_timer_reload = VisualWire([t_timer_reload.get_right(),
                              v_reset_or.get_in(1)])


        w_reset_out = VisualWire([v_reset_or.get_out(), v_pre_or.get_in(0)])


        temp_x = (v_reset_or.get_out() + v_pre_or.get_in(0)) / 2
        temp = temp_x + UP * 0.5

        w_reset_branch = VisualWire(
            [v_reset_or.get_out(),
             temp_x,
             temp,
             same_y(temp, v_main_or.get_in(0) + LEFT),
             v_main_or.get_in(0) + LEFT,
             v_main_or.get_in(0)])  # Long wire over top

        w_and_out = VisualWire(
            [v_and.get_out(), v_psc.get_left()])  # Clock input to PSC

        temp = v_psc.get_bottom() + DOWN * 0.5
        temp_x = same_y(temp, v_psc.get_left() + LEFT * 0.5)

        w_psc_feedback = VisualWire(
            [v_psc.get_bottom(), same_y(temp, v_psc.get_bottom()), temp_x,
             same_x(temp_x, v_pre_or.get_in(1)),
             v_pre_or.get_in(1)])

        w_psc_cascade = VisualWire([v_psc.get_bottom(),
                                    temp,
                                    same_y(temp, v_tim.get_left() + DOWN + LEFT),
                                    v_tim.get_left() + LEFT,
                                    v_tim.get_left()])

        w_pre_load = VisualWire([v_pre_or.get_out(),
                                 same_x(v_psc.get_top(), v_pre_or.get_out()),
                                 v_psc.get_top()])

        w_tim_borrow = VisualWire([v_tim.get_right(), t_tint.get_left()])

        temp = same_y(v_main_or.get_in(1), v_main_or.get_left() + LEFT * 2)

        w_tim_feedback = VisualWire(
            [v_tim.get_bottom(), v_tim.get_bottom() + DOWN, same_x(temp, v_tim.get_bottom() + DOWN + LEFT), temp,
             v_main_or.get_in(1)])
        w_main_load = VisualWire([v_main_or.get_out(), same_x(v_tim.get_top(), v_main_or.get_out()), v_tim.get_top()])

        # Mapping: Simulation Signal Name -> Visual Object
        # This dict tells the animation loop what to light up when a signal is 1


        signal_map = {
            ("SYSCLK", "clk"): w_sysclk,
            ("Reset_OR", "A"): w_reset,
            ("Reset_OR", "B"): w_timer_reload,
            ("INV", "in"): w_tcr_not,
            ("Gate_AND", "out"): VisualGroup(w_and_out, v_and),
            ("Reset_OR", "out"): VisualGroup(w_reset_out, w_reset_branch, v_reset_or),
            ("Pre_OR", "out"): VisualGroup(w_pre_load, v_pre_or),
            ("PSC", "borrow"): VisualGroup(w_psc_feedback, w_psc_cascade),
            ("Main_OR", "out"): VisualGroup(w_main_load, v_main_or),
            ("TIM", "borrow"): VisualGroup(w_tim_borrow, w_tim_feedback),
            ("INV", "out"): VisualGroup(w_tcr_and, v_not)
        }

        component_map = {
            "PSC": v_psc,
            "TIM": v_tim,
        }

        # Add everything to scene
        self.add(labels)
        self.add(w_sysclk, w_reset, w_tcr_not, w_tcr_and, w_timer_reload, w_reset_out, w_reset_branch, w_and_out,
                 w_psc_feedback, w_psc_cascade, w_pre_load,
                 w_tim_borrow, w_tim_feedback, w_main_load,


                 v_tddr, v_psc, v_prd, v_tim, v_reset_or, v_pre_or, v_main_or, v_and, v_not

                 )

        # 1. Reset Pulse
        self.wait(1)
        c.poke("Reset_OR", "A", 1)  # Trigger Reset

        # We run the simulation in small steps and update visuals
        for step in range(60 * 1):
            c.run(steps=period // 2)

            # Reset logic (Pulse reset for a few frames then drop)
            if step == 2: c.poke("Reset_OR", "A", 0)

            # Update Wires
            anims = []
            for (comp_name, sig_name), visual_wire in signal_map.items():
                visual_wire: VisualWire

                val = c.components[comp_name].signals[sig_name].value
                visual_wire.set_active(True if val else False)

            # Update Counters text
            for comp_name, visual_block in component_map.items():
                val = c.components[comp_name].signals["out"].value
                visual_block.update_val(val)
                # Flash block if loading
                is_loading = c.components[comp_name].signals["load"].value
                visual_block.set_active(is_loading)

            self.wait(1)
