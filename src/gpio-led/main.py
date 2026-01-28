import sys

sys.path.append('../')

from manim import *

from ti_timer import NOT, LED
from visual_circuit import VisualBlock, VisualGate, VisualWire, VisualGroup, get_animate, set_animate, same_y, VisualResistor

from circuit import Circuit, Clock


class GPIOLED(Scene):
    def construct(self):
        set_animate(True)

        period = 12

        c = Circuit(delay=1)

        gpio = Clock("GPIO", period=period)
        inv = NOT("INV")
        led = LED("LED")

        for comp in [gpio, inv, led]:
            c.add(comp)

        c.connect("GPIO", "clk", "INV", "in")
        c.connect("INV", "out", "LED", "out")

        c.poke("LED", "in", 1)
        c.poke("GPIO", "clk", 1)

        c.schedule("GPIO", 0)

        v_gpio = VisualBlock("GPIO", 31).shift(LEFT)

        v_not = VisualGate("NOT").move_to(v_gpio.get_right() + RIGHT * 1.5)

        # FIXME make this work better, the rotation is not proper
        v_led = VisualGate("LED").move_to(v_not.get_right() + RIGHT * 1.5)
        v_led.rotate(180 * DEGREES)
        v_led.flip(LEFT)

        v_resistor = VisualResistor().rotate(90 * DEGREES)
        v_resistor.move_to(v_led.get_in() + UP * 1.5 + RIGHT)

        w_gpio_not = VisualWire([
            v_gpio.get_right(), v_not.get_in()
        ])

        w_not_led = VisualWire([
            v_not.get_out(), v_led.get_out()
        ])

        up_mul = 10

        with v_led as v_led_ac:
            t_led = Text("LED").next_to(v_led_ac, UP)

        print(v_led.actual)

        t_3_3v_not = Text("3.3V").next_to(v_not, UP * up_mul)
        t_3_3v_led = Text("3.3V").move_to(v_led.get_in()).shift(RIGHT)
        t_3_3v_led.shift(t_3_3v_not.get_bottom() - same_y(t_3_3v_led.get_bottom(), t_3_3v_not.get_bottom()))

        t_ohms = Text("1kΩ").rotate(-90 * DEGREES).next_to(v_resistor, RIGHT)

        w_not_power = VisualWire([v_not.get_center() + UP * 0.1, t_3_3v_not.get_bottom()])
        w_led_power = VisualWire([
            v_led.get_in() , v_led.get_in()  + RIGHT, v_resistor.get_bottom()
        ])

        w_res_power = VisualWire([
            v_resistor.get_top(), t_3_3v_led.get_bottom()
        ])

        all_ = [
            t_led, t_3_3v_not, t_3_3v_led, t_ohms,

            w_not_power, w_res_power,
            w_gpio_not,

            w_not_led, w_led_power,

            v_gpio, v_not, v_led, v_resistor
        ]

        signal_map: dict[tuple[str, str], VisualGroup] = {
            ("GPIO", "clk"): VisualGroup(w_gpio_not, v_gpio),
            ("INV", "out"): VisualGroup(w_not_led, v_not),
            ("LED", "state"): VisualGroup(v_led),
        }

        # Centers everything
        all = VGroup(all_).center()

        t_led = Text("GPIO LED Control").next_to(all, UP, buff=1)

        all_.append(t_led)

        self.play(*map(Create, all_), run_tim=5)
        self.add(*all_)

        self.wait(1)

        animations = []

        animations.append(w_led_power.set_active(True))
        animations.append(w_not_power.set_active(True))
        animations.append(v_resistor.set_active(True))
        animations.append(w_res_power.set_active(True))

        # Set Inverter and LED to be proper initial state
        # 1 step to set the inverter
        # 1 step to set the LED
        c.run(steps=2)

        for step in range(20):
            c.run(steps=1)

            # Update Wires
            for (comp_name, sig_name), visual_wire in signal_map.items():
                visual_wire: VisualGroup

                val = c.components[comp_name].signals[sig_name].value
                animations.extend(visual_wire.set_active(True if val else False))

            if get_animate() and animations:
                self.play(*animations)

            animations.clear()

            self.wait(3)


if __name__ == '__main__':
    GPIOLED().construct()
