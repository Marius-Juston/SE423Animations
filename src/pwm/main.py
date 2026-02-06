from manim import *


class PWMAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Pulse Width Modulation (PWM)", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create axes for PWM signal
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.2, 0.5],
            x_length=8,
            y_length=2.5,
            axis_config={"color": BLUE},
            tips=False
        ).shift(UP * 0.5)

        # Labels
        x_label = Text("Time", font_size=24).next_to(axes.x_axis, DOWN)
        y_label = Text("Signal", font_size=24).next_to(axes.y_axis, LEFT).rotate(PI / 2)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # LED representation
        led_circle = Circle(radius=0.6, color=RED, fill_opacity=0.1).shift(DOWN * 2)
        led_label = Text("LED", font_size=28).next_to(led_circle, DOWN)

        self.play(Create(led_circle), Write(led_label))
        self.wait(0.5)

        # Function to create PWM signal
        def create_pwm_signal(duty_cycle, periods=4):
            points = []
            for i in range(periods):
                # High period
                points.extend([
                    axes.c2p(i, 0),
                    axes.c2p(i, 1),
                    axes.c2p(i + duty_cycle, 1),
                    axes.c2p(i + duty_cycle, 0),
                ])
                # Low period
                if duty_cycle < 1:
                    points.extend([
                        axes.c2p(i + duty_cycle, 0),
                        axes.c2p(i + 1, 0),
                    ])
            return VMobject().set_points_as_corners(points).set_color(YELLOW)

        # Start with 50% duty cycle
        duty_cycle = 0.5
        pwm_signal = create_pwm_signal(duty_cycle)
        duty_text = Text(f"Duty Cycle: {int(duty_cycle * 100)}%", font_size=32)
        duty_text.next_to(axes, UP, buff=0.5)

        self.play(Create(pwm_signal), Write(duty_text))

        # Create tracking dot
        dot = Dot(color=GREEN, radius=0.08).move_to(axes.c2p(0, 0))
        self.play(FadeIn(dot))

        # Animate the dot following the signal with LED response
        def update_led(mob, alpha):
            x_pos = alpha * 4
            # Determine if signal is high or low
            cycle_pos = x_pos % 1
            is_high = cycle_pos < duty_cycle

            if is_high:
                mob.set_fill(opacity=0.8)
                mob.set_color(YELLOW)
            else:
                mob.set_fill(opacity=0.1)
                mob.set_color(RED)

        # First animation: 50% duty cycle
        self.play(
            MoveAlongPath(dot, pwm_signal, rate_func=linear),
            UpdateFromAlphaFunc(led_circle, update_led),
            run_time=4,
            rate_func=linear
        )
        self.wait(0.5)

        # Transition to 25% duty cycle
        duty_cycle = 0.25
        new_pwm_signal = create_pwm_signal(duty_cycle)
        new_duty_text = Text(f"Duty Cycle: {int(duty_cycle * 100)}%", font_size=32)
        new_duty_text.next_to(axes, UP, buff=0.5)

        brightness_label = Text("Brightness: Lower", font_size=24, color=ORANGE)
        brightness_label.next_to(led_label, DOWN)

        self.play(
            Transform(pwm_signal, new_pwm_signal),
            Transform(duty_text, new_duty_text),
            Write(brightness_label),
            dot.animate.move_to(axes.c2p(0, 0))
        )

        def update_led_25(mob, alpha):
            x_pos = alpha * 4
            cycle_pos = x_pos % 1
            is_high = cycle_pos < duty_cycle

            if is_high:
                mob.set_fill(opacity=0.4)
                mob.set_color(YELLOW)
            else:
                mob.set_fill(opacity=0.1)
                mob.set_color(RED)

        self.play(
            MoveAlongPath(dot, new_pwm_signal, rate_func=linear),
            UpdateFromAlphaFunc(led_circle, update_led_25),
            run_time=4,
            rate_func=linear
        )
        self.wait(0.5)

        # Transition to 75% duty cycle
        duty_cycle = 0.75
        new_pwm_signal_75 = create_pwm_signal(duty_cycle)
        new_duty_text_75 = Text(f"Duty Cycle: {int(duty_cycle * 100)}%", font_size=32)
        new_duty_text_75.next_to(axes, UP, buff=0.5)

        new_brightness_label = Text("Brightness: Higher", font_size=24, color=ORANGE)
        new_brightness_label.next_to(led_label, DOWN)

        self.play(
            Transform(pwm_signal, new_pwm_signal_75),
            Transform(duty_text, new_duty_text_75),
            Transform(brightness_label, new_brightness_label),
            dot.animate.move_to(axes.c2p(0, 0))
        )

        def update_led_75(mob, alpha):
            x_pos = alpha * 4
            cycle_pos = x_pos % 1
            is_high = cycle_pos < duty_cycle

            if is_high:
                mob.set_fill(opacity=0.95)
                mob.set_color(YELLOW)
            else:
                mob.set_fill(opacity=0.1)
                mob.set_color(RED)

        self.play(
            MoveAlongPath(dot, new_pwm_signal_75, rate_func=linear),
            UpdateFromAlphaFunc(led_circle, update_led_75),
            run_time=4,
            rate_func=linear
        )
        self.wait(0.5)

        # Transition to macroscopic view
        self.play(
            FadeOut(brightness_label),
            FadeOut(dot)
        )

        transition_text = Text("Zooming out to human perception...", font_size=36, color=BLUE)
        transition_text.to_edge(DOWN)
        self.play(Write(transition_text))
        self.wait(1)
        self.play(FadeOut(transition_text))

        # Transform axes to show zoom out effect
        self.play(
            FadeOut(pwm_signal),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(duty_text),
            FadeOut(led_circle), 
            FadeOut(led_label)
        )
        self.wait(0.5)

        perception_text = Text("At high frequencies, human eye sees average brightness",
                               font_size=28).to_edge(DOWN, buff=1)
        self.play(Write(perception_text))

        # Show three LEDs with different duty cycles side by side
        led_positions = [LEFT * 3.5, ORIGIN, RIGHT * 3.5]
        duty_cycles = [0.25, 0.5, 0.75]
        led_labels_text = ["25% Duty", "50% Duty", "75% Duty"]
        brightness_levels = [0.25, 0.5, 0.75]

        leds = VGroup()
        led_texts = VGroup()

        for pos, duty, label_text, brightness in zip(led_positions, duty_cycles, led_labels_text, brightness_levels):
            # LED circle with constant brightness (average of PWM)
            led = Circle(radius=0.5, color=YELLOW, fill_opacity=brightness,
                         stroke_width=2, stroke_color=WHITE).shift(pos)

            # Label
            led_text = Text(label_text, font_size=22).next_to(led, UP, buff=0.3)

           
            leds.add(led)
            led_texts.add(led_text)
            
        # Show brightness bars
        bar_label = Text("Perceived Brightness", font_size=20).next_to(led_texts, UP)

        # Animate LEDs appearing with their perceived brightness
        self.play(
            LaggedStart(*[FadeIn(led) for led in leds], lag_ratio=0.3),
            LaggedStart(*[Write(text) for text in led_texts], lag_ratio=0.3),
            Write(bar_label),
            run_time=2
        )

        self.wait(1)

        # Add microscopic view reminder
        micro_reminder = Text(
            "Microscopic: Pulses ON/OFF rapidly",
            font_size=22,
            color=RED
        ).to_edge(UP, buff=1.5)

        macro_reminder = Text(
            "Macroscopic: Appears as steady brightness",
            font_size=22,
            color=GREEN
        ).next_to(micro_reminder, DOWN, buff=0.2)

        self.play(
            FadeOut(perception_text),
            Write(micro_reminder),
            Write(macro_reminder)
        )
        self.wait(2)

        self.play(
            FadeOut(title),
            FadeOut(micro_reminder),
            FadeOut(macro_reminder),
            FadeOut(leds),
            FadeOut(led_texts),
            FadeOut(bar_label),
        )

        self.wait(1)

# To render this animation, save it to a file (e.g., pwm_animation.py) and run:
# manim -pql pwm_animation.py PWMAnimation
#
# Options:
# -p : preview (auto-play after rendering)
# -q l : quality low (for faster rendering)
# -q h : quality high (for better output)
# -q k : quality 4k (for best output)
