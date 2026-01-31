from manim import *


def bits_to_string(x: int, n: int) -> str:
    """Return string of bits for an unsigned value x in n bits."""
    return format(x & ((1 << n) - 1), f"0{n}b")


def bit_boxes(bit_string: str, bit_size: float = 0.5, buff: float = 0.05) -> VGroup:
    """Returns a VGroup of square boxes with the bits centered inside."""
    boxes = VGroup()
    for ch in bit_string:
        sq = Square(side_length=bit_size, stroke_width=2, color=BLUE)
        bit = Text(ch, font_size=24)
        bit.move_to(sq.get_center())
        boxes.add(VGroup(sq, bit))
    boxes.arrange(RIGHT, buff=buff)
    return boxes

def same_y(target, val):
    val = val[:]
    val[1] = target[1]

    return val


class TwoComplement(Scene):
    def construct(self):
        # Introduction with animated title
        intro = Text("Understanding Two's Complement", font_size=40, gradient=(BLUE, GREEN))
        self.play(Write(intro, run_time=1.5))
        self.play(intro.animate.scale(1.1).set_color(YELLOW), run_time=0.5)
        self.play(intro.animate.scale(1 / 1.1).set_color(WHITE), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(intro, shift=UP))

        # Step 1: Modular arithmetic concept with animations
        title1 = Text("Two's Complement uses Modular Arithmetic", font_size=36)
        title1.to_edge(UP)
        self.play(Write(title1, run_time=1))
        self.wait(0.5)

        concept = VGroup(
            MathTex(r"\text{For } N \text{ bits, range is } [0, 2^N - 1]"),
            MathTex(r"\text{Define } -x \text{ as the value } y \text{ where:}"),
            MathTex(r"x + y \equiv 0 \pmod{2^N}")
        ).arrange(DOWN, buff=0.5)
        concept.shift(DOWN * 0.5)

        # Animate each line appearing
        for item in concept:
            self.play(Write(item, run_time=1))
            self.wait(0.5)

        self.wait(2)
        self.play(FadeOut(concept, shift=DOWN))

        # Step 2: Derivation with smooth transitions
        deriv_title = Text("Derivation", font_size=36, color=YELLOW)
        deriv_title.to_edge(UP)
        self.play(Transform(title1, deriv_title, run_time=1))
        self.wait(0.5)

        # Show derivation steps with better positioning
        step1 = MathTex(r"-x \equiv 2^N - x \pmod{2^N}")
        step1.shift(UP * 2)
        self.play(Write(step1, run_time=1))
        self.play(Indicate(step1, color=YELLOW))
        self.wait(2)

        down_scale = 1

        arrow_kwargs = dict(buff=0.1, color=GREEN, max_stroke_width_to_length_ratio=30,
                            max_tip_length_to_length_ratio=10, stroke_width=10)

        step2 = MathTex(r"2^N - x = (2^N - 1) - x + 1")
        step2.next_to(step1, DOWN, buff=down_scale)

        # Arrow to next step
        arrow1 = Arrow(start=step1.get_bottom(), end=step2.get_top(), **arrow_kwargs)

        self.play(Write(step2, run_time=1.5))
        self.play(GrowArrow(arrow1))

        # Highlight the key parts
        highlight_box = SurroundingRectangle(step2[0][4:11], color=YELLOW, buff=0.05)  # Highlight (2^N - 1)
        self.play(Create(highlight_box))
        self.wait(2)
        self.play(FadeOut(highlight_box))

        # Highlight that 2^N - 1 is all ones
        all_ones = MathTex(r"2^N - 1 = \underbrace{111...111}_{N \text{ ones}}")
        all_ones.next_to(step2, DOWN, buff=down_scale)

        # Arrow to next step
        arrow2 = Arrow(start=step2.get_bottom(), end=all_ones.get_top(), **arrow_kwargs)
        self.play(GrowArrow(arrow2))

        self.play(Write(all_ones, run_time=1.5))
        self.play(Circumscribe(all_ones, color=BLUE, run_time=1.5))
        self.wait(2)

        # VISUAL DEMONSTRATION: Show (2^N - 1) - x = ~x with actual bits
        demo_explanation = Text("Let's see this with actual bits!", font_size=24, color=YELLOW)
        demo_explanation.next_to(all_ones, DOWN, buff=down_scale)

        # Arrow to next step
        arrow3 = Arrow(start=all_ones.get_bottom(), end=demo_explanation.get_top(), **arrow_kwargs)
        self.play(GrowArrow(arrow3))

        self.play(Write(demo_explanation))
        self.wait(1)

        # Clear some space for the bit demonstration
        self.play(
            FadeOut(step1), FadeOut(arrow1), FadeOut(step2),
            FadeOut(arrow2), FadeOut(all_ones), FadeOut(arrow3),
            FadeOut(demo_explanation)
        )

        # Show bit-level demonstration
        bit_demo_title = Text("(2^N - 1) - x = ~x    [Bit-level]", font_size=28, color=BLUE)
        bit_demo_title.shift(UP * 2.5)
        self.play(Write(bit_demo_title))

        # Example: N=8, x=5 (easier to see)
        N_demo = 8
        x_demo = 5
        all_ones_bits = "11111111"
        x_bits = bits_to_string(x_demo, N_demo)

        # Show (2^N - 1) in binary
        all_ones_label = MathTex(r"2^8 - 1 = 255_{10} =")
        all_ones_boxes = bit_boxes(all_ones_bits, bit_size=0.5)
        all_ones_group = VGroup(all_ones_label, all_ones_boxes).arrange(RIGHT, buff=0.3)
        all_ones_group.shift(UP * 1.2)

        self.play(Write(all_ones_label))
        self.play(Create(all_ones_boxes, lag_ratio=0.1, run_time=1.5))
        self.wait(1)

        # Show x in binary
        minus_sign = MathTex(r"-")
        minus_sign.next_to(all_ones_boxes, DOWN, buff=MED_SMALL_BUFF)

        x_label = MathTex(rf"x = {x_demo}_{{10}} =")
        x_boxes = bit_boxes(x_bits, bit_size=0.5)
        x_group = VGroup(x_label, x_boxes).arrange(RIGHT, buff=0.3)
        x_group.shift(UP * 0.1)


        x_group.shift([(all_ones_boxes.get_center() - x_boxes.get_center())[0], 0 , 0])


        self.play(Write(minus_sign))
        self.play(Write(x_label))
        self.play(Create(x_boxes, lag_ratio=0.1, run_time=1.5))
        self.wait(1)

        # Show equals line
        equals_line = Line(
            x_boxes.get_left() + LEFT * 0.2 + DOWN * 0.5,
            x_boxes.get_right() + RIGHT * 0.2 + DOWN * 0.5,
            color=WHITE
        )
        self.play(Create(equals_line))

        # Perform subtraction bit by bit with animation
        result_bits = ''.join('1' if b == '0' else '0' for b in x_bits)  # This is ~x
        result_boxes = bit_boxes(result_bits, bit_size=0.5)
        result_label = MathTex(r"= \sim x =")
        result_group = VGroup(result_label, result_boxes).arrange(RIGHT, buff=0.3)
        result_group.shift(DOWN * 1)

        result_group.shift([(all_ones_boxes.get_center() - result_boxes.get_center())[0],0 , 0])

        # Animate each bit of the result appearing with corresponding bits highlighted
        self.play(Write(result_label))
        for i in range(N_demo):
            # Highlight the bits being operated on
            self.play(
                Indicate(all_ones_boxes[i], color=YELLOW),
                Indicate(x_boxes[i], color=YELLOW),
                run_time=0.3
            )
            # Show the result bit
            self.play(FadeIn(result_boxes[i], shift=UP * 0.2), run_time=0.3)

        self.wait(2)

        # Emphasize that subtraction from all 1s flips the bits
        flip_explanation = Text("Subtracting from all 1's flips each bit!", font_size=24, color=GREEN)
        flip_explanation.shift(DOWN * 2.2)
        self.play(Write(flip_explanation))

        # Visual emphasis: flash between original and inverted
        for _ in range(2):
            self.play(
                *[x_boxes[i].animate.set_color(RED) for i in range(N_demo)],
                *[result_boxes[i].animate.set_color(GREEN) for i in range(N_demo)],
                run_time=0.5
            )
            self.play(
                *[x_boxes[i].animate.set_color(WHITE) for i in range(N_demo)],
                *[result_boxes[i].animate.set_color(WHITE) for i in range(N_demo)],
                run_time=0.5
            )

        self.wait(2)

        # Clear bit demonstration
        self.play(
            *[FadeOut(mob) for mob in [
                bit_demo_title, all_ones_label, all_ones_boxes, minus_sign,
                x_label, x_boxes, equals_line, result_label, result_boxes,
                flip_explanation
            ]],
            run_time=1
        )

        # Back to derivation - final step
        step3 = MathTex(r"(2^N - 1) - x = \text{bitwise NOT}(x) = \sim x")
        step3.scale(1.2)
        step3.shift(UP * 0.5)
        self.play(Write(step3, run_time=1.5))
        self.play(Circumscribe(step3, color=GREEN, run_time=2))
        self.wait(2)

        # Final result - better positioned and animated
        arrow_final = Arrow(step3.get_bottom(), step3.get_bottom() + DOWN * 0.8, buff=0.1, color=GREEN, stroke_width=10)
        self.play(GrowArrow(arrow_final))

        result = MathTex(r"\therefore \quad -x = \sim x + 1", color=GREEN)
        result.scale(1.8)
        result.next_to(arrow_final, DOWN, buff=0.3)
        box = SurroundingRectangle(result, buff=0.3, color=GREEN, stroke_width=4)

        self.play(Write(result, run_time=2))
        self.play(Create(box))
        self.play(Flash(result, color=YELLOW, flash_radius=1.5, line_length=0.3, num_lines=20))
        self.wait(3)

        # Clear for algorithm demo
        self.play(
            *[FadeOut(mob) for mob in [step3, arrow_final, result, box, title1]],
            run_time=1
        )

        # Step 3: Algorithm visualization
        algo_title = Text("Algorithm: Invert bits, then add 1", font_size=36, color=YELLOW)
        algo_title.to_edge(UP)
        self.play(Write(algo_title, run_time=1))
        self.wait(0.5)

        # Example with 8 bits
        N = 8
        x = 54

        # Show original number with animation
        orig_label = MathTex(rf"x = {x}_{{10}}")
        orig_label.shift(UP * 2.2)
        self.play(Write(orig_label, run_time=1))

        # Show bits
        bit_string = bits_to_string(x, N)
        boxes = bit_boxes(bit_string, bit_size=0.6)
        boxes.shift(UP * 1.3)

        orig_bits = MathTex(rf"= {bit_string}_2")
        orig_bits.next_to(boxes, RIGHT, buff=0.5)

        self.play(Create(boxes, lag_ratio=0.1, run_time=1.5))
        self.play(Write(orig_bits))
        self.wait(1)

        # Step 1: Invert bits with cool animation
        step_label1 = Text("Step 1: Invert all bits (~x)", font_size=28, color=YELLOW)
        step_label1.shift(UP * 0.5)
        self.play(Write(step_label1))

        # Create inverted bits
        inverted_string = ''.join('1' if b == '0' else '0' for b in bit_string)
        inverted_boxes = bit_boxes(inverted_string, bit_size=0.6)
        inverted_boxes.shift(DOWN * 0.3)

        inverted_label = MathTex(rf"\sim x = {inverted_string}_2")
        inverted_label.next_to(inverted_boxes, RIGHT, buff=0.5)

        for i, (orig_box, inv_box) in enumerate(zip(boxes, inverted_boxes)):
            self.play(
                Indicate(orig_box[0], color=YELLOW, scale_factor=1.3),
                Flash(inv_box.get_center(), color=BLUE, flash_radius=0.3),
                FadeIn(inv_box),
                run_time=0.25
            )

        self.play(Write(inverted_label))

        self.wait(1.5)
        self.play(FadeOut(step_label1))

        # Step 2: Add 1 with carry animation
        step_label2 = Text("Step 2: Add 1", font_size=28, color=YELLOW)
        step_label2.shift(DOWN * 1.4)
        self.play(Write(step_label2))

        # Calculate result
        inverted_val = int(inverted_string, 2)
        result_val = (inverted_val + 1) & ((1 << N) - 1)
        result_string = bits_to_string(result_val, N)

        result_boxes = bit_boxes(result_string, bit_size=0.6)
        result_boxes.shift(DOWN * 2.1)

        result_label = MathTex(rf"-x &= {result_string}_2\\ &= {(1 << N) - x}_{{10}}", color=GREEN)
        result_label.next_to(result_boxes, RIGHT, buff=0.5)

        # Animate the addition with carry propagation
        carry = inverted_val & 1
        temp_boxes = inverted_boxes.copy()
        self.add(temp_boxes)

        # Show +1 at LSB
        plus_one = MathTex("+1", color=YELLOW).scale(0.7)
        plus_one.next_to(inverted_boxes[-1], DOWN, buff=0.2)
        self.play(Write(plus_one))

        max_bit = 0

        for i in range(N - 1, -1, -1):
            if carry == 0:
                break

            max_bit += 1

            # Highlight current bit with carry
            carry_indicator = Text("↑", color=RED, font_size=36)
            carry_indicator.next_to(temp_boxes[i], DOWN, buff=0.1)
            self.play(FadeIn(carry_indicator, shift=UP * 0.2), run_time=0.3)

            old_bit = int(inverted_string[i])
            new_bit = (old_bit + carry) % 2
            carry = (old_bit + carry) // 2

            # Show bit flip
            self.play(
                temp_boxes[i][1].animate.set_color(RED),
                Flash(temp_boxes[i].get_center(), color=RED),
                run_time=0.3
            )
            self.play(
                Transform(temp_boxes[i][1], result_boxes[i][1].copy()),
                FadeIn(result_boxes[i]),
                run_time=0.3
            )
            self.play(FadeOut(carry_indicator), run_time=0.2)

        # Make remaining bits visible
        for i in range(N - max_bit):
            if result_boxes[i].get_fill_opacity() < 1:
                self.play(FadeIn(result_boxes[i]), run_time=0.1)

        self.play(FadeOut(plus_one), FadeOut(temp_boxes))
        self.play(Write(result_label))
        self.wait(2)

        # Verification with animation
        verify = MathTex(rf"{x} + {(1 << N) - x} = {1 << N} \equiv 0 \pmod{{2^{N}}}",
                         color=BLUE)
        verify.scale(0.9)
        verify.to_edge(DOWN, buff=0.5)
        self.play(Write(verify, run_time=1.5))
        self.play(Circumscribe(verify, color=YELLOW, run_time=1.5))
        self.wait(2)

        # Clear everything for summary
        self.play(
            *[FadeOut(mob) for mob in [
                algo_title, orig_label, boxes, orig_bits, inverted_boxes,
                inverted_label, result_boxes, result_label, step_label2, verify
            ]],
            run_time=1
        )

        # Summary with animated bullets
        summary_title = Text("Summary", font_size=40, color=YELLOW)
        summary_title.to_edge(UP)
        self.play(Write(summary_title, run_time=1))

        summary = BulletedList(
            r"Two's complement: $-x = \sim x + 1$",
            r"Works in modular arithmetic (mod $2^N$)",
            r"Range: $-2^{N-1}$ to $2^{N-1} - 1$",
            "MSB is the sign bit (0=positive, 1=negative)"
        )
        summary.scale(0.85)

        # Animate each bullet point
        for item in summary:
            self.play(Write(item, run_time=1))
            self.wait(0.5)

        self.wait(2)

        # Final message with extra flair
        self.play(FadeOut(summary, shift=DOWN))
        final = Text("Two's Complement: Simple and Elegant!", font_size=40, gradient=(BLUE, GREEN))
        self.play(Write(final, run_time=1.5))
        self.play(
            final.animate.scale(1.2).set_color(YELLOW),
            run_time=0.5
        )
        self.play(
            final.animate.scale(1 / 1.2).set_color(WHITE),
            run_time=0.5
        )
        self.play(Flash(final, color=GREEN, line_length=0.5, num_lines=30, flash_radius=2))
        self.wait(2)
        self.play(FadeOut(final, shift=UP), FadeOut(summary_title, shift=UP))


if __name__ == '__main__':
    from manim import config

    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 30

    scene = TwoComplement()
    scene.render()
