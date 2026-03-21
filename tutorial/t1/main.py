from manim import *
from manim.utils.unit import Percent, Pixels


# Code from the tutorial series https://www.youtube.com/watch?v=rUsUrbWb2D4&list=PLsMrDyoG1sZm6-jIUQCgN3BVyEVOZz3LQ

class FirstExample(Scene):
    def construct(self):
        # Always puts the Mobject at the center of the screen first
        blue_circle = Circle(color=BLUE, fill_opacity=0.5)
        green_square = Square(color=GREEN, fill_opacity=0.8)

        # Reposition the green square
        green_square.next_to(blue_circle, RIGHT)

        # Adding them to the scene
        self.add(green_square, blue_circle)

        self.play(FadeOut(green_square))


class SecondExample(Scene):
    def construct(self) -> None:
        ax = Axes(x_range=(-3, 3), y_range=(-3, 3))

        curve = ax.plot(lambda x: (x + 2) * x * (x - 2) / 2, color=RED)
        area = ax.get_area(curve, x_range=(-2, 0))

        self.play(Create(ax, run_time=2), Create(curve, run_time=5))
        self.play(FadeIn(area))

        self.wait(2)


class SquareToCircle(Scene):
    def construct(self) -> None:
        green_square = Square(color=GREEN, fill_opacity=0.5)

        self.play(DrawBorderThenFill(green_square))

        blue_circle = Circle(color=BLUE, fill_opacity=0.5)

        self.play(ReplacementTransform(green_square, blue_circle))
        self.play(Indicate(blue_circle))
        self.play(FadeOut(blue_circle))

        self.wait(2)


class Positioning(Scene):
    def construct(self) -> None:
        plane = NumberPlane()

        # Each grid is one Munit length
        self.add(plane)

        # next_to from ep 1
        red_dot = Dot(color=RED)
        green_dor = Dot(color=GREEN)

        green_dor.next_to(red_dot, RIGHT + UP)

        self.add(red_dot, green_dor)

        # shift
        s = Square(color=ORANGE)

        s.shift(2 * UP + 4 * RIGHT)
        self.add(s)

        # move_to
        c = Circle(color=PURPLE)
        c.move_to([-3, -2, 0])

        self.add(c)

        # align_to
        c2 = Circle(radius=0.5, color=RED, fill_opacity=0.5)
        c3 = c2.copy().set_color(YELLOW)
        c4 = c2.copy().set_color(ORANGE)

        # Aligns the upper border of c2 to the upper bounder of s
        c2.align_to(s, UP)

        c3.align_to(s, RIGHT)
        c4.align_to(s, DOWN + RIGHT)

        self.add(c2, c3, c4)


class CriticalPoints(Scene):
    def construct(self) -> None:
        c = Circle(color=GREEN, fill_opacity=0.5)
        self.add(c)

        for d in [ORIGIN, UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL]:
            self.add(Cross(scale_factor=0.2).move_to(c.get_critical_point(d)))

        s = Square(color=RED, fill_opacity=0.5)
        # Aligns the left critical point to the specific point ( very useful )
        s.move_to([1, 0, 0], aligned_edge=LEFT)
        self.add(s)


class UsefulUnits(Scene):
    def construct(self) -> None:
        for perc in range(5, 51, 5):
            self.add(Circle(radius=perc * Percent(X_AXIS)))
            self.add(Square(side_length=2 * perc * Percent(Y_AXIS), color=YELLOW))

        d = Dot()
        d.shift(100 * Pixels * RIGHT)
        self.add(d)


class Grouping(Scene):
    def construct(self) -> None:
        red_dot = Dot(color=RED)
        green_dot = Dot(color=GREEN).next_to(red_dot, RIGHT)
        blue_dot = Dot(color=BLUE).next_to(red_dot, UP)

        # Vectorized vs non vectorized
        # a non-vectorized is an image for example

        dot_group = VGroup(red_dot, green_dot, blue_dot)

        dot_group.to_edge(RIGHT)

        self.add(dot_group)

        circles = VGroup([Circle(radius=0.2) for _ in range(10)])
        circles.arrange(UP, buff=0.5)
        self.add(circles)

        stars = VGroup([Star(color=YELLOW, fill_opacity=1).scale(0.5) for _ in range(20)])
        stars.arrange_in_grid(rows=4, cols=5, buff=0.2)
        self.add(stars)


class BasicAnimations(Scene):
    def construct(self) -> None:
        polys = VGroup(
            [RegularPolygon(5, radius=1, fill_opacity=0.5, color=ManimColor.from_hsl([j / 5., 1.0, 0.5])) for j in
             range(5)],
        )
        polys.arrange(RIGHT)

        self.play(DrawBorderThenFill(polys), run_time=2)

        self.play(
            Rotate(polys[0], PI, rate_func=lambda t: t),  # rate_func=linear
            Rotate(polys[1], PI, rate_func=smooth),
            Rotate(polys[2], PI, rate_func=lambda t: np.sin(t * PI)),
            Rotate(polys[3], PI, rate_func=there_and_back),
            Rotate(polys[4], PI, rate_func=lambda t: 1 - abs(1 - 2 * t)),
            run_time=5
        )


class ConflictingAnimations(Scene):
    def construct(self) -> None:
        s = Square()
        self.add(s)
        self.play(Rotate(s, PI), Rotate(s, -PI), run_time=2)


class LaggingGroup(Scene):
    def construct(self) -> None:
        num = 20

        squares = VGroup(
            [Square(color=ManimColor.from_hsl([j / num, 1.0, 0.5]), fill_opacity=0.5) for j in
             range(num)],
        ).arrange_in_grid(rows=4, cols=5).scale(0.75)

        self.play(AnimationGroup([FadeIn(s) for s in squares], lag_ratio=1))


class AnimateSyntax(Scene):
    def construct(self) -> None:
        s = Square(color=GREEN, fill_opacity=0.5)
        c = Circle(color=RED, fill_opacity=0.5)

        self.add(s, c)

        self.play(s.animate.shift(UP), c.animate.shift(DOWN))

        self.play(VGroup(s, c).animate.arrange(RIGHT, buff=1))

        self.play(c.animate(rate_func=linear).shift(RIGHT).scale(2))


class AnimateProblem(Scene):
    def construct(self) -> None:
        left_square = Square()
        right_square = Square()

        VGroup(left_square, right_square).arrange(RIGHT, buff=1)

        self.add(left_square, right_square)
        self.play(left_square.animate.rotate(PI), Rotate(right_square, PI), run_time=2)
        self.wait()


class AnimationMechanism(Scene):
    def construct(self) -> None:
        c = Circle()

        c.generate_target()
        c.target.set_fill(color=GREEN, opacity=0.5)
        c.target.shift(2 * RIGHT + UP).scale(0.5)

        self.add(c)
        self.wait()
        self.play(MoveToTarget(c))

        s = Square()
        s.save_state()
        self.play(FadeIn(s))
        self.play(s.animate.set_color(PURPLE).set_opacity(0.5).shift(2 * LEFT).scale(3))
        self.play(s.animate.shift(5 * DOWN).rotate(PI / 4))
        self.wait()

        self.play(Restore(s), run_time=2)
        self.wait()


class SimpleCustomAnimation(Scene):
    def construct(self) -> None:
        def spiral_out(mobject: Mobject, t):
            radius = 4 * t
            angle = 2 * t * 2 * PI
            mobject.move_to(radius * (np.cos(angle) * RIGHT + np.sin(angle) * UP))
            mobject.set_color(ManimColor.from_hsl([t, 1.0, 0.5]))
            mobject.set_opacity(1 - t)

        d = Dot(color=WHITE)
        self.add(d)
        self.play(UpdateFromAlphaFunc(d, spiral_out, run_time=5, remover=True))


if __name__ == '__main__':
    pass
