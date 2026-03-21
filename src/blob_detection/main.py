from manim import *
from collections import deque


class Pixel(VGroup):
    def __init__(self, text: int = 0):
        self.s1 = Square()

        self.tracker = ValueTracker(text)

        self.value = Integer(self.tracker.get_value()).scale(1.5)

        self.value.add_updater(lambda v: v.set_value(self.tracker.get_value())).align_to(
            self.s1
        )

        super().__init__(self.s1, self.value)

    def color_pixel(self):
        return self.s1.animate.set_fill(color=ManimColor([self.value.get_value() / 255., 0., 0., 1.0]), opacity=0.5)

    def mask_pixel(self):
        if self.value.get_value() == 1:
            return self.s1.animate.set_fill(color=ManimColor([255., 255., 255., 1.0]), opacity=0.5)
        else:
            return self.s1.animate.set_fill(color=ManimColor([0., 0., 0., 1.0]), opacity=0)

    def set_value(self, value):
        return self.tracker.animate(run_time=0).set_value(value)

    def set_square_color(self, color: ManimColor, run_time=0.1):
        return self.s1.animate(run_time=run_time).set_fill(color=color, opacity=0.5)


class BlobDetection(Scene):
    def construct(self):
        np.random.seed(0)

        binary_array = np.array([
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)

        value_array = binary_array.copy()

        mask = binary_array == 0
        value_array[mask] = np.random.randint(0, 50, int(np.sum(mask)), dtype=binary_array.dtype)
        value_array[~mask] = np.random.randint(100, 255, int(np.sum(~mask)), dtype=binary_array.dtype)

        index = np.arange(binary_array.size)

        H, W = binary_array.shape

        xI = np.arange(binary_array.size) // W
        yI = np.arange(binary_array.size) % W

        pose_to_index = np.arange(binary_array.size).reshape(H, W)

        pixels = [Pixel(value_array[x, y]) for x, y in zip(xI, yI)]

        pixel_grid = VGroup(*pixels)
        pixel_grid.arrange_in_grid(rows=H, cols=W)

        max_allowed_width = config.frame_width * 0.8
        max_allowed_height = config.frame_height * 0.8

        scale_factor = min(
            max_allowed_width / pixel_grid.width,
            max_allowed_height / pixel_grid.height
        )

        pixel_grid.scale(scale_factor)

        extra_margin = 0.05 * config.frame_height
        pixel_grid.move_to(ORIGIN)
        pixel_grid.shift(DOWN * extra_margin)

        self.play(Write(pixel_grid))

        self.wait(1)

        animations = [p.color_pixel() for p in pixels]

        self.play(*animations)

        self.wait(1)

        text = Text("We mask values > 75 to be red")

        text.next_to(pixel_grid, UP)

        self.play(Write(text))

        self.wait(1)

        animations = []

        for i, x, y in zip(index, xI, yI):
            animations.append(Indicate(pixels[i], run_time=0.1))
            animations.append(pixels[i].set_value(binary_array[x, y]))

        self.play(Succession(*animations))

        animations = [p.mask_pixel() for p in pixels]
        self.play(*animations)

        text2 = Text("Perform BFS on each pixel")
        text2.next_to(pixel_grid, UP)

        self.play(TransformMatchingShapes(text, text2))

        dashboard = VGroup()
        dashboard_title = Text("Moments", font_size=32, color=YELLOW)

        m00_tracker = ValueTracker(0.0)
        m10_tracker = ValueTracker(0.0)
        m01_tracker = ValueTracker(0.0)

        m00_row = VGroup(MathTex(r"M_{00} \text{ (Area)}:"), Integer(0)).arrange(RIGHT)
        m10_row = VGroup(MathTex(r"M_{10}:"), Integer(0)).arrange(RIGHT)
        m01_row = VGroup(MathTex(r"M_{01}:"), Integer(0)).arrange(RIGHT)
        xbar_row = VGroup(MathTex(r"\bar{x}:"), DecimalNumber(0, num_decimal_places=2)).arrange(RIGHT)
        ybar_row = VGroup(MathTex(r"\bar{y}:"), DecimalNumber(0, num_decimal_places=2)).arrange(RIGHT)

        dashboard.add(dashboard_title, m00_row, m10_row, m01_row, xbar_row, ybar_row)
        dashboard.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        dashboard.to_edge(LEFT, buff=0.5)

        m00_row[1].add_updater(lambda d: d.set_value(m00_tracker.get_value() if m00_tracker.get_value() >= 1 else 0))
        m10_row[1].add_updater(lambda d: d.set_value(m10_tracker.get_value()))
        m01_row[1].add_updater(lambda d: d.set_value(m01_tracker.get_value()))

        xbar_row[1].add_updater(lambda d: d.set_value(
            m10_tracker.get_value() / m00_tracker.get_value() if m00_tracker.get_value() >= 1 else 0))
        ybar_row[1].add_updater(lambda d: d.set_value(
            m01_tracker.get_value() / m00_tracker.get_value() if m00_tracker.get_value() >= 1 else 0))

        p00 = pixels[0].get_center()  # Origin (0,0)
        p10 = pixels[W].get_center()  # Row vector target (1,0) - one full row down
        p01 = pixels[1].get_center()  # Col vector target (0,1) - one full col right

        vec_row = p10 - p00
        vec_col = p01 - p00



        # animations = []

        colors = [
            ManimColor([0., 0., 255., 1.0]),
            ManimColor([0., 255., 0., 1.0]),
            ManimColor([0., 255., 255., 1.0]),
            ManimColor([255., 255., 0., 1.0]),
            ManimColor([255., 0., 255., 1.0]),
        ]

        group_count = 0

        val = set()

        # Do BFS
        diffX = [-1, 0, 1, 0]
        diffY = [0, -1, 0, 1]

        variables = VGroup()

        text = Text("Group:")

        text.next_to(pixel_grid, RIGHT).align_to(pixel_grid, UP)
        variables.add(text)


        for i, x, y in zip(index, xI, yI):
            current_pose = (x, y)

            if current_pose in val:
                continue

            val.add(current_pose)

            if binary_array[x, y] == 1:
                group_count += 1
                queue = deque([current_pose])

                group_color = colors[group_count % len(colors)]

                num_variables = 0

                variable = Variable(var=num_variables, label=f"{group_count}", var_type=Integer, color=group_color)

                variable.next_to(variables[-1], DOWN, buff=0.5)

                variables.add(variable)
                # animations.append(Write(variable))

                if group_count == 1:
                    self.play(Write(dashboard))
                    self.play(Write(text))

                self.play(Write(variable))

                m00_tracker.set_value(0.0)
                m10_tracker.set_value(0.0)
                m01_tracker.set_value(0.0)

                first_i = int(pose_to_index[current_pose])
                initial_px_center = pixels[first_i].get_center()

                radius = pixels[first_i].width / 2.0

                centroid_dot = Dot(color=RED, radius=radius * 0.1, fill_opacity=0.75)
                centroid_ring = Circle(radius=radius * 0.75, color=RED, stroke_width=4, stroke_opacity=0.75)
                centroid_group = VGroup(centroid_dot, centroid_ring)
                centroid_group.set_z_index(10)

                centroid_group.move_to(initial_px_center)

                def update_centroid_position(mob):
                    m00 = m00_tracker.get_value()
                    if m00 >= 1.0:
                        x_bar = m01_tracker.get_value() / m00
                        y_bar = m10_tracker.get_value() / m00

                        physical_pos = p00 + (x_bar * vec_row) + (y_bar * vec_col)
                        mob.move_to(physical_pos)

                centroid_group.add_updater(update_centroid_position)
                self.play(FadeIn(centroid_group), run_time=0.3)

                while len(queue) != 0:
                    num_variables += 1
                    next_pose = queue.popleft()

                    next_i = int(pose_to_index[next_pose])

                    # animations.append(Indicate(pixels[next_i], run_time=0.1))
                    # animations.append(AnimationGroup(pixels[next_i].set_value(group_count),
                    #                                  pixels[next_i].set_square_color(
                    #                                      colors[group_count % len(colors)]
                    #                                  )))
                    #
                    # animations.append(variable.tracker.animate(run_time=0).set_value(num_variables))

                    self.play(Indicate(pixels[next_i], run_time=0.1))
                    self.play(AnimationGroup(pixels[next_i].set_value(group_count),
                                             pixels[next_i].set_square_color(group_color)))

                    # Moment computations
                    px_center = next_pose

                    current_m00 = m00_tracker.get_value()
                    new_m00 = current_m00 + 1

                    new_m10 = m10_tracker.get_value() + px_center[1]
                    new_m01 = m01_tracker.get_value() + px_center[0]

                    self.play(
                        AnimationGroup(
                            variable.tracker.animate.set_value(num_variables),
                            m00_tracker.animate.set_value(new_m00),
                            m10_tracker.animate.set_value(new_m10),
                            m01_tracker.animate.set_value(new_m01)
                            , run_time=0.2)
                    )
                    # Moments finished

                    x, y = next_pose

                    for dx, dy in zip(diffX, diffY):
                        newX = x + dx
                        newY = y + dy

                        pose = (newX, newY)

                        if 0 <= newX < binary_array.shape[0] and 0 <= newY < binary_array.shape[1]:
                            if binary_array[newX, newY] == 1:
                                if pose not in val:
                                    queue.append(pose)
                                    val.add(pose)

                centroid_group.remove_updater(update_centroid_position)
            else:
                self.play(Indicate(pixels[i], run_time=0.1))

        # self.play(Succession(*animations))

        self.wait(2)


# class Test(Scene):
#     def construct(self):
#         variable = Variable(var=1, label=f"1", var_type=Integer)
#
#         self.play(Write(variable))
#
#         animations = []
#
#         for i in range(10):
#             animations.append(variable.tracker.animate(run_time=0.1).set_value(i))
#             animations.append(Wait(1))
#
#         self.play(Succession(*animations))
#
#         self.wait(1)

if __name__ == '__main__':
    from manim import config

    config.pixel_height = 1920
    config.pixel_width = 1920
    config.frame_rate = 30

    scene = BlobDetection()
    scene.render()
