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

        text = Text("We mask value > 75 to be red")

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

        animations = []

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

        self.play(Write(text))

        for i, x, y in zip(index, xI, yI):
            current_pose = (x, y)

            if current_pose in val:
                continue

            val.add(current_pose)

            if binary_array[x, y] == 1:
                group_count += 1
                queue = deque([current_pose])

                num_variables = 0

                variable = Variable(var=num_variables, label=f"{group_count}", var_type=Integer)

                variable.next_to(variables[-1], DOWN, buff=0.5)

                variables.add(variable)
                # animations.append(Write(variable))
                self.play(Write(variable))

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
                                                     pixels[next_i].set_square_color(
                                                         colors[group_count % len(colors)]
                                                     )))

                    self.play(variable.tracker.animate(run_time=0.1).set_value(num_variables))

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
            else:
                self.play(Indicate(pixels[i], run_time=0.1))

        # self.play(Succession(*animations))

        self.wait(1)


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
