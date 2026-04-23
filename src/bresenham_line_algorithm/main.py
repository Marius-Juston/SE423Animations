from manim import *


class BresenhamLabeledLineAlgorithm(Scene):
    def construct(self):
        # Create the grid and shift it so that integer coordinates are cell centers
        ax = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],  # Adjusted y_range for visibility
            background_line_style={"stroke_opacity": 0.4}
        )
        ax.shift((DOWN + LEFT) * 0.5)

        self.play(Create(ax))

        # Start and end coordinates
        start = np.array([-1, -3, 0])
        end = np.array([1, 2, 0])

        # Draw the true mathematical line and points
        true_line_group = VGroup()
        true_line_group += Line(start, end, color=YELLOW)
        true_line_group += Dot(start, color=RED)
        true_line_group += Dot(end, color=RED)
        self.play(Create(true_line_group))

        # Add "Start" and "End" labels for the true line points
        start_label = Text("Start", color=YELLOW).next_to(start, direction=LEFT, buff=0.5)
        end_label = Text("End", color=YELLOW).next_to(end, direction=LEFT, buff=0.5)
        self.play(Write(start_label), Write(end_label))

        # Get the pixel coordinates for the algorithm
        x0, y0 = int(start[0]), int(start[1])  # Fixed y0 typo from original script
        x1, y1 = int(end[0]), int(end[1])

        # Initialize the generalized Bresenham algorithm variables
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        # Group for our labeled pixels
        labeled_pixels = VGroup()

        while True:
            # We are at the end cell
            if x0 == x1 and y0 == y1:
                # Use special coloring and the specified label: +0.9
                pixel = Square(side_length=1, fill_opacity=0.3, fill_color=GREEN, stroke_width=2)
                value_text = Text("+0.9", color=GREEN_A).scale(0.5)
            else:
                # Non-end cell: blue background, red text: -0.4
                pixel = Square(side_length=1, fill_opacity=0.3, fill_color=BLUE, stroke_width=2)
                value_text = Text("-0.4", color=RED).scale(0.5)

            # Position both the pixel and the text within the cell
            pixel.move_to(np.array([x0, y0, 0]))
            value_text.move_to(np.array([x0, y0, 0]))

            # Animate the creation of both together
            self.play(Create(pixel), Write(value_text), run_time=0.5)
            labeled_pixels.add(VGroup(pixel, value_text))

            # Break the loop when the end coordinate is reached
            if x0 == x1 and y0 == y1:
                break

            # Calculate the next pixel position
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        self.wait(2.0)