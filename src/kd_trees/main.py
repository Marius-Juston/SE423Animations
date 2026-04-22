# This KD tree code was ported over from https://github.com/BaqablH/KDTreeAnimation/blob/master/kdtree.py, ALL credits go to BaqablH, the only thing that I did was convert it to modern Manim community vizualization standard

from manim import *
import random
import math
import copy
import numpy as np


class Range:
    def __init__(self, MIN, MAX):
        self.MIN = MIN
        self.MAX = MAX

    def rand(self):
        return random.uniform(self.MIN, self.MAX)

    def get_color_256(self, x):
        # Constrain between 0 and 255
        return int(min(255, max(0, 256 * (x - self.MIN) / (self.MAX - self.MIN))))

    def contains(self, x):
        return self.MIN <= x <= self.MAX

    def midpoint(self):
        return (self.MAX + self.MIN) / 2

    def cut(self, get_right_subrange):
        if get_right_subrange:
            self.MIN = self.midpoint()
        else:
            self.MAX = self.midpoint()
        return get_right_subrange

    def cut_from_point(self, x):
        return self.cut(x > self.midpoint())

    def len(self):
        return self.MAX - self.MIN

    def get_closest(self, x):
        return self.MIN if x < self.midpoint() else self.MAX

    def copy(self):
        return copy.copy(self)


class Rektangle(Rectangle):
    def __init__(self, RX, RY, **kwargs):
        super().__init__(width=RX.len(), height=RY.len(), **kwargs)
        self.move_to([RX.midpoint(), RY.midpoint(), 0])

    def get_lower_left(self):
        return self.get_corner(DL)

    def get_upper_right(self):
        return self.get_corner(UR)

    def get_coordinate_range(self, p):
        return Range(self.get_lower_left()[p], self.get_upper_right()[p])

    def RX(self):
        return self.get_coordinate_range(0)

    def RY(self):
        return self.get_coordinate_range(1)

    def rand(self):
        return [self.RX().rand(), self.RY().rand(), 0]

    def get_N_random_points(self, N):
        return [self.rand() for _ in range(N)]

    def get_color_256(self, p):
        return [0, self.RX().get_color_256(p[0]), self.RY().get_color_256(p[1])]

    def midpoint(self):
        return [self.RX().midpoint(), self.RY().midpoint(), 0]

    def contains(self, x, y):
        return self.RX().contains(x) and self.RY().contains(y)

    def copy(self):
        return copy.copy(self)

    def get_closest_point(self, p):
        assert not (self.RX().contains(p[0]) and self.RY().contains(p[1]))
        x = p[0] if self.RX().contains(p[0]) else self.RX().get_closest(p[0])
        y = p[1] if self.RY().contains(p[1]) else self.RY().get_closest(p[1])
        return [x, y, 0]

    def get_subrektangle(self, take_right, dir_index):
        rx = self.RX().copy()
        ry = self.RY().copy()
        rx.cut(take_right) if dir_index == 0 else ry.cut(take_right)
        return Rektangle(rx, ry)

    def get_subrektangle_including_point(self, point, dir_index):
        take_right = self.RX().copy().cut_from_point(
            point[dir_index]) if dir_index == 0 else self.RY().copy().cut_from_point(point[dir_index])
        return [self.get_subrektangle(take_right, dir_index), take_right]


class NodeObjs:
    SCENE = None

    def __init__(self, node_orig):
        self.orig = node_orig
        self.dot = None
        self.best = None
        self.line = None
        self.node = None
        self.node_text = None

    def get_parent(self):
        return self.orig.parent.get_objs()

    def get_orig(self):
        return self.orig

    def copy(self):
        return copy.copy(self)

    def get_objs(self):
        return self


def get_scene():
    return NodeObjs.SCENE


def make_new_rektangle(rektangle):
    return Rektangle(rektangle.RX().copy(), rektangle.RY().copy())


def rgb_to_hex(col_rgb):
    hex_str = "#"
    for i in range(3):
        val = int(col_rgb[i])
        hex_str += f"{val:02X}"
    return hex_str


def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


class Node(NodeObjs):
    N_DIMENSIONS = 2

    def __init__(self, parent, rekt, tree_depth=0, moved_to_right=False, is_terminal=True, point=None):
        super().__init__(self)
        self.parent = parent
        self.rekt = rekt
        self.tree_depth = tree_depth
        self.moved_to_right = moved_to_right
        self.is_terminal = is_terminal
        self.point = point
        self.left = None
        self.right = None
        if is_terminal and point is not None:
            get_scene().animate_new_point(self.get_objs(), self)

    def get_direction(self):
        return self.tree_depth % self.N_DIMENSIONS

    def add_point(self, point):
        if self.is_terminal:
            if self.point is None:
                self.point = point
                return point
            self.is_terminal = False
            self.add_point(self.point)

        new_rekt, take_right = self.rekt.get_subrektangle_including_point(point, self.get_direction())
        if take_right and self.right is None:
            self.right = Node(self, new_rekt, self.tree_depth + 1, take_right, True, point)
            return self.right
        elif not take_right and self.left is None:
            self.left = Node(self, new_rekt, self.tree_depth + 1, take_right, True, point)
            return self.left
        elif take_right:
            return self.right.add_point(point)
        else:
            return self.left.add_point(point)

    def decide_if_visit_other_rectangle(self, child, p):
        closest = child.rekt.get_closest_point(p)
        lower_bound_dist = dist(closest, p)
        success = lower_bound_dist < KDTree.MAX_DIST
        get_scene().animate_look_for_closest_point(child.get_objs(), closest, p, success, KDTree.MAX_DIST,
                                                   lower_bound_dist, child.moved_to_right)
        return success

    def find_closest(self, p):
        assert not (self.is_terminal and self.point is None)

        get_scene().animate_enter_node(self)

        if self.is_terminal:
            dst = dist(self.point, p)
            get_scene().animate_update_minimum_distance(self.get_objs(), KDTree.MAX_DIST, dst, self.point, p,
                                                        (dst < KDTree.MAX_DIST))
            KDTree.MAX_DIST = min(KDTree.MAX_DIST, dst)

        take_right = self.rekt.get_coordinate_range(self.get_direction()).copy().cut_from_point(p[self.get_direction()])

        print("NODE NOW", take_right, self.tree_depth)

        first_son = self.right if take_right else self.left
        second_son = self.left if take_right else self.right

        if first_son is not None:
            first_son.find_closest(p)
        elif not self.is_terminal:
            get_scene().animate_empty_child(self.get_objs(), self, take_right)

        if second_son is not None:
            if self.decide_if_visit_other_rectangle(second_son, p):
                second_son.find_closest(p)
        elif not self.is_terminal:
            get_scene().animate_empty_child(self.get_objs(), self, not take_right)

        get_scene().animate_exit_node(self.get_objs())


class KDTree(Scene):
    NPOINTS = 25
    MAX_DIST = 1000

    BEST_NODE_REF = None

    kdtree_canvas = None
    yellow_rektangle = None
    min_dist_text = None
    min_dist_obj = None
    kd_tree_title = None
    grid = None
    main_dot = None

    def animate_new_point(self, objs, node):
        objs.dot = Dot(np.array(node.point), color=rgb_to_hex(KDTree.kdtree_canvas.get_color_256(node.point)))
        self.add(objs.dot)
        self.wait(0.1)

    def animate_point_is_no_longer_optimal(self, objs):
        self.remove(objs.best)
        self.add(objs.dot)

    def animate_point_is_optimal(self, objs):
        if KDTree.BEST_NODE_REF is not None:
            self.animate_point_is_no_longer_optimal(KDTree.BEST_NODE_REF)
        objs.best = objs.dot.copy()
        objs.best.set_color("#FDDA25").scale(1.25)
        self.remove(objs.dot)
        self.add(objs.best)

        circ = Circle(radius=0.3).set_color(WHITE).move_to(objs.best.get_center())
        self.play(Create(circ))
        self.play(FadeOut(circ))
        self.wait(0.5)
        KDTree.BEST_NODE_REF = objs

    def animate_make_subrectangle(self, node):
        assert node.rekt is not None
        node.rekt = make_new_rektangle(node.rekt)
        node.rekt.set_z_index(-1)
        self.add(node.rekt)

    def make_node(self, objs, moved_to_right, node, col=None):
        for attr in [objs.node, objs.node_text, objs.line]:
            if attr is not None:
                self.remove(attr)
        objs.node = Circle(radius=0.3)
        y_val = 3.5 - node.tree_depth
        if node.tree_depth == 0:
            objs.node.to_edge(LEFT).set_y(y_val).shift(0.5 * (DR + DOWN)).set_color(BLUE)
            objs.node_text = None
            objs.line = None
        elif moved_to_right:
            objs.node.to_edge(LEFT).set_y(y_val).shift(DR)
            objs.node_text = MathTex("U" if node.get_direction() % 2 == 0 else "R").move_to(objs.node.get_center())
            objs.line = Line(objs.node.get_center() + 0.3 * UP,
                             objs.get_parent().node.get_center() + 0.3 * DOWN).set_color(RED)
        else:
            objs.node.to_edge(LEFT).set_y(y_val).shift(DOWN)
            objs.node_text = MathTex("D" if node.get_direction() % 2 == 0 else "L").move_to(objs.node.get_center())
            objs.line = Line(objs.node.get_center() + 0.3 * UP,
                             objs.get_parent().node.get_center() + 0.3 * DOWN).set_color(RED)

        if node.is_terminal:
            objs.node.set_color(GREEN)
        if col is not None:
            objs.node.set_color(col)

        self.add(objs.node)
        if objs.node_text is not None:
            self.add(objs.node_text)
        if objs.line is not None:
            self.add(objs.line)

        self.wait(0.5)

    def animate_update_minimum_distance(self, objs, best_cur_dist, new_dist, point, p, success):
        success_color = GREEN if success else RED
        segment = Line(point, p).set_color(success_color)
        segment.set_z_index(-2)
        inequality_string = ' < ' if success else ' > '
        cur_dist_tex = KDTree.min_dist_obj.copy()
        new_dist_tex = DecimalNumber(new_dist)
        inequality_text = MathTex(inequality_string)
        VGroup(new_dist_tex, inequality_text, cur_dist_tex).arrange(RIGHT).shift(2 * UL).set_color(success_color)

        self.add(segment)
        self.add(new_dist_tex, inequality_text, cur_dist_tex)
        self.wait(2)

        if success:
            new_obj = DecimalNumber(new_dist)
            new_obj.move_to(KDTree.min_dist_obj.get_center())
            get_scene().animate_point_is_optimal(objs)
            self.play(
                FadeOut(new_dist_tex, shift=KDTree.min_dist_obj.get_center() - new_dist_tex.get_center()),
                Transform(KDTree.min_dist_obj, new_obj),
                FadeOut(segment),
                FadeOut(inequality_text),
                FadeOut(cur_dist_tex)
            )
        else:
            self.play(
                FadeOut(new_dist_tex, shift=DL),
                FadeOut(segment),
                FadeOut(inequality_text),
                FadeOut(cur_dist_tex)
            )
        self.wait(3)

    def animate_look_for_closest_point(self, objs, closest, p, success, best_cur_dist, new_dist, moved_to_right):
        success_color = GREEN if success else RED
        segment = Line(closest, p).set_color(success_color)
        segment.set_z_index(-1)

        square = objs.orig.rekt.copy()
        square.set_fill(color=success_color, opacity=0.5)
        square.set_stroke(color=success_color)

        inequality_string = ' < ' if success else ' > '
        cur_dist_tex = DecimalNumber(best_cur_dist)
        new_dist_tex = DecimalNumber(new_dist)
        inequality_text = MathTex(inequality_string)
        VGroup(new_dist_tex, inequality_text, cur_dist_tex).arrange(RIGHT).shift(2 * UL).set_color(success_color)

        self.make_node(objs, moved_to_right, objs.orig, YELLOW)
        self.add(segment)
        self.add(square)
        self.add(new_dist_tex, inequality_text, cur_dist_tex)
        self.wait(3)

        square.set_opacity(0)
        self.remove(new_dist_tex, inequality_text, cur_dist_tex)
        if not success:
            self.remove(objs.node, objs.node_text, objs.line)
        self.remove(segment)
        self.play(FadeOut(square))

    def animate_update_yellow_rektangle_in(self, objs):
        assert objs.get_orig().rekt is not None
        KDTree.yellow_rektangle.become(make_new_rektangle(objs.get_orig().rekt).set_color(YELLOW))

    def animate_update_yellow_rektangle_out(self, objs):
        if objs.orig.parent is not None:
            assert objs.get_parent().get_orig().rekt is not None
            KDTree.yellow_rektangle.become(make_new_rektangle(objs.get_parent().get_orig().rekt).set_color(YELLOW))

    def animate_empty_child(self, objs, node, take_right):
        fail_rekt = node.rekt.get_subrektangle(take_right, node.get_direction())
        fail_rekt.set_fill(PURPLE, opacity=0.4).set_stroke(PURPLE).set_z_index(-1)

        upper_right = fail_rekt.get_upper_right()
        lower_left = fail_rekt.get_lower_left()
        main_diagonal = Line([lower_left[0], upper_right[1], 0], [upper_right[0], lower_left[1], 0]).set_color(RED)
        secondary_diagonal = Line([lower_left[0], lower_left[1], 0], [upper_right[0], upper_right[1], 0]).set_color(RED)

        fake_node = objs.node.copy().to_edge(LEFT).shift(DOWN)
        if take_right:
            fake_node.shift(RIGHT)
        fake_edge = Line(objs.node.get_center() + 0.3 * DOWN, fake_node.get_center() + 0.3 * UP).set_color(PURPLE)

        self.add(main_diagonal, secondary_diagonal, fail_rekt, fake_edge)
        self.wait(3)
        self.remove(main_diagonal, secondary_diagonal, fail_rekt, fake_edge)
        self.wait(1)

    def animate_enter_node(self, node):
        print("DEPTH", node.tree_depth)
        get_scene().make_node(node.get_objs(), node.moved_to_right, node)
        get_scene().animate_make_subrectangle(node)
        get_scene().animate_update_yellow_rektangle_in(node.get_objs())

    def animate_exit_node(self, objs):
        for attr in [objs.node, objs.node_text, objs.line, objs.orig.rekt]:
            if attr is not None:
                self.remove(attr)
        self.animate_update_yellow_rektangle_out(objs)
        self.wait(0.5)
        print("EXIT:", objs.orig.tree_depth)

    def animate_mark_closest_node(self):
        circ = Circle(radius=0.3).set_color(YELLOW).move_to(KDTree.BEST_NODE_REF.orig.point)
        self.add(circ)

    def animate_end(self):
        dist_to_shift = 3.5 * LEFT - KDTree.min_dist_text.get_center()
        new_min_dist_text = KDTree.min_dist_text.copy().shift(dist_to_shift)
        new_min_dist_obj = KDTree.min_dist_obj.copy().set_color(YELLOW).shift(dist_to_shift)
        self.play(
            Transform(KDTree.min_dist_text, new_min_dist_text),
            Transform(KDTree.min_dist_obj, new_min_dist_obj)
        )
        self.animate_mark_closest_node()
        self.wait(5)

    def animate_make_title(self):
        KDTree.kd_tree_title = MathTex(r"k\text{-d tree}").move_to(3.5 * UP + 3.5 * RIGHT)
        self.play(FadeIn(KDTree.kd_tree_title, shift=2 * DOWN))

    def animate_make_grid(self):
        KDTree.grid = NumberPlane(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1])
        KDTree.grid.move_to([3.5, -0.5, 0])
        self.play(Create(KDTree.grid), run_time=5, lag_ratio=0.2)

    def animate_show_distance_title(self):
        KDTree.min_dist_text = MathTex(r"\min d = ")
        KDTree.min_dist_obj = MathTex(r"\infty")

        VGroup(KDTree.min_dist_text, KDTree.min_dist_obj).arrange(RIGHT).to_corner(UL)
        self.play(
            Write(KDTree.min_dist_text),
            FadeIn(KDTree.min_dist_obj, shift=UP),
        )

    def animate_create_main_dot_and_yellow_rektangle(self, the_point):
        KDTree.main_dot = Dot(the_point).set_color(RED)
        self.play(FadeIn(KDTree.main_dot))
        self.wait(1)

        self.add(KDTree.yellow_rektangle)

    def construct(self):
        # Reset globals to avoid cross-render pollution
        NodeObjs.SCENE = self
        KDTree.MAX_DIST = 1000
        KDTree.BEST_NODE_REF = None
        random.seed(0)
        np.random.seed(0)

        # Scene Mobjects
        KDTree.kdtree_canvas = Rektangle(Range(0, 7), Range(-4, 3))
        KDTree.yellow_rektangle = Rektangle(Range(0, 7), Range(-4, 3)).set_color(YELLOW)

        self.animate_make_title()
        self.animate_make_grid()

        points = KDTree.kdtree_canvas.get_N_random_points(self.NPOINTS)
        root = Node(None, KDTree.kdtree_canvas.copy())

        for point in points:
            root.add_point(point)

        the_point = KDTree.kdtree_canvas.rand()

        self.animate_show_distance_title()
        self.animate_create_main_dot_and_yellow_rektangle(the_point)

        root.find_closest(the_point)

        self.animate_end()
