"""
Pathfinding Algorithms — A Visual Journey
==========================================
Based on Red Blob Games' "Introduction to A*" by Amit Patel.

Scenes (in narrative order):
    1.  TitleScene              — Intro + algorithm family overview
    2.  WhatIsAGraphScene       — Vertices, edges, what the algorithm sees
    3.  MapRepresentationsScene — Grid vs NavMesh vs Visibility graph
    4.  FrontierConceptScene    — The expanding ring idea (core concept)
    5.  BFSScene                — BFS with code stepping + came_from arrows
    6.  PathReconstructionScene — Following arrows backward
    7.  EarlyExitScene          — Why stop early, side-by-side
    8.  MovementCostsScene      — Steps != distance, why BFS isn't enough
    9.  DijkstraScene           — Priority queue + weighted grid demo
   10.  HeuristicScene          — What a heuristic is, Manhattan distance
   11.  GreedyBFSScene          — Greedy: fast but can be wrong
   12.  AStarScene              — A* = cost + heuristic, full demo
   13.  TripleComparisonScene   — Dijkstra vs Greedy vs A* on same map
   14.  SummaryScene            — Decision flowchart + recap

Render:
    manim -pqh pathfinding_manim.py SceneName
    manim -qh pathfinding_manim.py          # all scenes
"""

from manim import *
import heapq
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════
BG = "#0f0f1a"
PANEL = "#1a1a2e"
PANEL_LIGHT = "#252545"
GRID_STROKE = "#3a3a5c"

C_START = "#22c55e"
C_GOAL = "#ef4444"
C_FRONTIER = "#facc15"
C_VISITED = "#3b82f6"
C_CURRENT = "#f97316"
C_PATH = "#10b981"
C_WALL = "#374151"
C_EDGE = "#6366f1"
C_VERTEX = "#8b5cf6"
C_ACCENT = "#a78bfa"
C_HEURISTIC = "#ec4899"
C_COST = "#06b6d4"
C_CODE_BG = "#16162a"
C_CODE_HL = "#2d2b55"
C_TEXT = "#e2e8f0"
C_FOREST = "#1a3a1a"

CODEFONT = "Monospace"

config.background_color = BG


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID WORLD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
class GridWorld:
    """Simple unweighted grid."""

    def __init__(self, cols, rows, walls=None):
        self.cols, self.rows = cols, rows
        self.walls = set(walls or [])

    def in_bounds(self, p):
        return 0 <= p[0] < self.cols and 0 <= p[1] < self.rows

    def passable(self, p):
        return p not in self.walls

    def neighbors(self, p):
        c, r = p
        out = [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        return [q for q in out if self.in_bounds(q) and self.passable(q)]

    def cost(self, a, b):
        return 1


class WeightedGrid(GridWorld):
    """Grid with per-cell movement costs."""

    def __init__(self, cols, rows, walls=None, weights=None):
        super().__init__(cols, rows, walls)
        self.weights = weights or {}

    def cost(self, a, b):
        return self.weights.get(b, 1)


def heuristic_fn(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_grid(cols, rows, cell_size=0.46, origin=ORIGIN):
    """Return dict (col,row) -> Square, positioned so (0,0) is top-left."""
    cells = {}
    tw = cols * cell_size
    th = rows * cell_size
    for c in range(cols):
        for r in range(rows):
            sq = Square(
                side_length=cell_size,
                fill_color=PANEL_LIGHT, fill_opacity=0.7,
                stroke_color=GRID_STROKE, stroke_width=1,
            )
            x = origin[0] - tw / 2 + cell_size / 2 + c * cell_size
            y = origin[1] + th / 2 - cell_size / 2 - r * cell_size
            sq.move_to([x, y, 0])
            cells[(c, r)] = sq
    return cells


def paint_walls(cells, walls):
    for w in walls:
        if w in cells:
            cells[w].set_fill(C_WALL, opacity=0.92)


class CodePanel:
    """
    Right-side pseudocode panel with line highlighting.
    BUG FIX: empty lines are replaced with a space character
    to avoid zero-point mobjects that crash align_to.
    """

    def __init__(self, lines, font_size=14, panel_width=4.8, line_spacing=0.26):
        self.lines = lines
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.panel_width = panel_width

        self.text_objs = []
        for raw in lines:
            raw: str
            # ── FIX: never pass empty string to Text() ──
            display = raw if raw.strip() else "a"  # Cannot have empty text rendered, but opacity 0 substitute

            left_side = raw.lstrip()

            if len(left_side) == len(raw):
                t = Text(display, font=CODEFONT, font_size=font_size, color=C_TEXT)
            else:
                val = 'a' * (len(raw) - len(left_side))
                t_space = Text(val, font=CODEFONT, font_size=font_size, color=C_TEXT, opacity=0)
                t_space.set_opacity(0)

                t = Text(display.strip(), font=CODEFONT, font_size=font_size, color=C_TEXT)
                t = VGroup(t_space, t ).arrange(RIGHT, buff=0)

            if not raw.strip():
                t.set_opacity(0)  # invisible spacer line
            t.set(max_width=panel_width - 0.5)
            self.text_objs.append(t)

        total_h = len(lines) * line_spacing
        left_x = -(panel_width / 2 - 0.25)
        for i, t in enumerate(self.text_objs):
            t.move_to([0, total_h / 2 - i * line_spacing, 0])
            t.align_to(np.array([left_x, 0, 0]), LEFT)

        self.code_group = VGroup(*self.text_objs)

        self.bg = RoundedRectangle(
            corner_radius=0.15,
            width=panel_width,
            height=self.code_group.height + 0.55,
            fill_color=C_CODE_BG, fill_opacity=0.95,
            stroke_color=C_ACCENT, stroke_width=1.2,
        )
        self.bg.move_to(self.code_group.get_center())

        self.hl = Rectangle(
            width=panel_width - 0.2,
            height=line_spacing + 0.04,
            fill_color=C_CODE_HL, fill_opacity=0,
            stroke_width=0,
        )
        self.hl.move_to(self.bg.get_center())

        self.group = VGroup(self.bg, self.hl, self.code_group)

    def pos(self, target):
        self.group.move_to(target)
        return self

    def get_group(self):
        return self.group

    def hl_line(self, idx):
        t = self.text_objs[idx]
        vect = t.get_center()
        vect[0] = self.hl.get_center()[0]

        return [self.hl.animate.move_to(vect).set_fill(opacity=0.9)]

    def hl_off(self):
        return [self.hl.animate.set_fill(opacity=0)]


def section_card(scene, num, title, sub="", hold=2.0):
    n = Text(
        "0" + str(num) if num < 10 else str(num),
        font_size=72, color=C_ACCENT, weight=BOLD,
    )
    t = Text(title, font_size=40, color=WHITE, weight=BOLD)
    g = VGroup(n, t).arrange(DOWN, buff=0.35)
    if sub:
        s = Text(sub, font_size=20, color=GRAY_B)
        s.next_to(t, DOWN, buff=0.25)
        g.add(s)
    g.move_to(ORIGIN)
    scene.play(FadeIn(g, shift=UP * 0.3), run_time=0.7)
    scene.wait(hold)
    scene.play(FadeOut(g, shift=UP * 0.5), run_time=0.5)
    scene.wait(0.2)


def info_box(text, color=C_ACCENT, font_size=17, width=None):
    """Small rounded box with centered text, auto-sized."""
    txt = Text(text, font_size=font_size, color=C_TEXT, line_spacing=1.3)
    w = width or (txt.width + 0.6)
    box = RoundedRectangle(
        corner_radius=0.12,
        width=w,
        height=txt.height + 0.35,
        fill_color=PANEL, fill_opacity=0.95,
        stroke_color=color, stroke_width=1.5,
    )
    txt.move_to(box)
    return VGroup(box, txt)


def fade_all(scene, rt=0.7):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=rt)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════════════
class TitleScene(Scene):
    def construct(self):
        # Faint background grid
        bg = VGroup(*build_grid(18, 10, 0.38).values()).set_opacity(0.08)
        self.add(bg)

        title = Text("Pathfinding Algorithms", font_size=52,
                     color=WHITE, weight=BOLD)
        sub = Text("A Visual Journey from Graphs to A*",
                   font_size=24, color=C_ACCENT)
        stack = VGroup(title, sub).arrange(DOWN, buff=0.35).move_to(UP * 0.6)

        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.9)
        self.play(FadeIn(sub, shift=DOWN * 0.2), run_time=0.6)
        self.wait(0.5)

        # Algorithm family overview — three colored cards
        algos = [
            ("BFS", C_VISITED, "Explores equally\nin all directions"),
            ("Dijkstra", C_COST, "Accounts for\nmovement costs"),
            ("A*", C_ACCENT, "Explores toward\nthe goal"),
        ]
        cards = VGroup()
        for name, col, desc in algos:
            n = Text(name, font_size=22, color=col, weight=BOLD)
            d = Text(desc, font_size=13, color=GRAY_B, line_spacing=1.2)
            icon = Square(
                side_length=0.35, fill_color=col, fill_opacity=0.25,
                stroke_color=col, stroke_width=2,
            )
            inner = VGroup(n, d).arrange(DOWN, buff=0.12)
            card_bg = RoundedRectangle(
                corner_radius=0.15,
                width=inner.width + icon.width + 0.9,
                height=max(inner.height, icon.height) + 0.35,
                fill_color=PANEL, fill_opacity=0.8,
                stroke_color=col, stroke_width=1,
            )
            icon.move_to(card_bg.get_left() + RIGHT * 0.4)
            inner.next_to(icon, RIGHT, buff=0.2)
            cards.add(VGroup(card_bg, icon, inner))

        cards.arrange(RIGHT, buff=0.3).next_to(stack, DOWN, buff=0.8)
        if cards.width > 13:
            cards.scale_to_fit_width(13)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards],
                        lag_ratio=0.2),
            run_time=1.2,
        )
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 2 — WHAT IS A GRAPH?
# ═══════════════════════════════════════════════════════════════════════════════
class WhatIsAGraphScene(Scene):
    def construct(self):
        section_card(self, 1, "What Is a Graph?",
                     "The data structure behind every pathfinder")

        # Build a small abstract graph
        positions = {
            "A": [-4.0, 1.6, 0], "B": [-2.0, 2.6, 0],
            "C": [0.0, 1.6, 0], "D": [-3.0, 0.0, 0],
            "E": [-1.0, 0.0, 0], "F": [1.0, 0.0, 0],
            "G": [-2.0, -1.6, 0], "H": [0.0, -1.6, 0],
        }
        edge_pairs = [
            ("A", "B"), ("A", "D"), ("B", "C"), ("B", "E"),
            ("C", "F"), ("D", "E"), ("D", "G"), ("E", "F"),
            ("E", "H"), ("F", "H"), ("G", "H"),
        ]

        verts = {}
        vlbls = {}
        for name, pos in positions.items():
            d = Dot(pos, radius=0.2, color=C_VERTEX, z_index=2)
            lb = Text(name, font_size=18, color=WHITE, weight=BOLD, z_index=3)
            lb.move_to(d)
            verts[name] = d
            vlbls[name] = lb

        graph_grp = VGroup(*verts.values(), *vlbls.values())
        graph_grp.shift(LEFT * 1.5)
        for n in positions:
            positions[n] = list(verts[n].get_center())

        # Right-side explanation
        exp_title = Text("A graph has:", font_size=22, color=C_TEXT)
        v_dot = Dot(radius=0.1, color=C_VERTEX)
        v_txt = Text("Vertices (nodes) = locations", font_size=16, color=GRAY_B)
        v_row = VGroup(v_dot, v_txt).arrange(RIGHT, buff=0.15)
        e_line = Line(ORIGIN, RIGHT * 0.5, color=C_EDGE, stroke_width=3)
        e_txt = Text("Edges = connections", font_size=16, color=GRAY_B)
        e_row = VGroup(e_line, e_txt).arrange(RIGHT, buff=0.15)
        exp = VGroup(exp_title, v_row, e_row).arrange(
            DOWN, buff=0.25, aligned_edge=LEFT
        )
        exp.to_edge(RIGHT, buff=0.6).shift(UP * 2.2)

        # Animate vertices
        self.play(FadeIn(exp_title, shift=DOWN * 0.15), run_time=0.5)
        self.play(
            LaggedStart(
                *[AnimationGroup(GrowFromCenter(verts[n]), FadeIn(vlbls[n]))
                  for n in positions],
                lag_ratio=0.08,
            ),
            FadeIn(v_row, shift=LEFT * 0.15),
            run_time=1.5,
        )
        self.wait(0.4)

        # Animate edges
        elines = {}
        for a, b in edge_pairs:
            line = Line(
                verts[a].get_center(), verts[b].get_center(),
                color=C_EDGE, stroke_width=2.5, stroke_opacity=0.7, z_index=1,
            )
            elines[(a, b)] = line

        self.play(
            LaggedStart(*[Create(l) for l in elines.values()], lag_ratio=0.06),
            FadeIn(e_row, shift=LEFT * 0.15),
            run_time=1.5,
        )
        self.wait(1)

        # Key insight: algorithm only sees the graph
        insight = info_box(
            "The algorithm only sees the graph.\n"
            "It knows nothing about rooms, doors,\n"
            "terrain, or how things look!",
            color=C_VERTEX, font_size=15,
        )
        insight.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.8)
        self.play(FadeIn(insight, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)

        # Highlight a path A -> D -> E -> F -> H
        path_seq = ["A", "D", "E", "F", "H"]
        for i in range(len(path_seq) - 1):
            a, b = path_seq[i], path_seq[i + 1]
            key = (a, b) if (a, b) in elines else (b, a)
            self.play(
                elines[key].animate.set_color(C_PATH).set_stroke(width=4, opacity=1),
                verts[path_seq[i]].animate.set_color(C_PATH),
                run_time=0.35,
            )
        self.play(verts[path_seq[-1]].animate.set_color(C_PATH), run_time=0.3)

        out_note = info_box(
            "Output: a sequence of nodes and edges.\n"
            "You decide what 'move along edge' means.",
            color=C_PATH, font_size=14,
        )
        out_note.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(out_note, shift=UP * 0.15), run_time=0.5)
        self.wait(2.5)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 3 — MAP REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
class MapRepresentationsScene(Scene):
    def construct(self):
        section_card(self, 2, "Map Representations",
                     "Turning a game world into a graph")

        # Three columns: Grid | NavMesh | Visibility Graph
        headers = [
            Text("Grid", font_size=22, color=C_ACCENT, weight=BOLD),
            Text("Navigation Mesh", font_size=22, color=C_ACCENT, weight=BOLD),
            Text("Visibility Graph", font_size=22, color=C_ACCENT, weight=BOLD),
        ]
        x_off = [-4.5, 0, 4.5]
        for h, xo in zip(headers, x_off):
            h.move_to([xo, 3.2, 0])

        # ── GRID ──
        cs = 0.3
        gc = build_grid(7, 5, cs, np.array([-4.5, 0.8, 0]))
        gv = VGroup(*gc.values())
        gwalls = [(2, 1), (2, 2), (2, 3), (3, 1)]
        paint_walls(gc, gwalls)

        cc_pos = gc[(3, 2)].get_center()
        arrows_4 = VGroup(*[
            Arrow(
                cc_pos, cc_pos + d * cs * 0.8, buff=0.02,
                color=C_EDGE, stroke_width=2,
                max_tip_length_to_length_ratio=0.3,
            )
            for d in [RIGHT, LEFT, UP, DOWN]
        ])
        grid_note = Text(
            "Each tile = 1 vertex\n4 neighbors = edges\nSimple but many nodes",
            font_size=11, color=GRAY_B, line_spacing=1.3,
        )
        grid_note.move_to([-4.5, -1.5, 0])

        # ── NAV MESH ──
        p1 = Polygon(
            [-1.5, 1.8, 0], [0, 2.0, 0], [0.4, 0.5, 0], [-1.1, 0.3, 0],
            fill_color=C_VISITED, fill_opacity=0.2,
            stroke_color=C_VISITED, stroke_width=1.8,
        )
        p2 = Polygon(
            [0, 2.0, 0], [1.6, 1.7, 0], [1.8, 0, 0], [0.4, 0.5, 0],
            fill_color=C_COST, fill_opacity=0.2,
            stroke_color=C_COST, stroke_width=1.8,
        )
        p3 = Polygon(
            [-1.1, 0.3, 0], [0.4, 0.5, 0], [1.8, 0, 0],
            [1.3, -1.3, 0], [-1.2, -0.8, 0],
            fill_color=C_PATH, fill_opacity=0.2,
            stroke_color=C_PATH, stroke_width=1.8,
        )
        obs = Polygon(
            [0, 0.5, 0], [0.4, 0.5, 0], [0.4, 0, 0], [0, -0.1, 0],
            fill_color=C_WALL, fill_opacity=0.85,
            stroke_color=WHITE, stroke_width=1.5,
        )
        c1, c2, c3 = p1.get_center(), p2.get_center(), p3.get_center()
        nav_dots = VGroup(
            *[Dot(p, radius=0.07, color=C_VERTEX) for p in [c1, c2, c3]]
        )
        nav_edges = VGroup(
            Line(c1, c2, color=C_EDGE, stroke_width=1.5),
            Line(c1, c3, color=C_EDGE, stroke_width=1.5),
            Line(c2, c3, color=C_EDGE, stroke_width=1.5),
        )
        nav_note = Text(
            "Walkable polygons\nFewer vertices = faster\nGood for open areas",
            font_size=11, color=GRAY_B, line_spacing=1.3,
        )
        nav_note.move_to([0, -1.5, 0])

        # ── VISIBILITY GRAPH ──
        obs_corners = [
            np.array([3.5, 1.0, 0]), np.array([4.8, 1.2, 0]),
            np.array([4.8, -0.2, 0]), np.array([3.5, 0.0, 0]),
        ]
        vis_obs = Polygon(
            *obs_corners, fill_color=C_WALL, fill_opacity=0.8,
            stroke_color=WHITE, stroke_width=1.5,
        )
        sp = np.array([2.8, 1.8, 0])
        ep = np.array([5.6, -0.7, 0])
        vis_pts = [sp] + obs_corners + [ep]
        vis_dots = VGroup(
            Dot(sp, radius=0.09, color=C_START),
            Dot(ep, radius=0.09, color=C_GOAL),
            *[Dot(p, radius=0.06, color=C_HEURISTIC) for p in obs_corners],
        )
        vis_lines = VGroup()
        for i in range(len(vis_pts)):
            for j in range(i + 1, len(vis_pts)):
                vis_lines.add(DashedLine(
                    vis_pts[i], vis_pts[j],
                    color=C_EDGE, stroke_width=1, stroke_opacity=0.35,
                    dash_length=0.07,
                ))
        vis_note = Text(
            "Corner-to-corner edges\nOptimal paths possible\nbut O(n^2) edges",
            font_size=11, color=GRAY_B, line_spacing=1.3,
        )
        vis_note.move_to([4.5, -1.5, 0])

        # Dividers
        div1 = Line([-2.2, 3.4, 0], [-2.2, -2.0, 0], color=GRAY_C, stroke_width=0.8)
        div2 = Line([2.2, 3.4, 0], [2.2, -2.0, 0], color=GRAY_C, stroke_width=0.8)

        # Animate
        self.play(
            *[FadeIn(h) for h in headers],
            FadeIn(div1), FadeIn(div2),
            run_time=0.5,
        )
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.85) for c in gv], lag_ratio=0.005),
            run_time=0.8,
        )
        self.play(FadeIn(arrows_4), FadeIn(grid_note), run_time=0.5)

        self.play(
            LaggedStart(
                DrawBorderThenFill(p1), DrawBorderThenFill(p2),
                DrawBorderThenFill(p3), FadeIn(obs), lag_ratio=0.15,
            ),
            run_time=1.0,
        )
        self.play(Create(nav_edges), FadeIn(nav_dots), FadeIn(nav_note), run_time=0.6)

        self.play(FadeIn(vis_obs), FadeIn(vis_dots), run_time=0.5)
        self.play(
            LaggedStart(*[Create(l) for l in vis_lines], lag_ratio=0.03),
            FadeIn(vis_note),
            run_time=1.2,
        )

        ib = info_box(
            "Fewer graph nodes = faster search. We use grids\n"
            "for visualization, but all algorithms work on any graph.",
            color=C_ACCENT, font_size=14,
        )
        ib.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(ib, shift=UP * 0.15), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 4 — THE FRONTIER CONCEPT
# ═══════════════════════════════════════════════════════════════════════════════
class FrontierConceptScene(Scene):
    def construct(self):
        section_card(self, 3, "The Frontier",
                     "An expanding ring of exploration")

        cols, rows = 11, 9
        cs = 0.38
        cells = build_grid(cols, rows, cs, ORIGIN)
        gv = VGroup(*cells.values())
        center = (5, 4)
        center_dot = Dot(
            cells[center].get_center(), radius=0.12, color=C_START, z_index=5,
        )

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.003),
            run_time=0.7,
        )
        self.play(FadeIn(center_dot), run_time=0.3)

        # Pre-compute BFS rings
        world = GridWorld(cols, rows)
        frontier = deque([center])
        reached = {center}
        rings = [[center]]

        while frontier:
            ring = []
            for _ in range(len(frontier)):
                cur = frontier.popleft()
                for n in world.neighbors(cur):
                    if n not in reached:
                        reached.add(n)
                        frontier.append(n)
                        ring.append(n)
            if ring:
                rings.append(ring)

        ring_colors = [
            interpolate_color(
                ManimColor(C_FRONTIER), ManimColor(C_VISITED),
                i / max(len(rings) - 1, 1),
            )
            for i in range(len(rings))
        ]

        ring_label = Text("Ring 0", font_size=20, color=C_FRONTIER, weight=BOLD)
        ring_label.to_edge(UP, buff=0.4)
        self.play(FadeIn(ring_label), run_time=0.3)

        for i, ring in enumerate(rings[1:], 1):
            anims = [cells[p].animate.set_fill(ring_colors[i], 0.75) for p in ring]
            new_lbl = Text(
                "Ring " + str(i), font_size=20,
                color=ring_colors[i], weight=BOLD,
            )
            new_lbl.to_edge(UP, buff=0.4)
            self.play(
                *anims,
                Transform(ring_label, new_lbl),
                run_time=0.25 if i < 4 else 0.15,
            )

        self.wait(0.5)

        exp = info_box(
            "BFS explores in rings: all cells at distance 1,\n"
            "then distance 2, then 3 ... This guarantees the\n"
            "shortest path when all steps cost the same.",
            color=C_FRONTIER, font_size=15,
        )
        exp.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(exp, shift=UP * 0.2), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 5 — BREADTH FIRST SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
class BFSScene(Scene):
    def construct(self):
        section_card(self, 4, "Breadth First Search",
                     "Exploring equally in all directions")

        cols, rows = 8, 6
        cs = 0.44
        grid_ctr = LEFT * 2.6 + DOWN * 0.15
        walls = [(3, 1), (3, 2), (3, 3), (3, 4), (5, 0), (5, 1), (5, 2)]
        world = GridWorld(cols, rows, walls)
        cells = build_grid(cols, rows, cs, grid_ctr)
        gv = VGroup(*cells.values())
        paint_walls(cells, walls)

        start, goal = (1, 2), (6, 3)
        s_dot = Dot(cells[start].get_center(), radius=0.12, color=C_START, z_index=5)
        g_dot = Dot(cells[goal].get_center(), radius=0.12, color=C_GOAL, z_index=5)
        s_lbl = Text("S", font_size=12, color=WHITE, weight=BOLD, z_index=6)
        s_lbl.move_to(s_dot)
        g_lbl = Text("G", font_size=12, color=WHITE, weight=BOLD, z_index=6)
        g_lbl.move_to(g_dot)

        code_lines = [
            "frontier = Queue()",
            "frontier.put( start )",
            "came_from = { start: None }",
            " ",
            "while frontier not empty:",
            "  current = frontier.get()",
            "  for next in neighbors(current):",
            "    if next not in came_from:",
            "      frontier.put( next )",
            "      came_from[next] = current",
        ]
        code = CodePanel(code_lines, font_size=13, panel_width=4.4, line_spacing=0.25)
        code.pos(RIGHT * 4.1 + UP * 1.5)

        leg_data = [
            (C_START, "Start"), (C_GOAL, "Goal"), (C_FRONTIER, "Frontier"),
            (C_VISITED, "Visited"), (C_CURRENT, "Current"),
        ]
        leg = VGroup()
        for col, lbl in leg_data:
            sq = Square(
                side_length=0.18, fill_color=col, fill_opacity=0.85,
                stroke_width=0,
            )
            tx = Text(lbl, font_size=11, color=GRAY_B)
            leg.add(VGroup(sq, tx).arrange(RIGHT, buff=0.1))
        leg.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        leg.next_to(code.get_group(), DOWN, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.004),
            run_time=0.6,
        )
        self.play(
            FadeIn(s_dot), FadeIn(s_lbl), FadeIn(g_dot), FadeIn(g_lbl),
            FadeIn(code.get_group()), FadeIn(leg),
            run_time=0.6,
        )

        # Init highlight
        for i in [0, 1, 2]:
            self.play(*code.hl_line(i), run_time=0.25)
        cells[start].set_fill(C_FRONTIER, 0.85)

        frontier_q = deque([start])
        came_from = {start: None}
        arrow_mobs = VGroup()

        step_count = 0
        while frontier_q and step_count < 45:
            step_count += 1
            self.play(*code.hl_line(4), run_time=0.08)

            current = frontier_q.popleft()
            self.play(*code.hl_line(5), run_time=0.08)

            if current not in (start, goal):
                self.play(
                    cells[current].animate.set_fill(C_CURRENT, 0.85),
                    run_time=0.06,
                )

            self.play(*code.hl_line(6), run_time=0.05)

            new_anims = []
            new_arrows = []
            for nxt in world.neighbors(current):
                if nxt not in came_from:
                    frontier_q.append(nxt)
                    came_from[nxt] = current
                    if nxt not in (start, goal):
                        new_anims.append(
                            cells[nxt].animate.set_fill(C_FRONTIER, 0.85)
                        )
                    ac = cells[current].get_center()
                    bc = cells[nxt].get_center()
                    arr = Arrow(
                        ac, bc, buff=cs * 0.18,
                        color=GRAY_B, stroke_width=1.2,
                        max_tip_length_to_length_ratio=0.2,
                        stroke_opacity=0.5, z_index=3,
                    )
                    new_arrows.append(arr)

            if new_anims:
                self.play(*code.hl_line(8), run_time=0.04)
                self.play(*new_anims, run_time=0.1)
                for a in new_arrows:
                    arrow_mobs.add(a)
                self.play(*[GrowArrow(a) for a in new_arrows], run_time=0.08)

            if current not in (start, goal):
                cells[current].set_fill(C_VISITED, 0.65)

            if current == goal:
                break

        self.play(*code.hl_off(), run_time=0.2)
        note = info_box(
            "came_from[B] = A means 'I reached B from A'.\n"
            "These arrows form breadcrumbs back to the start.",
            color=C_VISITED, font_size=13,
        )
        note.next_to(leg, DOWN, buff=0.2)
        self.play(FadeOut(leg), FadeIn(note, shift=UP * 0.15), run_time=0.5)
        self.wait(2)

        # Reconstruct path
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()

        path_arrows = VGroup()
        for i in range(len(path) - 1):
            ac = cells[path[i]].get_center()
            bc = cells[path[i + 1]].get_center()
            pa = Arrow(
                ac, bc, buff=cs * 0.1, color=C_PATH, stroke_width=3,
                max_tip_length_to_length_ratio=0.2, z_index=4,
            )
            path_arrows.add(pa)

        self.play(arrow_mobs.animate.set_opacity(0.15), run_time=0.3)
        for p in path:
            if p not in (start, goal):
                self.play(cells[p].animate.set_fill(C_PATH, 0.9), run_time=0.1)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in path_arrows], lag_ratio=0.1),
            run_time=0.7,
        )
        self.wait(2)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 6 — PATH RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
class PathReconstructionScene(Scene):
    def construct(self):
        section_card(self, 5, "Path Reconstruction",
                     "Follow the arrows backward from goal to start")

        cols, rows = 6, 4
        cs = 0.55
        walls = [(2, 1), (2, 2)]
        world = GridWorld(cols, rows, walls)
        cells = build_grid(cols, rows, cs, LEFT * 1.5)
        gv = VGroup(*cells.values())
        paint_walls(cells, walls)

        start, goal = (0, 1), (5, 2)
        s_dot = Dot(cells[start].get_center(), radius=0.12, color=C_START, z_index=5)
        g_dot = Dot(cells[goal].get_center(), radius=0.12, color=C_GOAL, z_index=5)

        # Run BFS quietly
        frontier_q = deque([start])
        came_from = {start: None}
        while frontier_q:
            cur = frontier_q.popleft()
            if cur == goal:
                break
            for n in world.neighbors(cur):
                if n not in came_from:
                    frontier_q.append(n)
                    came_from[n] = cur

        # Draw all came_from arrows
        cf_arrows = VGroup()
        for child, parent in came_from.items():
            if parent is not None:
                ac = cells[parent].get_center()
                bc = cells[child].get_center()
                a = Arrow(
                    ac, bc, buff=cs * 0.15,
                    color=GRAY_B, stroke_width=1.5,
                    max_tip_length_to_length_ratio=0.18,
                    stroke_opacity=0.4, z_index=3,
                )
                cf_arrows.add(a)

        recon_code = [
            "current = goal",
            "path = []",
            "while current != start:",
            "  path.append( current )",
            "  current = came_from[current]",
            "path.reverse()",
        ]
        code = CodePanel(recon_code, font_size=14, panel_width=4.5, line_spacing=0.28)
        code.pos(RIGHT * 3.8 + UP * 1.0)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.005),
            FadeIn(s_dot), FadeIn(g_dot),
            run_time=0.6,
        )

        # Color visited
        for pos in came_from:
            if pos not in (start, goal) and pos not in walls:
                cells[pos].set_fill(C_VISITED, 0.5)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in cf_arrows], lag_ratio=0.03),
            run_time=0.8,
        )
        self.play(FadeIn(code.get_group()), run_time=0.5)
        self.wait(0.5)

        # Step through reconstruction
        self.play(*code.hl_line(0), run_time=0.3)
        cur = goal
        self.play(cells[goal].animate.set_fill(C_PATH, 0.9), run_time=0.3)

        self.play(*code.hl_line(2), run_time=0.2)
        while cur != start:
            self.play(*code.hl_line(3), run_time=0.15)
            self.play(*code.hl_line(4), run_time=0.15)
            prev = came_from[cur]

            ac = cells[prev].get_center()
            bc = cells[cur].get_center()
            bright = Arrow(
                ac, bc, buff=cs * 0.1, color=C_PATH, stroke_width=3.5,
                max_tip_length_to_length_ratio=0.2, z_index=5,
            )
            self.play(GrowArrow(bright), run_time=0.2)
            if prev not in (start, goal):
                self.play(
                    cells[prev].animate.set_fill(C_PATH, 0.9), run_time=0.15,
                )

            cur = prev
            self.play(*code.hl_line(2), run_time=0.1)

        self.play(*code.hl_line(5), run_time=0.3)
        self.wait(1)

        done = info_box(
            "The path is built by walking backward through\n"
            "came_from, then reversing the list.",
            color=C_PATH, font_size=15,
        )
        done.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(done, shift=UP * 0.15), run_time=0.5)
        self.wait(2.5)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 7 — EARLY EXIT
# ═══════════════════════════════════════════════════════════════════════════════
class EarlyExitScene(Scene):
    def construct(self):
        section_card(self, 6, "Early Exit",
                     "Why explore everything if we found the goal?")

        cols, rows = 8, 6
        cs = 0.38
        walls = [(3, 1), (3, 2), (3, 3), (3, 4)]
        world = GridWorld(cols, rows, walls)
        start, goal = (1, 2), (6, 3)

        lbl_no = Text("No early exit", font_size=16, color=C_GOAL, weight=BOLD)
        lbl_yes = Text("With early exit", font_size=16, color=C_PATH, weight=BOLD)
        lbl_no.move_to([-3.5, 3.2, 0])
        lbl_yes.move_to([3.5, 3.2, 0])
        divider = Line([0, 3.4, 0], [0, -3.3, 0], color=GRAY_C, stroke_width=0.8)

        cells_no = build_grid(cols, rows, cs, np.array([-3.5, 0.5, 0]))
        cells_yes = build_grid(cols, rows, cs, np.array([3.5, 0.5, 0]))
        gv_no = VGroup(*cells_no.values())
        gv_yes = VGroup(*cells_yes.values())
        paint_walls(cells_no, walls)
        paint_walls(cells_yes, walls)

        sdots = [
            Dot(cells_no[start].get_center(), radius=0.08, color=C_START, z_index=5),
            Dot(cells_yes[start].get_center(), radius=0.08, color=C_START, z_index=5),
        ]
        gdots = [
            Dot(cells_no[goal].get_center(), radius=0.08, color=C_GOAL, z_index=5),
            Dot(cells_yes[goal].get_center(), radius=0.08, color=C_GOAL, z_index=5),
        ]

        self.play(
            FadeIn(lbl_no), FadeIn(lbl_yes), FadeIn(divider),
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv_no], lag_ratio=0.003),
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv_yes], lag_ratio=0.003),
            *[FadeIn(d) for d in sdots + gdots],
            run_time=0.8,
        )

        def bfs_order(w, s, g, early):
            fq = deque([s])
            reached = {s}
            order = []
            while fq:
                cur = fq.popleft()
                order.append(cur)
                if early and cur == g:
                    break
                for n in w.neighbors(cur):
                    if n not in reached:
                        reached.add(n)
                        fq.append(n)
            return order

        ord_no = bfs_order(world, start, goal, False)
        ord_yes = bfs_order(world, start, goal, True)
        mx = max(len(ord_no), len(ord_yes))

        batch = 4
        for i in range(0, mx, batch):
            anims = []
            for j in range(batch):
                idx = i + j
                if idx < len(ord_no):
                    p = ord_no[idx]
                    if p not in (start, goal):
                        anims.append(cells_no[p].animate.set_fill(C_VISITED, 0.65))
                if idx < len(ord_yes):
                    p = ord_yes[idx]
                    if p not in (start, goal):
                        anims.append(cells_yes[p].animate.set_fill(C_VISITED, 0.65))
            if anims:
                self.play(*anims, run_time=0.08)

        remain = []
        for idx in range(len(ord_yes), len(ord_no)):
            p = ord_no[idx]
            if p not in (start, goal):
                remain.append(cells_no[p].animate.set_fill(C_VISITED, 0.65))
        for i in range(0, len(remain), 6):
            self.play(*remain[i: i + 6], run_time=0.06)

        cnt_no = Text(
            str(len(ord_no)) + " cells explored",
            font_size=15, color=C_GOAL,
        )
        cnt_yes = Text(
            str(len(ord_yes)) + " cells explored",
            font_size=15, color=C_PATH,
        )
        cnt_no.next_to(gv_no, DOWN, buff=0.3)
        cnt_yes.next_to(gv_yes, DOWN, buff=0.3)
        self.play(FadeIn(cnt_no), FadeIn(cnt_yes), run_time=0.4)

        added = info_box(
            "Add: if current == goal: break\n"
            "Same shortest path, far less exploration!",
            color=C_PATH, font_size=14,
        )
        added.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(added, shift=UP * 0.15), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 8 — MOVEMENT COSTS (WHY BFS IS NOT ENOUGH)
# ═══════════════════════════════════════════════════════════════════════════════
class MovementCostsScene(Scene):
    def construct(self):
        section_card(self, 7, "Movement Costs",
                     "Why BFS is not always enough")

        cols, rows = 7, 5
        cs = 0.46

        forest = set()
        for c in range(2, 5):
            for r in range(1, 4):
                forest.add((c, r))

        # Left: BFS straight through
        lbl_bfs = Text("BFS: counts steps", font_size=17, color=C_VISITED, weight=BOLD)
        lbl_bfs.move_to([-3.5, 3.2, 0])
        cells_bfs = build_grid(cols, rows, cs, np.array([-3.5, 0.3, 0]))
        gv_bfs = VGroup(*cells_bfs.values())
        for f in forest:
            if f in cells_bfs:
                cells_bfs[f].set_fill(C_FOREST, 0.85)

        # Right: real cost
        lbl_cost = Text("Reality: forest costs 5x", font_size=17, color=C_COST, weight=BOLD)
        lbl_cost.move_to([3.5, 3.2, 0])
        cells_cost = build_grid(cols, rows, cs, np.array([3.5, 0.3, 0]))
        gv_cost = VGroup(*cells_cost.values())
        for f in forest:
            if f in cells_cost:
                cells_cost[f].set_fill(C_FOREST, 0.85)

        divider = Line([0, 3.4, 0], [0, -2.5, 0], color=GRAY_C, stroke_width=0.8)

        start = (0, 2)
        goal = (6, 2)
        sdots = [
            Dot(cells_bfs[start].get_center(), radius=0.09, color=C_START, z_index=5),
            Dot(cells_cost[start].get_center(), radius=0.09, color=C_START, z_index=5),
        ]
        gdots = [
            Dot(cells_bfs[goal].get_center(), radius=0.09, color=C_GOAL, z_index=5),
            Dot(cells_cost[goal].get_center(), radius=0.09, color=C_GOAL, z_index=5),
        ]

        self.play(
            FadeIn(lbl_bfs), FadeIn(lbl_cost), FadeIn(divider),
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv_bfs], lag_ratio=0.004),
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv_cost], lag_ratio=0.004),
            *[FadeIn(d) for d in sdots + gdots],
            run_time=0.8,
        )

        # Cost labels on forest cells (right side)
        for f in forest:
            if f in cells_cost:
                ct = Text("5", font_size=10, color=GRAY_A, z_index=4)
                ct.move_to(cells_cost[f].get_center())
                self.add(ct)

        # BFS path: straight through forest
        bfs_path = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]
        bfs_real = 2 + 3 * 5 + 1  # 2 plains + 3 forest + 1 plain = 18

        # Dijkstra path: around forest
        weights = {p: 5 for p in forest}
        wworld = WeightedGrid(cols, rows, weights=weights)
        pq = [(0, start)]
        csf = {start: 0}
        cf = {start: None}
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                break
            for n in wworld.neighbors(cur):
                nc = csf[cur] + wworld.cost(cur, n)
                if n not in csf or nc < csf[n]:
                    csf[n] = nc
                    heapq.heappush(pq, (nc, n))
                    cf[n] = cur
        dij_path = []
        c = goal
        while c is not None:
            dij_path.append(c)
            c = cf.get(c)
        dij_path.reverse()
        dij_cost = csf[goal]

        # Show BFS path
        for p in bfs_path:
            if p not in (start, goal):
                self.play(cells_bfs[p].animate.set_fill(C_PATH, 0.9), run_time=0.08)

        bfs_stat = Text(
            "6 steps ... but real cost = " + str(bfs_real) + "!",
            font_size=14, color=C_GOAL,
        )
        bfs_stat.next_to(gv_bfs, DOWN, buff=0.3)
        self.play(FadeIn(bfs_stat), run_time=0.3)
        self.wait(0.5)

        # Show Dijkstra path
        for p in dij_path:
            if p not in (start, goal):
                self.play(cells_cost[p].animate.set_fill(C_PATH, 0.9), run_time=0.06)

        dij_stat = Text(
            "More steps ... but real cost = " + str(dij_cost),
            font_size=14, color=C_PATH,
        )
        dij_stat.next_to(gv_cost, DOWN, buff=0.3)
        self.play(FadeIn(dij_stat), run_time=0.3)

        insight = info_box(
            "BFS finds the fewest steps, not the cheapest path.\n"
            "When movement costs vary, we need Dijkstra's Algorithm.",
            color=C_COST, font_size=14,
        )
        insight.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 9 — DIJKSTRA'S ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════
class DijkstraScene(Scene):
    def construct(self):
        section_card(self, 8, "Dijkstra's Algorithm",
                     "Use a priority queue to always pick the cheapest next step")

        cols, rows = 8, 6
        cs = 0.44
        grid_ctr = LEFT * 2.7 + DOWN * 0.1

        walls = [(3, 0), (3, 1)]
        forest = set()
        for c in range(3, 6):
            for r in range(2, 5):
                forest.add((c, r))
        weights = {p: 5 for p in forest}
        world = WeightedGrid(cols, rows, walls, weights)

        cells = build_grid(cols, rows, cs, grid_ctr)
        gv = VGroup(*cells.values())
        paint_walls(cells, walls)
        for f in forest:
            if f in cells and f not in walls:
                cells[f].set_fill(C_FOREST, 0.85)

        start, goal = (1, 2), (6, 3)
        s_dot = Dot(cells[start].get_center(), radius=0.11, color=C_START, z_index=5)
        g_dot = Dot(cells[goal].get_center(), radius=0.11, color=C_GOAL, z_index=5)

        code_lines = [
            "frontier = PriorityQueue()",
            "frontier.put( start, 0 )",
            "cost_so_far = { start: 0 }",
            "came_from   = { start: None }",
            " ",
            "while frontier not empty:",
            "  current = frontier.get()",
            "  if current == goal: break",
            " ",
            "  for next in neighbors(current):",
            "    new_cost = cost_so_far[current]",
            "              + cost(current, next)",
            "    if new_cost < best known cost:",
            "      cost_so_far[next] = new_cost",
            "      priority = new_cost",
            "      frontier.put(next, priority)",
            "      came_from[next] = current",
        ]
        code = CodePanel(code_lines, font_size=11, panel_width=4.5, line_spacing=0.21)
        code.pos(RIGHT * 4.1 + UP * 0.6)

        leg = VGroup(
            VGroup(
                Square(0.18, fill_color=C_FOREST, fill_opacity=0.85, stroke_width=0),
                Text("Forest (cost 5)", font_size=11, color=GRAY_B),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Square(0.18, fill_color=PANEL_LIGHT, fill_opacity=0.7, stroke_width=0),
                Text("Plains (cost 1)", font_size=11, color=GRAY_B),
            ).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        leg.next_to(gv, DOWN, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.004),
            FadeIn(s_dot), FadeIn(g_dot), FadeIn(leg),
            FadeIn(code.get_group()),
            run_time=1,
        )

        # Explain priority queue
        pq_note = info_box(
            "Key change: a Priority Queue always returns\n"
            "the lowest-cost item first. This makes the\n"
            "frontier expand slower through expensive terrain,\n"
            "unlike BFS which treats all steps equally.",
            color=C_COST, font_size=12,
        )
        pq_note.next_to(code.get_group(), DOWN, buff=0.15)
        self.play(FadeIn(pq_note, shift=UP * 0.1), run_time=0.5)
        self.wait(2)
        self.play(FadeOut(pq_note), run_time=0.3)

        # Run Dijkstra
        for i in [0, 1, 2, 3]:
            self.play(*code.hl_line(i), run_time=0.15)

        pq = [(0, start)]
        cost_so_far = {start: 0}
        came_from = {start: None}
        step = 0

        while pq and step < 55:
            step += 1
            self.play(*code.hl_line(5), run_time=0.05)
            _, current = heapq.heappop(pq)
            self.play(*code.hl_line(6), run_time=0.05)

            if current not in (start, goal):
                self.play(
                    cells[current].animate.set_fill(C_CURRENT, 0.85),
                    run_time=0.05,
                )

            self.play(*code.hl_line(7), run_time=0.04)
            if current == goal:
                break

            self.play(*code.hl_line(9), run_time=0.03)
            new_anims = []
            for nxt in world.neighbors(current):
                nc = cost_so_far[current] + world.cost(current, nxt)
                if nxt not in cost_so_far or nc < cost_so_far[nxt]:
                    cost_so_far[nxt] = nc
                    heapq.heappush(pq, (nc, nxt))
                    came_from[nxt] = current
                    if nxt not in (start, goal):
                        t = min(nc / 25, 1)
                        col = interpolate_color(
                            ManimColor(C_FRONTIER), ManimColor(C_VISITED), t,
                        )
                        new_anims.append(cells[nxt].animate.set_fill(col, 0.8))

            if new_anims:
                self.play(*code.hl_line(15), run_time=0.03)
                self.play(*new_anims, run_time=0.08)

            if current not in (start, goal):
                cells[current].set_fill(C_VISITED, 0.6)

        # Path
        self.play(*code.hl_off(), run_time=0.2)
        path = []
        c = goal
        while c is not None:
            path.append(c)
            c = came_from.get(c)
        path.reverse()

        parrs = VGroup()
        for i in range(len(path) - 1):
            a = Arrow(
                cells[path[i]].get_center(), cells[path[i + 1]].get_center(),
                buff=cs * 0.08, color=C_PATH, stroke_width=3,
                max_tip_length_to_length_ratio=0.2, z_index=4,
            )
            parrs.add(a)

        for p in path:
            if p not in (start, goal):
                self.play(cells[p].animate.set_fill(C_PATH, 0.9), run_time=0.07)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in parrs], lag_ratio=0.1),
            run_time=0.7,
        )

        insight = info_box(
            "Dijkstra goes around the forest - cheaper total cost!\n"
            "The priority queue ensures we always expand the cheapest option.",
            color=C_PATH, font_size=14,
        )
        insight.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=0.5)
        self.wait(2.5)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 10 — THE HEURISTIC
# ═══════════════════════════════════════════════════════════════════════════════
class HeuristicScene(Scene):
    def construct(self):
        section_card(self, 9, "Heuristic Functions",
                     "Estimating how far we still have to go")

        title = Text("What is a heuristic?", font_size=30,
                     color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.6)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        cols, rows = 8, 6
        cs = 0.48
        cells = build_grid(cols, rows, cs, LEFT * 0.5 + DOWN * 0.2)
        gv = VGroup(*cells.values())

        a_pos = (1, 4)
        b_pos = (6, 1)
        s_dot = Dot(cells[a_pos].get_center(), radius=0.12, color=C_START, z_index=5)
        g_dot = Dot(cells[b_pos].get_center(), radius=0.12, color=C_GOAL, z_index=5)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.004),
            FadeIn(s_dot), FadeIn(g_dot),
            run_time=0.7,
        )

        # Manhattan distance visualization
        corner_pos = (b_pos[0], a_pos[1])
        h_line = Line(
            cells[a_pos].get_center(), cells[corner_pos].get_center(),
            color=C_HEURISTIC, stroke_width=4,
        )
        v_line = Line(
            cells[corner_pos].get_center(), cells[b_pos].get_center(),
            color=C_HEURISTIC, stroke_width=4,
        )

        dx = abs(b_pos[0] - a_pos[0])
        dy = abs(b_pos[1] - a_pos[1])

        dx_lbl = Text("dx = " + str(dx), font_size=16, color=C_HEURISTIC)
        dx_lbl.next_to(h_line, DOWN, buff=0.1)
        dy_lbl = Text("dy = " + str(dy), font_size=16, color=C_HEURISTIC)
        dy_lbl.next_to(v_line, RIGHT, buff=0.1)

        self.play(Create(h_line), FadeIn(dx_lbl), run_time=0.5)
        self.play(Create(v_line), FadeIn(dy_lbl), run_time=0.5)

        h_val = dx + dy
        formula = Text(
            "h = |dx| + |dy| = " + str(dx) + " + " + str(dy)
            + " = " + str(h_val),
            font_size=18, color=C_HEURISTIC, weight=BOLD,
        )
        formula.to_edge(DOWN, buff=1.8)
        self.play(FadeIn(formula), run_time=0.4)
        self.wait(1)

        # Why heuristics matter
        why = VGroup(
            Text("Why estimate?", font_size=20, color=WHITE, weight=BOLD),
            Text("BFS and Dijkstra expand in all directions -",
                 font_size=14, color=GRAY_B),
            Text("they don't know where the goal is.",
                 font_size=14, color=GRAY_B),
            Text("A heuristic lets us bias the search",
                 font_size=14, color=GRAY_B),
            Text("toward the goal!", font_size=14, color=C_HEURISTIC),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        why.to_edge(RIGHT, buff=0.4).shift(UP * 0.5)
        self.play(
            LaggedStart(*[FadeIn(w, shift=LEFT * 0.1) for w in why],
                        lag_ratio=0.2),
            run_time=1,
        )

        rule = info_box(
            "Key rule: the heuristic must never overestimate.\n"
            "Manhattan distance is perfect for 4-directional grids.\n"
            "This guarantees A* will find the optimal path.",
            color=C_HEURISTIC, font_size=13,
        )
        rule.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(rule, shift=UP * 0.15), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 11 — GREEDY BEST-FIRST SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
class GreedyBFSScene(Scene):
    def construct(self):
        section_card(self, 10, "Greedy Best-First Search",
                     "Rush toward the goal: fast but risky")

        cols, rows = 8, 6
        cs = 0.35
        start, goal = (1, 2), (6, 2)

        # Map 1: simple wall — greedy does well
        walls1 = [(3, 1), (3, 2), (3, 3)]
        w1 = GridWorld(cols, rows, walls1)

        # Map 2: U-shaped trap — greedy gets fooled
        walls2 = [
            (3, 0), (3, 1), (3, 2), (3, 3),
            (5, 1), (5, 2), (5, 3), (5, 4), (4, 4),
        ]
        w2 = GridWorld(cols, rows, walls2)

        # Row 1: Simple map
        r1_title = Text("Simple map: Greedy wins!", font_size=16,
                        color=C_HEURISTIC, weight=BOLD)
        r1_title.move_to([0, 3.2, 0])

        lbl_d1 = Text("Dijkstra", font_size=13, color=C_COST, weight=BOLD)
        lbl_d1.move_to([-3.5, 2.7, 0])
        cells_d1 = build_grid(cols, rows, cs, np.array([-3.5, 1.0, 0]))
        gv_d1 = VGroup(*cells_d1.values())
        paint_walls(cells_d1, walls1)

        lbl_g1 = Text("Greedy", font_size=13, color=C_HEURISTIC, weight=BOLD)
        lbl_g1.move_to([3.5, 2.7, 0])
        cells_g1 = build_grid(cols, rows, cs, np.array([3.5, 1.0, 0]))
        gv_g1 = VGroup(*cells_g1.values())
        paint_walls(cells_g1, walls1)

        div1 = Line([0, 2.8, 0], [0, -0.5, 0], color=GRAY_C, stroke_width=0.6)

        # Row 2: Trap map
        r2_title = Text("Trap map: Greedy gets fooled!", font_size=16,
                        color=C_GOAL, weight=BOLD)
        r2_title.move_to([0, -0.8, 0])

        lbl_d2 = Text("Dijkstra", font_size=13, color=C_COST, weight=BOLD)
        lbl_d2.move_to([-3.5, -1.3, 0])
        cells_d2 = build_grid(cols, rows, cs, np.array([-3.5, -2.8, 0]))
        gv_d2 = VGroup(*cells_d2.values())
        paint_walls(cells_d2, walls2)

        lbl_g2 = Text("Greedy", font_size=13, color=C_HEURISTIC, weight=BOLD)
        lbl_g2.move_to([3.5, -1.3, 0])
        cells_g2 = build_grid(cols, rows, cs, np.array([3.5, -2.8, 0]))
        gv_g2 = VGroup(*cells_g2.values())
        paint_walls(cells_g2, walls2)

        div2 = Line([0, -0.7, 0], [0, -3.8, 0], color=GRAY_C, stroke_width=0.6)

        all_grids = [gv_d1, gv_g1, gv_d2, gv_g2]
        all_labels = [lbl_d1, lbl_g1, lbl_d2, lbl_g2, r1_title, r2_title]
        sdots = []
        gdots_list = []
        for cc_map in [cells_d1, cells_g1, cells_d2, cells_g2]:
            sdots.append(
                Dot(cc_map[start].get_center(), radius=0.06, color=C_START, z_index=5)
            )
            gdots_list.append(
                Dot(cc_map[goal].get_center(), radius=0.06, color=C_GOAL, z_index=5)
            )

        self.play(
            *[FadeIn(l) for l in all_labels],
            FadeIn(div1), FadeIn(div2),
            *[
                LaggedStart(*[FadeIn(c, scale=0.9) for c in g], lag_ratio=0.003)
                for g in all_grids
            ],
            *[FadeIn(d) for d in sdots + gdots_list],
            run_time=1,
        )

        def run_dij(w, s, g):
            pq_l = [(0, s)]
            co = {s: 0}
            cf = {s: None}
            order = []
            while pq_l:
                _, cur = heapq.heappop(pq_l)
                order.append(cur)
                if cur == g:
                    break
                for n in w.neighbors(cur):
                    nc = co[cur] + w.cost(cur, n)
                    if n not in co or nc < co[n]:
                        co[n] = nc
                        heapq.heappush(pq_l, (nc, n))
                        cf[n] = cur
            path = []
            c = g
            while c is not None:
                path.append(c)
                c = cf.get(c)
            return order, list(reversed(path))

        def run_greedy(w, s, g):
            pq_l = [(heuristic_fn(s, g), s)]
            cf = {s: None}
            order = []
            while pq_l:
                _, cur = heapq.heappop(pq_l)
                order.append(cur)
                if cur == g:
                    break
                for n in w.neighbors(cur):
                    if n not in cf:
                        heapq.heappush(pq_l, (heuristic_fn(n, g), n))
                        cf[n] = cur
            path = []
            c = g
            while c is not None:
                path.append(c)
                c = cf.get(c)
            return order, list(reversed(path))

        runs = [
            (run_dij, w1, cells_d1, C_COST),
            (run_greedy, w1, cells_g1, C_HEURISTIC),
            (run_dij, w2, cells_d2, C_COST),
            (run_greedy, w2, cells_g2, C_HEURISTIC),
        ]
        results = []
        for func, w, cc, col in runs:
            order, path = func(w, start, goal)
            results.append((order, path, cc, col))

        # Animate per row
        for row_results in [results[:2], results[2:]]:
            mx = max(len(r[0]) for r in row_results)
            for i in range(0, mx, 3):
                anims = []
                for order, path, cc, col in row_results:
                    for j in range(3):
                        idx = i + j
                        if idx < len(order):
                            p = order[idx]
                            if p not in (start, goal):
                                anims.append(cc[p].animate.set_fill(col, 0.5))
                if anims:
                    self.play(*anims, run_time=0.06)

        # Show paths
        for order, path, cc, col in results:
            for p in path:
                if p not in (start, goal):
                    self.play(cc[p].animate.set_fill(C_PATH, 0.9), run_time=0.04)

        # Stats
        for order, path, cc, col in results:
            bottom = VGroup(*cc.values()).get_bottom()
            st = Text(
                "Exp " + str(len(order)) + " Path " + str(len(path)),
                font_size=11, color=GRAY_B,
            )
            st.next_to(bottom, DOWN, buff=0.15)
            self.play(FadeIn(st), run_time=0.15)

        insight = info_box(
            "Greedy is fast when unobstructed, but it can be\n"
            "tricked by walls into finding longer paths.\n"
            "It only considers distance-to-goal, ignoring cost-so-far.",
            color=C_HEURISTIC, font_size=13,
        )
        insight.to_edge(DOWN, buff=0.08)
        self.play(FadeIn(insight, shift=UP * 0.1), run_time=0.4)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 12 — A* ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════
class AStarScene(Scene):
    def construct(self):
        section_card(self, 11, "The A* Algorithm",
                     "Best of both worlds: cost + heuristic")

        # Part A — Formula explanation
        idea = Text("The key insight", font_size=28, color=WHITE, weight=BOLD)
        idea.to_edge(UP, buff=0.6)

        parts = [
            Text("priority", font_size=24, color=C_ACCENT, weight=BOLD),
            Text("=", font_size=24, color=WHITE),
            Text("cost_so_far", font_size=24, color=C_COST, weight=BOLD),
            Text("+", font_size=24, color=WHITE),
            Text("heuristic", font_size=24, color=C_HEURISTIC, weight=BOLD),
        ]
        eq = VGroup(*parts).arrange(RIGHT, buff=0.2)

        sub_cost = Text("actual distance\nfrom start", font_size=13,
                        color=C_COST, line_spacing=1.2)
        sub_cost.next_to(parts[2], DOWN, buff=0.2)
        sub_heur = Text("estimated distance\nto goal", font_size=13,
                        color=C_HEURISTIC, line_spacing=1.2)
        sub_heur.next_to(parts[4], DOWN, buff=0.2)

        why = VGroup(
            Text("Dijkstra uses only cost_so_far",
                 font_size=15, color=C_COST),
            Text("   Finds shortest paths but explores everywhere",
                 font_size=13, color=GRAY_B),
            Text(" ", font_size=6, color=BG),
            Text("Greedy uses only heuristic",
                 font_size=15, color=C_HEURISTIC),
            Text("   Fast but can find non-optimal paths",
                 font_size=13, color=GRAY_B),
            Text(" ", font_size=6, color=BG),
            Text("A* uses both: optimal AND focused!",
                 font_size=16, color=C_ACCENT, weight=BOLD),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        why.move_to(DOWN * 1.5)

        self.play(FadeIn(idea, shift=DOWN * 0.2), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(p, shift=RIGHT * 0.1) for p in parts],
                        lag_ratio=0.1),
            run_time=0.8,
        )
        self.play(FadeIn(sub_cost), FadeIn(sub_heur), run_time=0.5)
        self.wait(1)
        self.play(
            LaggedStart(*[FadeIn(w, shift=LEFT * 0.1) for w in why],
                        lag_ratio=0.15),
            run_time=1.2,
        )
        self.wait(3)
        fade_all(self)

        # Part B — Full A* demo with code stepping
        cols, rows = 8, 6
        cs = 0.44
        grid_ctr = LEFT * 2.7 + DOWN * 0.1
        walls = [(3, 1), (3, 2), (3, 3), (3, 4), (5, 0), (5, 1), (5, 2)]
        world = GridWorld(cols, rows, walls)
        cells = build_grid(cols, rows, cs, grid_ctr)
        gv = VGroup(*cells.values())
        paint_walls(cells, walls)

        start, goal = (1, 2), (6, 4)
        s_dot = Dot(cells[start].get_center(), radius=0.11, color=C_START, z_index=5)
        g_dot = Dot(cells[goal].get_center(), radius=0.11, color=C_GOAL, z_index=5)
        s_lbl = Text("S", font_size=11, color=WHITE, weight=BOLD, z_index=6)
        s_lbl.move_to(cells[start])
        g_lbl = Text("G", font_size=11, color=WHITE, weight=BOLD, z_index=6)
        g_lbl.move_to(cells[goal])

        code_lines = [
            "frontier = PriorityQueue()",
            "frontier.put( start, 0 )",
            "cost_so_far = { start: 0 }",
            "came_from   = { start: None }",
            " ",
            "while frontier not empty:",
            "  current = frontier.get()",
            "  if current == goal: break",
            " ",
            "  for next in neighbors(current):",
            "    new_cost = cost_so_far[current]",
            "              + cost(current, next)",
            "    if new_cost < best known:",
            "      cost_so_far[next] = new_cost",
            "      priority = new_cost",
            "               + heuristic(goal,next)",
            "      frontier.put(next, priority)",
            "      came_from[next] = current",
        ]
        code = CodePanel(code_lines, font_size=11, panel_width=4.5, line_spacing=0.21)
        code.pos(RIGHT * 4.1 + UP * 0.6)

        leg = VGroup(
            VGroup(
                Square(0.16, fill_color=C_FRONTIER, fill_opacity=0.85, stroke_width=0),
                Text("Frontier", font_size=11, color=GRAY_B),
            ).arrange(RIGHT, buff=0.08),
            VGroup(
                Square(0.16, fill_color=C_VISITED, fill_opacity=0.65, stroke_width=0),
                Text("Explored", font_size=11, color=GRAY_B),
            ).arrange(RIGHT, buff=0.08),
            VGroup(
                Square(0.16, fill_color=C_PATH, fill_opacity=0.9, stroke_width=0),
                Text("Final path", font_size=11, color=GRAY_B),
            ).arrange(RIGHT, buff=0.08),
        ).arrange(RIGHT, buff=0.3).next_to(gv, DOWN, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in gv], lag_ratio=0.004),
            FadeIn(s_dot), FadeIn(s_lbl), FadeIn(g_dot), FadeIn(g_lbl),
            FadeIn(code.get_group()), FadeIn(leg),
            run_time=0.9,
        )

        # Run A*
        for i in [0, 1, 2, 3]:
            self.play(*code.hl_line(i), run_time=0.15)
        cells[start].set_fill(C_FRONTIER, 0.85)

        pq = [(0, 0, start)]  # (priority, tiebreaker, pos)
        cost_so_far = {start: 0}
        came_from = {start: None}
        counter = 0
        step = 0

        while pq and step < 55:
            step += 1
            self.play(*code.hl_line(5), run_time=0.04)
            _, _, current = heapq.heappop(pq)
            self.play(*code.hl_line(6), run_time=0.04)

            if current not in (start, goal):
                self.play(
                    cells[current].animate.set_fill(C_CURRENT, 0.85),
                    run_time=0.05,
                )

            self.play(*code.hl_line(7), run_time=0.03)
            if current == goal:
                break

            self.play(*code.hl_line(9), run_time=0.03)
            new_anims = []
            for nxt in world.neighbors(current):
                nc = cost_so_far[current] + world.cost(current, nxt)
                if nxt not in cost_so_far or nc < cost_so_far[nxt]:
                    cost_so_far[nxt] = nc
                    pri = nc + heuristic_fn(goal, nxt)
                    counter += 1
                    heapq.heappush(pq, (pri, counter, nxt))
                    came_from[nxt] = current
                    if nxt not in (start, goal):
                        new_anims.append(
                            cells[nxt].animate.set_fill(C_FRONTIER, 0.8)
                        )

            if new_anims:
                self.play(*code.hl_line(16), run_time=0.03)
                self.play(*new_anims, run_time=0.08)

            if current not in (start, goal):
                cells[current].set_fill(C_VISITED, 0.6)

        # Path
        self.play(*code.hl_off(), run_time=0.2)
        path = []
        c = goal
        while c is not None:
            path.append(c)
            c = came_from.get(c)
        path.reverse()

        for p in path:
            if p not in (start, goal):
                self.play(cells[p].animate.set_fill(C_PATH, 0.9), run_time=0.08)

        parrs = VGroup()
        for i in range(len(path) - 1):
            a = Arrow(
                cells[path[i]].get_center(),
                cells[path[i + 1]].get_center(),
                buff=cs * 0.08, color=C_PATH, stroke_width=3,
                max_tip_length_to_length_ratio=0.2, z_index=4,
            )
            parrs.add(a)
        self.play(
            LaggedStart(*[GrowArrow(a) for a in parrs], lag_ratio=0.1),
            run_time=0.7,
        )

        insight = info_box(
            "A* finds the shortest path like Dijkstra, but\n"
            "explores far fewer cells. The heuristic guides\n"
            "the search toward the goal without sacrificing optimality.",
            color=C_ACCENT, font_size=14,
        )
        insight.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=0.5)
        self.wait(3)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 13 — TRIPLE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
class TripleComparisonScene(Scene):
    def construct(self):
        section_card(self, 12, "Side-by-Side Comparison",
                     "Dijkstra vs Greedy vs A* on the same map")

        cols, rows = 7, 5
        cs = 0.34
        walls = [(3, 0), (3, 1), (3, 2), (3, 3)]
        world = GridWorld(cols, rows, walls)
        start, goal = (0, 2), (6, 2)

        algo_info = [
            ("Dijkstra", C_COST, LEFT * 4.5),
            ("Greedy", C_HEURISTIC, ORIGIN),
            ("A*", C_ACCENT, RIGHT * 4.5),
        ]

        all_cells = {}
        for name, col, off in algo_info:
            lbl = Text(name, font_size=18, color=col, weight=BOLD)
            lbl.move_to(off + UP * 2.8)
            self.add(lbl)
            cc = build_grid(cols, rows, cs, off + DOWN * 0.2)
            paint_walls(cc, walls)
            self.add(VGroup(*cc.values()))
            self.add(
                Dot(cc[start].get_center(), radius=0.07, color=C_START, z_index=5)
            )
            self.add(
                Dot(cc[goal].get_center(), radius=0.07, color=C_GOAL, z_index=5)
            )
            all_cells[name] = cc

        self.wait(0.3)

        def run_dij(w, s, g):
            pq_l = [(0, s)]
            co = {s: 0}
            cf = {s: None}
            order = []
            while pq_l:
                _, cur = heapq.heappop(pq_l)
                order.append(cur)
                if cur == g:
                    break
                for n in w.neighbors(cur):
                    nc = co[cur] + w.cost(cur, n)
                    if n not in co or nc < co[n]:
                        co[n] = nc
                        heapq.heappush(pq_l, (nc, n))
                        cf[n] = cur
            path = []
            c = g
            while c is not None:
                path.append(c)
                c = cf.get(c)
            return order, list(reversed(path))

        def run_greedy(w, s, g):
            pq_l = [(heuristic_fn(s, g), s)]
            cf = {s: None}
            order = []
            while pq_l:
                _, cur = heapq.heappop(pq_l)
                order.append(cur)
                if cur == g:
                    break
                for n in w.neighbors(cur):
                    if n not in cf:
                        heapq.heappush(pq_l, (heuristic_fn(n, g), n))
                        cf[n] = cur
            path = []
            c = g
            while c is not None:
                path.append(c)
                c = cf.get(c)
            return order, list(reversed(path))

        def run_astar(w, s, g):
            pq_l = [(0, s)]
            co = {s: 0}
            cf = {s: None}
            order = []
            while pq_l:
                _, cur = heapq.heappop(pq_l)
                order.append(cur)
                if cur == g:
                    break
                for n in w.neighbors(cur):
                    nc = co[cur] + w.cost(cur, n)
                    if n not in co or nc < co[n]:
                        co[n] = nc
                        heapq.heappush(pq_l, (nc + heuristic_fn(n, g), n))
                        cf[n] = cur
            path = []
            c = g
            while c is not None:
                path.append(c)
                c = cf.get(c)
            return order, list(reversed(path))

        funcs = [run_dij, run_greedy, run_astar]
        results = {}
        for (name, col, _), func in zip(algo_info, funcs):
            order, path = func(world, start, goal)
            results[name] = (order, path, col)

        mx = max(len(r[0]) for r in results.values())

        for i in range(0, mx, 3):
            anims = []
            for name in results:
                order, path, col = results[name]
                cc = all_cells[name]
                for j in range(3):
                    idx = i + j
                    if idx < len(order):
                        p = order[idx]
                        if p not in (start, goal):
                            anims.append(cc[p].animate.set_fill(col, 0.5))
            if anims:
                self.play(*anims, run_time=0.06)

        for name in results:
            order, path, col = results[name]
            cc = all_cells[name]
            for p in path:
                if p not in (start, goal):
                    self.play(cc[p].animate.set_fill(C_PATH, 0.9), run_time=0.03)

        for name, col, off in algo_info:
            order, path = results[name][:2]
            cc = all_cells[name]
            grid_bot = VGroup(*cc.values()).get_bottom()
            stat = Text(
                "Explored: " + str(len(order)) + "  Path: " + str(len(path)),
                font_size=12, color=GRAY_B,
            )
            stat.next_to(grid_bot, DOWN, buff=0.2)
            stat.move_to([off[0], stat.get_y(), 0])
            self.play(FadeIn(stat), run_time=0.15)

        insight = info_box(
            "A* explores as little as Greedy when unobstructed,\n"
            "and finds optimal paths like Dijkstra when obstructed.\n"
            "It truly is the best general-purpose pathfinder.",
            color=C_ACCENT, font_size=14,
        )
        insight.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=0.5)
        self.wait(3.5)
        fade_all(self)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENE 14 — SUMMARY / DECISION GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
class SummaryScene(Scene):
    def construct(self):
        section_card(self, 13, "Which Algorithm?",
                     "Choosing the right tool for the job")

        def make_box(text, txt_color, border_color, is_answer=False):
            lbl = Text(text, font_size=14, color=txt_color, line_spacing=1.2)
            box = RoundedRectangle(
                corner_radius=0.15,
                width=max(lbl.width + 0.5, 2.0),
                height=lbl.height + 0.35,
                fill_color=border_color if is_answer else PANEL,
                fill_opacity=0.25 if is_answer else 0.85,
                stroke_color=border_color,
                stroke_width=2 if is_answer else 1.2,
            )
            lbl.move_to(box)
            return VGroup(box, lbl)

        q1 = make_box("Do all moves\ncost the same?", WHITE, GRAY_B)
        a_bfs = make_box("BFS", C_VISITED, C_VISITED, True)
        q2 = make_box("Need to reach\none specific goal?", WHITE, GRAY_B)
        a_dij = make_box("Dijkstra", C_COST, C_COST, True)
        q3 = make_box("Need the\nshortest path?", WHITE, GRAY_B)
        a_greedy = make_box("Greedy\nBest-First", C_HEURISTIC, C_HEURISTIC, True)
        a_astar = make_box("A*", C_ACCENT, C_ACCENT, True)

        q1.move_to(UP * 2.2 + LEFT * 1.5)
        a_bfs.next_to(q1, RIGHT, buff=1.2)
        q2.next_to(q1, DOWN, buff=0.7)
        a_dij.next_to(q2, RIGHT, buff=1.2)
        q3.next_to(q2, DOWN, buff=0.7)
        a_greedy.next_to(q3, RIGHT, buff=1.2)
        a_astar.next_to(q3, DOWN, buff=0.7)

        def arrow_lbl(src, dst, label, dir_type="right"):
            if dir_type == "right":
                arr = Arrow(
                    src.get_right(), dst.get_left(), buff=0.08,
                    color=GRAY_B, stroke_width=2,
                    max_tip_length_to_length_ratio=0.12,
                )
                lb = Text(label, font_size=11, color=GRAY_A)
                lb.next_to(arr, UP, buff=0.05)
            else:
                arr = Arrow(
                    src.get_bottom(), dst.get_top(), buff=0.08,
                    color=GRAY_B, stroke_width=2,
                    max_tip_length_to_length_ratio=0.12,
                )
                lb = Text(label, font_size=11, color=GRAY_A)
                lb.next_to(arr, LEFT, buff=0.05)
            return VGroup(arr, lb)

        arrows = [
            arrow_lbl(q1, a_bfs, "Yes", "right"),
            arrow_lbl(q1, q2, "No", "down"),
            arrow_lbl(q2, a_dij, "No", "right"),
            arrow_lbl(q2, q3, "Yes", "down"),
            arrow_lbl(q3, a_greedy, "No", "right"),
            arrow_lbl(q3, a_astar, "Yes", "down"),
        ]

        # Animate step by step
        self.play(FadeIn(q1, shift=DOWN * 0.15), run_time=0.4)
        self.play(GrowFromPoint(arrows[0], q1.get_right()), run_time=0.3)
        self.play(FadeIn(a_bfs, shift=LEFT * 0.15), run_time=0.3)

        self.play(GrowFromPoint(arrows[1], q1.get_bottom()), run_time=0.3)
        self.play(FadeIn(q2, shift=DOWN * 0.15), run_time=0.3)
        self.play(GrowFromPoint(arrows[2], q2.get_right()), run_time=0.3)
        self.play(FadeIn(a_dij, shift=LEFT * 0.15), run_time=0.3)

        self.play(GrowFromPoint(arrows[3], q2.get_bottom()), run_time=0.3)
        self.play(FadeIn(q3, shift=DOWN * 0.15), run_time=0.3)
        self.play(GrowFromPoint(arrows[4], q3.get_right()), run_time=0.3)
        self.play(FadeIn(a_greedy, shift=LEFT * 0.15), run_time=0.3)

        self.play(GrowFromPoint(arrows[5], q3.get_bottom()), run_time=0.3)
        self.play(FadeIn(a_astar, scale=1.15), run_time=0.4)

        star = Star(
            n=5, outer_radius=0.55, inner_radius=0.25,
            color=C_ACCENT, fill_opacity=0.12, stroke_width=1,
        )
        star.move_to(a_astar)
        self.play(GrowFromCenter(star), run_time=0.4)
        self.wait(1.5)

        # Key takeaways
        takeaways = VGroup(
            Text("BFS and Dijkstra guarantee the shortest path",
                 font_size=13, color=C_COST),
            Text("Greedy is fast but not always optimal",
                 font_size=13, color=C_HEURISTIC),
            Text("A* is optimal AND focused (best general choice)",
                 font_size=13, color=C_ACCENT),
            Text("As heuristic shrinks toward 0, A* becomes Dijkstra",
                 font_size=12, color=GRAY_B),
            Text("As heuristic grows large, A* becomes Greedy",
                 font_size=12, color=GRAY_B),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        tk_bg = RoundedRectangle(
            corner_radius=0.15,
            width=takeaways.width + 0.5,
            height=takeaways.height + 0.4,
            fill_color=PANEL, fill_opacity=0.95,
            stroke_color=C_ACCENT, stroke_width=1.2,
        )
        tk_bg.move_to(takeaways)
        tk_grp = VGroup(tk_bg, takeaways)
        tk_grp.to_edge(DOWN, buff=0.15)
        self.play(FadeIn(tk_grp, shift=UP * 0.15), run_time=0.6)
        self.wait(4)

        fade_all(self)

        thanks = Text("Happy Pathfinding!", font_size=42,
                      color=C_ACCENT, weight=BOLD)
        credit = Text("Based on Red Blob Games by Amit Patel",
                      font_size=16, color=GRAY_B)
        outro = VGroup(thanks, credit).arrange(DOWN, buff=0.3)
        self.play(FadeIn(outro, scale=0.9), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(outro), run_time=0.6)


if __name__ == '__main__':
    a = AStarScene()
    a.render()
