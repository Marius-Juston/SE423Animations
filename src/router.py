import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple, Set

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- Configuration ---
GRID_SCALE = 0.1  # Resolution of the routing grid
BEND_PENALTY = 50  # Cost added for making a 90-degree turn
BASE_COST = 1  # Cost to move 1 unit
MAX_ITERATIONS = 4  # Max rip-up and reroute attempts

MARGIN = 2


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __le__(self, other):
        return (self.x, self.y) <= (other.x, other.y)

    def __gt__(self, other):
        return (self.x, self.y) > (other.x, other.y)


@dataclass
class Component:
    name: str
    x: float
    y: float
    width: float
    height: float

    @property
    def bbox(self):
        # Returns (min_x, min_y, max_x, max_y)
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def int_bbox(self):
        # Returns (min_x, min_y, max_x, max_y)
        return (math.floor(self.x), math.floor(self.y), math.ceil(self.x + self.width), math.ceil(self.y + self.height))

    def __add__(self, margin: float):
        return Component(self.name, self.x - margin, self.y - margin, self.width + margin * 2, self.height + margin * 2)


class Router:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # grid_occupancy tracks how many nets use a specific cell (x, y)
        self.grid_occupancy = np.zeros((width, height), dtype=int)
        # history_cost tracks historically congested cells to discourage repeated use
        self.history_cost = np.zeros((width, height), dtype=float)
        self.components = []
        self.nets = []  # List of list of Points
        self.routed_paths = {}  # Map net_id -> List of Points

    def _pin_escape(self, pin: Point):
        """
        If pin is inside a component margin, return (escape_path, escape_point).
        Otherwise, return ([], pin).
        """
        for comp in self.components:
            comp_expanded = comp + MARGIN
            xmin, ymin, xmax, ymax = comp_expanded.int_bbox

            if xmin <= pin.x < xmax and ymin <= pin.y < ymax:
                # Determine nearest face
                dists = {
                    Point(-1, 0): abs(pin.x - xmin),
                    Point(1, 0): abs(xmax - pin.x),
                    Point(0, -1): abs(pin.y - ymin),
                    Point(0, 1): abs(ymax - pin.y),
                }
                direction = min(dists, key=dists.get)

                path = [pin]
                p = pin
                while True:
                    p = p + direction
                    path.append(p)
                    if not (xmin <= p.x < xmax and ymin <= p.y < ymax):
                        break

                return path, p

        return [], pin

    def add_component(self, comp: Component):
        self.components.append(comp)
        comp_scale = comp + MARGIN
        comp_int_bounding = comp_scale.int_bbox

        # Mark component area as effectively blocked
        # Use a cost high enough that going around is ALWAYS cheaper,
        # but low enough that we can 'escape' a pin if it starts inside.
        OBSTACLE_COST = 100000

        for i in range(comp_int_bounding[0], comp_int_bounding[2]):
            for j in range(comp_int_bounding[1], comp_int_bounding[3]):
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.grid_occupancy[i, j] += OBSTACLE_COST

    def add_net(self, pins: List[Tuple[int, int]]):
        self.nets.append([Point(x, y) for x, y in pins])

    def _get_neighbors(self, current: Point):
        neighbors = []
        directions = [Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1)]
        for d in directions:
            n = current + d
            if 0 <= n.x < self.width and 0 <= n.y < self.height:
                neighbors.append(n)
        return neighbors

    def _decompose_multipins(self, pins: List[Point]) -> List[Tuple[Point, Point]]:
        """
        Decomposes a multi-pin net into a set of 2-pin connections using MST.
        This approximates the Rectilinear Steiner Minimal Tree.
        """
        if len(pins) < 2:
            return []

        # Create a complete graph where weights are Manhattan distances
        G = nx.Graph()
        for i, p1 in enumerate(pins):
            for j, p2 in enumerate(pins):
                if i < j:
                    dist = abs(p1.x - p2.x) + abs(p1.y - p2.y)
                    G.add_edge(i, j, weight=dist)

        # Compute Minimum Spanning Tree
        mst = nx.minimum_spanning_tree(G)

        # Return pairs of points to connect
        connections = []
        for u, v in mst.edges():
            connections.append((pins[u], pins[v]))
        return connections

    # ---------------------------------------------------------
    # 1. UPGRADED A*: Supports Multiple Targets (Steiner Zones)
    # ---------------------------------------------------------
    def _astar_multi_target(self, start: Point, targets: Set[Point], congestion_multiplier: float):
        pq = []
        # Priority Queue: (f_score, g_score, current_point, last_direction)
        heapq.heappush(pq, (0, 0, start, Point(0, 0)))

        came_from = {}
        g_scores = {(start, Point(0, 0)): 0}

        best_end_state = None
        min_g_to_target = float('inf')

        while pq:
            f, g, current, last_dir = heapq.heappop(pq)

            if g >= min_g_to_target:
                continue

            if current in targets:
                if g < min_g_to_target:
                    min_g_to_target = g
                    best_end_state = (current, last_dir)
                continue

            for neighbor in self._get_neighbors(current):
                new_dir = neighbor - current

                # 1. Base Cost
                step_cost = BASE_COST
                if last_dir != Point(0, 0) and new_dir != last_dir:
                    step_cost += BEND_PENALTY

                # 2. Congestion Cost
                occ = max(0, self.grid_occupancy[neighbor.x, neighbor.y])
                hist = self.history_cost[neighbor.x, neighbor.y]

                # CRITICAL FIX: Pin Exemption
                # If the neighbor is a target, we ignore its occupancy cost.
                # This allows the wire to 'dock' into a pin located inside a red margin.
                if neighbor in targets:
                    congestion_cost = 0
                else:
                    congestion_cost = (occ * congestion_multiplier) + hist

                new_g = g + step_cost + congestion_cost

                if new_g < g_scores.get((neighbor, new_dir), float('inf')):
                    g_scores[(neighbor, new_dir)] = new_g
                    h = min(abs(neighbor.x - t.x) + abs(neighbor.y - t.y) for t in targets)
                    heapq.heappush(pq, (new_g + h, new_g, neighbor, new_dir))
                    came_from[(neighbor, new_dir)] = (current, last_dir)

        if best_end_state:
            path = []
            curr, direction = best_end_state
            while curr != start:
                path.append(curr)
                if (curr, direction) not in came_from: break
                prev_node, prev_dir = came_from[(curr, direction)]
                curr = prev_node
                direction = prev_dir
            path.append(start)
            return path[::-1]

        return None

    # ---------------------------------------------------------
    # 2. NEW ROUTING LOGIC: Iterative Steiner Construction
    # ---------------------------------------------------------
    def route(self):
        congestion_multiplier = 0.5
        print(f"Starting Steiner Routing for {len(self.nets)} nets...")

        for iteration in range(MAX_ITERATIONS):
            # 1. Clear Occupancy
            self.grid_occupancy.fill(0)

            # 2. Re-apply Component Obstacles (High Cost)
            OBSTACLE_COST = 100000
            for c in self.components:
                c_scale = c + MARGIN
                bbox = c_scale.int_bbox
                # Use slicing for speed
                # Note: bbox is (min_x, min_y, max_x, max_y)
                # Numpy slicing is [min_x:max_x, min_y:max_y]
                self.grid_occupancy[bbox[0]:bbox[2], bbox[1]:bbox[3]] += OBSTACLE_COST

            # 3. Re-apply Net Occupancy from previous iteration (for negotiations)
            # We want wires to negotiate with OTHER wires, but never with components.
            # So wires add a smaller cost (e.g., 1) compared to components (100,000)
            for net_id, path in self.routed_paths.items():
                for p in path:
                    self.grid_occupancy[p.x, p.y] += 100  # Wire overlap cost

            current_overlaps = 0

            # 4. Route Each Net
            for net_id, pins in enumerate(self.nets):
                if not pins: continue

                # Rip-up current net from occupancy to route it fresh
                if net_id in self.routed_paths:
                    for p in self.routed_paths[net_id]:
                        self.grid_occupancy[p.x, p.y] -= 100

                routed_pixels = set()
                full_net_path = []

                # ---- NEW: escape first pin ----
                escape_path, escape_point = self._pin_escape(pins[0])
                for p in escape_path:
                    routed_pixels.add(p)
                    full_net_path.append(p)

                routed_pixels.add(escape_point)
                unconnected_pins = set(pins[1:])

                while unconnected_pins:
                    best_pin = None
                    best_dist = float('inf')

                    for p in unconnected_pins:
                        d = min(abs(p.x - rp.x) + abs(p.y - rp.y) for rp in routed_pixels)
                        if d < best_dist:
                            best_dist = d
                            best_pin = p

                    # ---- NEW: escape target pin first ----
                    escape_path, escaped_pin = self._pin_escape(best_pin)
                    path = self._astar_multi_target(
                        escaped_pin,
                        routed_pixels,
                        congestion_multiplier
                    )

                    if path:
                        # Add escape stub
                        for p in escape_path:
                            routed_pixels.add(p)
                            full_net_path.append(p)

                        # Add A* path
                        for p in path:
                            routed_pixels.add(p)
                            full_net_path.append(p)

                        unconnected_pins.remove(best_pin)
                    else:
                        break

                self.routed_paths[net_id] = full_net_path

                # Re-apply occupancy
                for p in full_net_path:
                    self.grid_occupancy[p.x, p.y] += 100

            # 5. Check Congestion
            # We only care about Wire-Wire overlaps (value > 100 but < OBSTACLE_COST)
            # Or strict overlaps where value > 100 (implies 2 wires)
            # Component overlap is technically allowed (value > 100000) only for escape

            # Simple overlap check: finding cells with > 1 wire
            # Since 1 wire = 100, 2 wires = 200. Components are 100000.
            # We check modulo or ranges.

            overlaps = []
            for x in range(self.width):
                for y in range(self.height):
                    val = self.grid_occupancy[x, y]
                    # If val > 100000: It's a component.
                    # If val % 100000 > 100: It implies > 1 wire is here.
                    wire_cost = val % 100000
                    if wire_cost > 100:  # More than 1 wire (each wire is 100)
                        overlaps.append((x, y))

            current_overlaps = len(overlaps)
            print(f"  Iteration {iteration + 1}: Overlaps: {current_overlaps}")

            if current_overlaps == 0:
                break

            for x, y in overlaps:
                self.history_cost[x, y] += 10 * (iteration + 1)

            congestion_multiplier *= 1.5

    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw Grid (Optional, makes it look like graph paper)
        ax.grid(True, which='both', color='lightgrey', linestyle='--', alpha=0.5)

        # Draw Components
        for comp in self.components:
            comp_border = comp + MARGIN

            rect = patches.Rectangle((comp_border.x, comp_border.y), comp_border.width, comp_border.height,
                                     linewidth=0, facecolor='red', zorder=1)
            ax.add_patch(rect)

            rect = patches.Rectangle((comp.x, comp.y), comp.width, comp.height,
                                     linewidth=2, edgecolor='black', facecolor='lightgray', zorder=2)
            ax.add_patch(rect)

            ax.text(comp.x + comp.width / 2, comp.y + comp.height / 2, comp.name,
                    ha='center', va='center', weight='bold')

        # Draw Routes
        colors = plt.cm.jet(np.linspace(0, 1, len(self.nets)))

        for net_id, path in self.routed_paths.items():
            if not path: continue

            segments = [[]]

            prev_point = path[0]

            segement_counter = 0

            segments[segement_counter].append(prev_point)

            for i in range(1, len(path)):
                new_point = path[i]

                if not (new_point.x == prev_point.x or new_point.y == prev_point.y):
                    segement_counter += 1
                    segments.append([])

                prev_point = new_point

                segments[segement_counter].append(new_point)

            for path in segments:
                xs = [p.x for p in path]
                ys = [p.y for p in path]

                # Draw wires
                ax.plot(xs, ys, color=colors[net_id], linewidth=2.5, alpha=0.8, zorder=1)

                # Draw pins
                pins = self.nets[net_id]
                px = [p.x for p in pins]
                py = [p.y for p in pins]
                ax.scatter(px, py, color=colors[net_id], s=100, edgecolors='black', zorder=3, label=f'Net {net_id}')

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        plt.title("Orthogonal Circuit Routing (PathFinder Algorithm)")
        plt.show()


def main():
    # --- Execution Example ---

    # 1. Setup Grid
    router = Router(width=50, height=50)

    # 2. Add Components (x, y, w, h)
    # Aligned to grid, no rotation logic needed as per prompt (0, 90, etc is handled by bbox placement)
    c1 = Component("CPU", 10, 10, 8, 8)
    c2 = Component("MEM", 30, 10, 8, 8)
    c3 = Component("IO", 20, 30, 6, 6)
    c4 = Component("PWR", 5, 35, 5, 5)

    router.add_component(c1)
    router.add_component(c2)
    router.add_component(c3)
    router.add_component(c4)

    # 3. Add Nets (Pin Locations)
    # Net 1: Connecting CPU to MEM (2 pins)
    router.add_net([(18, 14), (30, 14)])

    # Net 2: Connecting CPU to IO to PWR (3 pins - Multi-pin routing)
    # Pins at edges of components
    router.add_net([(14, 18), (23, 30), (7, 35)])

    # Net 3: Crossing Net 1 to test Overlap Resolution (CPU bottom to far right)
    # This forces a route that might conflict with Net 1 or Net 2
    router.add_net([(14, 10), (10, 11), (40, 5)])

    router.add_net([(5, 39), (40, 10)])

    # 4. Run Routing
    router.route()

    # 5. Show Result
    router.visualize()


if __name__ == '__main__':
    main()
