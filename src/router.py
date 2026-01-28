import math

import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

# --- Configuration ---
GRID_SCALE = 1  # Resolution of the routing grid
BEND_PENALTY = 50  # Cost added for making a 90-degree turn
BASE_COST = 1  # Cost to move 1 unit
MAX_ITERATIONS = 20  # Max rip-up and reroute attempts


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

    def add_component(self, comp: Component):
        self.components.append(comp)

        comp_scale = comp + 0.1
        comp_int_bounding = comp_scale.int_bbox

        # Mark component area as obstacle (high base occupancy)
        # In a real scenario, you might allow routing OVER components on different layers,
        # but here we treat them as blockages except for their pins.
        for i in range(comp_int_bounding[0], comp_int_bounding[2]):
            for j in range(comp_int_bounding[1], comp_int_bounding[3]):
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.grid_occupancy[i, j] += 1000  # High initial cost for component bodies

    def add_net(self, pins: List[Tuple[int, int]]):
        # Convert tuples to Points
        self.nets.append([Point(x, y) for x, y in pins])

    def _get_neighbors(self, current: Point):
        neighbors = []
        # Directions: Right, Left, Up, Down
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

    def _astar(self, start: Point, end: Point, congestion_multiplier: float):
        """
        A* Search with Bend Penalty and Congestion Costs.
        State: (cost, x, y, last_dx, last_dy)
        """
        # Priority Queue: (f_score, g_score, current_point, last_direction)
        pq = []
        heapq.heappush(pq, (0, 0, start, Point(0, 0)))

        came_from = {}
        g_scores = {(start, Point(0, 0)): 0}

        best_path_cost = float('inf')
        best_end_state = None

        while pq:
            f, g, current, last_dir = heapq.heappop(pq)

            if g > best_path_cost:
                continue

            if current == end:
                if g < best_path_cost:
                    best_path_cost = g
                    best_end_state = (current, last_dir)
                continue

            for neighbor in self._get_neighbors(current):
                # Calculate movement vector
                new_dir = neighbor - current

                # --- COST CALCULATION ---

                # 1. Base distance cost
                step_cost = BASE_COST

                # 2. Bend Penalty
                if last_dir != Point(0, 0) and new_dir != last_dir:
                    step_cost += BEND_PENALTY

                # 3. Congestion Cost (The PathFinder Magic)
                # Cost = Base + (Occupancy * Multiplier) + History
                # We subtract 1 from occupancy because the net itself counts as 1,
                # but self-overlap shouldn't be penalized during its own routing phase.
                occ = max(0, self.grid_occupancy[neighbor.x, neighbor.y])
                hist = self.history_cost[neighbor.x, neighbor.y]

                # If occupancy > 0 (meaning another net is here), cost shoots up
                congestion_cost = (occ * congestion_multiplier) + hist

                # Note: We relax the component body blockages for the pin entry/exit points
                # by assuming pins sit ON the boundary. Ideally, we check if neighbor is a pin.

                total_step_cost = step_cost + congestion_cost
                new_g = g + total_step_cost

                if new_g < g_scores.get((neighbor, new_dir), float('inf')):
                    g_scores[(neighbor, new_dir)] = new_g
                    # Heuristic: Manhattan Distance
                    h = abs(neighbor.x - end.x) + abs(neighbor.y - end.y)
                    heapq.heappush(pq, (new_g + h, new_g, neighbor, new_dir))
                    came_from[(neighbor, new_dir)] = (current, last_dir)

        if best_end_state:
            # Reconstruct path
            path = []
            curr, direction = best_end_state
            while curr != start:
                path.append(curr)
                prev_node, prev_dir = came_from[(curr, direction)]
                curr = prev_node
                direction = prev_dir
            path.append(start)
            return path[::-1]

        return None

    def route(self):
        """
        Executes the PathFinder iterative routing algorithm.
        """
        congestion_multiplier = 0.5  # Starts low, increases every iteration

        print(f"Starting routing for {len(self.nets)} nets...")

        for iteration in range(MAX_ITERATIONS):
            print(f"Iteration {iteration + 1}...")
            max_overlap = 0

            # 1. Rip-up and Reroute all nets
            for net_id, pins in enumerate(self.nets):

                # A. Rip-up: Remove old path from occupancy grid
                if net_id in self.routed_paths:
                    for p in self.routed_paths[net_id]:
                        self.grid_occupancy[p.x, p.y] -= 1

                # B. Decompose Multi-pin net to 2-pin segments
                segments = self._decompose_multipins(pins)

                full_net_path = []

                # C. Route each segment
                for start, end in segments:
                    # Note: Ideally we route on the graph of the partially routed net,
                    # but simple segment routing works for small N.
                    path = self._astar(start, end, congestion_multiplier)
                    if path:
                        # Don't duplicate points where segments join
                        if full_net_path and path[0] == full_net_path[-1]:
                            full_net_path.extend(path[1:])
                        else:
                            full_net_path.extend(path)

                self.routed_paths[net_id] = full_net_path

                # D. Update Occupancy
                for p in full_net_path:
                    self.grid_occupancy[p.x, p.y] += 1

            # 2. Check Congestion & Update History
            overlaps = np.where(self.grid_occupancy > 1)
            overlap_count = len(overlaps[0])
            print(f"  -> Overlaps found: {overlap_count}")

            if overlap_count == 0:
                print("  -> Solution found with 0 overlaps.")
                break

            # 3. Increase Penalties for next round
            # Increase history cost for currently congested nodes
            # This "poisons" the node so nets try to avoid it permanently
            for x, y in zip(*overlaps):
                if self.grid_occupancy[x, y] > 100:  # Ignore component bodies
                    continue
                self.history_cost[x, y] += 1 * (iteration + 1)

            # Increase the immediate cost of conflict
            congestion_multiplier *= 1.5

    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw Grid (Optional, makes it look like graph paper)
        ax.grid(True, which='both', color='lightgrey', linestyle='--', alpha=0.5)

        # Draw Components
        for comp in self.components:
            rect = patches.Rectangle((comp.x, comp.y), comp.width, comp.height,
                                     linewidth=2, edgecolor='black', facecolor='lightgray', zorder=2)
            ax.add_patch(rect)
            ax.text(comp.x + comp.width / 2, comp.y + comp.height / 2, comp.name,
                    ha='center', va='center', weight='bold')

        # Draw Routes
        colors = plt.cm.jet(np.linspace(0, 1, len(self.nets)))

        for net_id, path in self.routed_paths.items():
            if not path: continue

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
    router.add_net([(14, 10), (40, 5)])

    # 4. Run Routing
    router.route()

    # 5. Show Result
    router.visualize()

if __name__ == '__main__':
    main()