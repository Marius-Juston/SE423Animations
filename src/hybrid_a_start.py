import math
import heapq

# --- Configuration & Parameters ---
GRID_SIZE = 20
CELL_SIZE = 30  # Pixels per grid cell for SVG
SVG_WIDTH = GRID_SIZE * CELL_SIZE
SVG_HEIGHT = GRID_SIZE * CELL_SIZE

# Kinematic parameters
V = 1.0  # Forward velocity (m/s)
DT = 0.1  # Integration time step (s)
ARC_LENGTH = 2  # Total length of each expansion arc (m)
STEPS = int(ARC_LENGTH / (V * DT))  # Number of integration steps per arc
W_MAX = 1.2  # Max angular velocity (rad/s)
N_STEERS = 5  # Number of discrete steering angles

# Start and Goal poses (x, y, theta)
START = (2.5, 2.5, 0.0)
GOAL = (17.5, 17.5, 0.0)


def normalize_theta(theta):
    """Normalize angle to the range [-pi, pi]."""
    while theta > math.pi: theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def get_grid_key(x, y, theta):
    """
    Discretize the continuous state for the closed set.
    Resolution: 1x1 grid cells, 15-degree angular bins.
    """
    grid_x = int(math.floor(x))
    grid_y = int(math.floor(y))
    grid_theta = int(round(normalize_theta(theta) / (math.pi / 12)))
    return (grid_x, grid_y, grid_theta)


def heuristic(x1, y1, x2, y2):
    """Euclidean distance heuristic."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def generate_obstacles():
    """Create a basic map with a few walls."""
    obstacles = set()
    # Central block
    for r in range(8, 13):
        for c in range(8, 12):
            obstacles.add((c, r))
    # Wall near goal
    for c in range(12, 18):
        obstacles.add((c, 15))
    return obstacles


def is_collision_free(x, y, obstacles):
    """Check grid bounds and obstacles."""
    gx, gy = int(math.floor(x)), int(math.floor(y))
    if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
        return False
    if (gx, gy) in obstacles:
        return False
    return True


def hybrid_a_star(start, goal, obstacles):
    """Execute Hybrid A* and return the final path and all explored arcs."""
    # Generate discrete steering commands
    steers = [-W_MAX + (2 * W_MAX * i) / (N_STEERS - 1) for i in range(N_STEERS)]

    start_node = {
        'x': start[0], 'y': start[1], 'theta': start[2],
        'g': 0, 'f': heuristic(start[0], start[1], goal[0], goal[1]),
        'path': [(start[0], start[1], start[2])],
        'parent': None
    }

    open_set = []
    # Use an incrementing ID to prevent heapq from comparing dictionaries during ties
    heapq.heappush(open_set, (start_node['f'], 0, start_node))
    node_id_counter = 1

    closed_set = {}
    start_key = get_grid_key(start[0], start[1], start[2])
    closed_set[start_key] = 0

    explored_arcs = []

    while open_set:
        _, _, current = heapq.heappop(open_set)

        # Check if we are within the goal tolerance
        if heuristic(current['x'], current['y'], goal[0], goal[1]) < 1.0:
            return current['path'], explored_arcs

        for w in steers:
            nx, ny, ntheta = current['x'], current['y'], current['theta']
            arc = [(nx, ny, ntheta)]
            valid = True

            # Forward Euler Integration
            for _ in range(STEPS):
                nx += V * math.cos(ntheta) * DT
                ny += V * math.sin(ntheta) * DT
                ntheta = normalize_theta(ntheta + w * DT)

                if not is_collision_free(nx, ny, obstacles):
                    valid = False
                    break
                arc.append((nx, ny, ntheta))

            if not valid:
                continue

            explored_arcs.append(arc)
            end_state = arc[-1]

            # Continuous action cost + slight penalty for turning
            cost_step = ARC_LENGTH + abs(w) * 0.3
            g_new = current['g'] + cost_step

            grid_key = get_grid_key(end_state[0], end_state[1], end_state[2])

            if grid_key not in closed_set or g_new < closed_set[grid_key]:
                closed_set[grid_key] = g_new
                f_new = g_new + heuristic(end_state[0], end_state[1], goal[0], goal[1])

                new_node = {
                    'x': end_state[0], 'y': end_state[1], 'theta': end_state[2],
                    'g': g_new, 'f': f_new,
                    'path': current['path'] + arc[1:],
                    'parent': current
                }
                heapq.heappush(open_set, (f_new, node_id_counter, new_node))
                node_id_counter += 1

    return None, explored_arcs  # Path not found


def generate_svg(obstacles, explored_arcs, final_path, filename="hybrid_astar.svg"):
    """Render the environment and trajectories to an SVG file."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '  ',
        '  <g stroke="#1a2535" stroke-width="0.5">'
    ]

    # Draw Grid
    for i in range(GRID_SIZE + 1):
        lines.append(f'    <line x1="{i * CELL_SIZE}" y1="0" x2="{i * CELL_SIZE}" y2="{SVG_HEIGHT}" />')
        lines.append(f'    <line x1="0" y1="{i * CELL_SIZE}" x2="{SVG_WIDTH}" y2="{i * CELL_SIZE}" />')
    lines.append('  </g>')

    # Draw Obstacles
    lines.append('  ')
    for (cx, cy) in obstacles:
        lines.append(
            f'  <rect x="{cx * CELL_SIZE}" y="{cy * CELL_SIZE}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="#3a2535" />')

    # Draw Explored Arcs
    lines.append('  ')
    lines.append('  <g fill="none" stroke="rgba(168,123,239,0.5)" stroke-width="1">')
    for arc in explored_arcs:
        d = f"M {arc[0][0] * CELL_SIZE} {arc[0][1] * CELL_SIZE} "
        for pt in arc[1:]:
            d += f"L {pt[0] * CELL_SIZE} {pt[1] * CELL_SIZE} "
        lines.append(f'    <path d="{d}"/>')
    lines.append('  </g>')

    # Draw Final Path
    if final_path:
        lines.append('  ')
        d = f"M {final_path[0][0] * CELL_SIZE} {final_path[0][1] * CELL_SIZE} "
        for pt in final_path[1:]:
            d += f"L {pt[0] * CELL_SIZE} {pt[1] * CELL_SIZE} "
        lines.append(f'  <path d="{d}" fill="none" stroke="#e85454" stroke-width="3" stroke-dasharray="6,4" />')

    # Draw Start and Goal
    def draw_pose(x, y, theta, color, label):
        px, py = x * CELL_SIZE, y * CELL_SIZE
        al = CELL_SIZE * 0.7
        ax, ay = px + math.cos(theta) * al, py + math.sin(theta) * al
        return f'''
  <g>
    <circle cx="{px}" cy="{py}" r="{CELL_SIZE * 0.3}" fill="{color}" />
    <text x="{px}" y="{py + 4}" font-family="monospace" font-size="12" font-weight="bold" fill="#0b0e14" text-anchor="middle">{label}</text>
    <line x1="{px}" y1="{py}" x2="{ax}" y2="{ay}" stroke="{color}" stroke-width="2" />
  </g>'''

    lines.append('  ')
    lines.append(draw_pose(START[0], START[1], START[2], "#45c878", "S"))
    lines.append(draw_pose(GOAL[0], GOAL[1], GOAL[2], "#e8924a", "G"))

    lines.append('</svg>')

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"SVG saved to {filename}")


if __name__ == "__main__":
    obstacles = generate_obstacles()
    print("Running Hybrid A*...")
    final_path, explored_arcs = hybrid_a_star(START, GOAL, obstacles)

    if final_path:
        print(f"Path found! Generating SVG with {len(explored_arcs)} explored arcs...")
    else:
        print("No path found! Generating SVG with explored regions anyway...")

    generate_svg(obstacles, explored_arcs, final_path)