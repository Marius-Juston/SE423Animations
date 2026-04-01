import numpy as np
import math
import matplotlib.pyplot as plt
from queue import PriorityQueue


# ==========================================
# 1. Standard 2D A* Planner
# ==========================================
def a_star_2d(start, goal, grid):
    """Simple 2D grid-based A* pathfinding."""
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get start -> goal

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check bounds and obstacles (0 is free, 1 is obstacle)
            if (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1] and
                    grid[neighbor[0], neighbor[1]] == 0):

                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    open_set.put((f_score, neighbor))
                    came_from[neighbor] = current
    return []


# ==========================================
# 2. Non-Linear Lyapunov Controller
# ==========================================
class DifferentialDriveController:
    def __init__(self, Kx, Ky, Ktheta, v_max, omega_max):
        self.Kx = Kx
        self.Ky = Ky
        self.Ktheta = Ktheta
        self.v_max = v_max
        self.omega_max = omega_max

    def compute_commands(self, state, target_x, target_y, target_theta, v_ref, omega_ref=0.0):
        x, y, theta = state

        # 1. Global errors
        dx = target_x - x
        dy = target_y - y

        # Normalize angle error to [-pi, pi]
        dtheta = math.atan2(math.sin(target_theta - theta), math.cos(target_theta - theta))

        # 2. Transform to robot's local error frame (e)
        ex = math.cos(theta) * dx + math.sin(theta) * dy
        ey = -math.sin(theta) * dx + math.cos(theta) * dy
        etheta = dtheta

        # 3. Lyapunov Control Law
        v_cmd = v_ref * math.cos(etheta) + self.Kx * ex
        omega_cmd = omega_ref + v_ref * (self.Ky * ey + self.Ktheta * math.sin(etheta))

        # 4. Actuator Saturation limits
        v_cmd = max(-self.v_max, min(v_cmd, self.v_max))
        omega_cmd = max(-self.omega_max, min(omega_cmd, self.omega_max))

        return v_cmd, omega_cmd


# ==========================================
# 3. Kinematic Simulation Loop
# ==========================================
def simulate():
    # Environment Setup
    grid = np.zeros((20, 20))
    grid[5:15, 10] = 1  # Add an obstacle wall

    start_idx = (2, 2)
    goal_idx = (18, 18)

    # 1. Generate discrete (x,y) plan
    path_indices = a_star_2d(start_idx, goal_idx, grid)
    path_points = np.array(path_indices) * 1.0  # Scale up for continuous space

    # Robot Initialization
    dt = 0.1
    state = np.array([path_points[0][0], path_points[0][1], 0.0])  # [x, y, theta]

    # Controller Initialization
    controller = DifferentialDriveController(Kx=1.5, Ky=5.0, Ktheta=3.0, v_max=2.0, omega_max=1.0)
    v_ref = 0.8  # Constant forward reference velocity

    trajectory_x, trajectory_y = [], []
    current_target_idx = 1

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while current_target_idx < len(path_points):
        target = path_points[current_target_idx]

        # --- INFER THETA FROM 2D PATH ---
        prev_target = path_points[current_target_idx - 1]
        target_theta = math.atan2(target[1] - prev_target[1], target[0] - prev_target[0])

        # Calculate control commands
        v, omega = controller.compute_commands(state, target[0], target[1], target_theta, v_ref)

        # Update Robot Kinematics (Euler Integration)
        state[0] += v * math.cos(state[2]) * dt
        state[1] += v * math.sin(state[2]) * dt
        state[2] += omega * dt

        trajectory_x.append(state[0])
        trajectory_y.append(state[1])

        # Waypoint transition logic (if close enough, move to next point)
        dist_to_target = math.hypot(target[0] - state[0], target[1] - state[1])
        if dist_to_target < 0.55:
            current_target_idx += 1

        # Visualization
        ax.clear()
        ax.imshow(grid.T, cmap='Greys', origin='lower', alpha=0.3)
        ax.plot(path_points[:, 0], path_points[:, 1], 'r--', label="A* Path (Discrete)")
        ax.plot(trajectory_x, trajectory_y, 'b-', label="Robot Trajectory (Continuous)")
        ax.plot(state[0], state[1], 'go', markersize=8)

        ax.plot(target[0], target[1], 'ro', markersize=8, label="Target Position")

        # Plot robot heading vector
        ax.plot([state[0], state[0] + math.cos(state[2])],
                [state[1], state[1] + math.sin(state[2])], 'g-', linewidth=2)

        ax.set_xlim(-1, 20)
        ax.set_ylim(-1, 20)
        ax.legend()
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    simulate()