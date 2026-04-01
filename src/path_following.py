import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from queue import PriorityQueue


# ==========================================
# 1. Standard 2D A* Planner (Unchanged)
# ==========================================
def a_star_2d(start, goal, grid):
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
            return path[::-1]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
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
# 2. Non-Linear Lyapunov Controller (Unchanged)
# ==========================================
class DifferentialDriveController:
    def __init__(self, Kx, Ky, Ktheta, v_max, omega_max):
        self.Kx, self.Ky, self.Ktheta = Kx, Ky, Ktheta
        self.v_max, self.omega_max = v_max, omega_max

    def compute_commands(self, state, target_x, target_y, target_theta, v_ref):
        x, y, theta = state
        dx, dy = target_x - x, target_y - y
        dtheta = math.atan2(math.sin(target_theta - theta), math.cos(target_theta - theta))
        ex = math.cos(theta) * dx + math.sin(theta) * dy
        ey = -math.sin(theta) * dx + math.cos(theta) * dy
        v_cmd = v_ref * math.cos(dtheta) + self.Kx * ex
        omega_cmd = v_ref * (self.Ky * ey + self.Ktheta * math.sin(dtheta))
        v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)
        omega_cmd = np.clip(omega_cmd, -self.omega_max, self.omega_max)
        return v_cmd, omega_cmd


# ==========================================
# 3. Animation Logic
# ==========================================
def save_simulation_gif():
    # Environment Setup
    grid = np.zeros((20, 20))
    grid[5:15, 10] = 1

    grid[16:19, 10] = 1
    start_idx, goal_idx = (2, 2), (18, 18)
    path_points = np.array(a_star_2d(start_idx, goal_idx, grid)) * 1.0

    # Simulation Variables
    dt = 0.1
    state = np.array([path_points[0][0], path_points[0][1], 0.0])
    controller = DifferentialDriveController(Kx=1.5, Ky=5.0, Ktheta=2.5, v_max=2.0, omega_max=1.0)
    v_ref = 0.8

    trajectory_x, trajectory_y = [], []
    sim_data = {'idx': 1, 'state': state}

    # Plot Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.T, cmap='Greys', origin='lower', alpha=0.3)
    ax.plot(path_points[:, 0], path_points[:, 1], 'r--', label="A* Path")

    line_traj, = ax.plot([], [], 'b-', label="Robot Trajectory")
    robot_dot, = ax.plot([], [], 'go', markersize=8)
    target_dot, = ax.plot([], [], 'ro', markersize=8, label="Current Target")
    heading_line, = ax.plot([], [], 'g-', linewidth=2)

    fig.tight_layout()

    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 20)
    ax.legend(loc='upper left')

    def update(frame):
        if sim_data['idx'] >= len(path_points):
            print(frame)
            return line_traj, robot_dot, target_dot, heading_line

        # Current State & Target
        curr_state = sim_data['state']
        target = path_points[sim_data['idx']]
        prev_target = path_points[sim_data['idx'] - 1]
        target_theta = math.atan2(target[1] - prev_target[1], target[0] - prev_target[0])

        # Control & Integration
        v, omega = controller.compute_commands(curr_state, target[0], target[1], target_theta, v_ref)
        curr_state[0] += v * math.cos(curr_state[2]) * dt
        curr_state[1] += v * math.sin(curr_state[2]) * dt
        curr_state[2] += omega * dt

        trajectory_x.append(curr_state[0])
        trajectory_y.append(curr_state[1])

        # Waypoint logic
        if math.hypot(target[0] - curr_state[0], target[1] - curr_state[1]) < 0.55:
            sim_data['idx'] += 1

        # Update Plot Objects
        line_traj.set_data(trajectory_x, trajectory_y)
        robot_dot.set_data([curr_state[0]], [curr_state[1]])
        target_dot.set_data([target[0]], [target[1]])
        heading_line.set_data([curr_state[0], curr_state[0] + math.cos(curr_state[2])],
                              [curr_state[1], curr_state[1] + math.sin(curr_state[2])])

        return line_traj, robot_dot, target_dot, heading_line

    # Create Animation
    # frames=200 is a safe estimate, repeat=False prevents looping in the processing
    ani = FuncAnimation(fig, update, frames=150, blit=True, interval=50, repeat=True)

    print("Saving animation to robot_path.gif...")
    writer = PillowWriter(fps=20)
    ani.save("robot_path.gif", writer=writer, dpi=300)
    print("Done!")
    plt.close(fig)


if __name__ == '__main__':
    save_simulation_gif()
