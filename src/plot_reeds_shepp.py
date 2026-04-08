import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the specific Enums and functions from the local reeds_shepp.py
import reeds_shepp
from reeds_shepp import Steering, Gear


def draw_vehicle_patch(ax, x, y, theta_rad, width=1.4, length=2.8):
    """
    Renders a vehicle bounding box at configuration q = (x, y, theta_rad)
    """
    # Define local footprint coordinates relative to the vehicle's center
    dx = -length / 2
    dy = -width / 2

    # Generate the rigid body patch
    car = patches.Rectangle(
        (dx, dy), length, width,
        angle=math.degrees(theta_rad),
        rotation_point='center',
        linewidth=1.2, edgecolor='#1a1a1a', facecolor='#e6e6e6', zorder=3
    )

    # Apply SE(2) affine transformation to global coordinates
    transform = plt.matplotlib.transforms.Affine2D().translate(x, y)
    car.set_transform(transform + ax.transData)
    ax.add_patch(car)


def _integrate_segment(points, length, steering, gear, rho, step_size):
    """
    Numerically integrates a single PathElement to generate waypoints.
    """
    current_q = points[-1]
    x, y, theta = current_q[0], current_q[1], current_q[2]

    num_steps = int(length / step_size)

    for _ in range(num_steps):
        # Kinematic state transition
        if steering == Steering.LEFT:
            d_theta = (step_size / rho) * gear.value
        elif steering == Steering.RIGHT:
            d_theta = -(step_size / rho) * gear.value
        else:
            d_theta = 0.0

        x += step_size * gear.value * math.cos(theta)
        y += step_size * gear.value * math.sin(theta)
        theta += d_theta

        points.append((x, y, theta))

    # Ensure the exact end of the segment is reached (handling floating point remainders)
    remainder = length % step_size
    if remainder > 1e-6:
        if steering == Steering.LEFT:
            d_theta = (remainder / rho) * gear.value
        elif steering == Steering.RIGHT:
            d_theta = -(remainder / rho) * gear.value
        else:
            d_theta = 0.0

        x += remainder * gear.value * math.cos(theta)
        y += remainder * gear.value * math.sin(theta)
        theta += d_theta

        points.append((x, y, theta))

    return points


def sample_path(start_node, end_node, rho, step_size=0.05):
    """
    Computes the optimal path and discretizes it into SE(2) waypoints.
    Expects angles in degrees to interface with the library, returns radians.
    """
    # 1. Scale boundary conditions to compute paths for a unit turning radius
    q0_scaled = (start_node[0] / rho, start_node[1] / rho, start_node[2])
    q1_scaled = (end_node[0] / rho, end_node[1] / rho, end_node[2])

    # 2. Solve the boundary value problem using the analytical functions
    path_elements = reeds_shepp.get_optimal_path(q0_scaled, q1_scaled)

    # 3. Integrate the path elements to generate discrete waypoints
    # Start point requires conversion to radians for integration math
    start_rad = (start_node[0], start_node[1], math.radians(start_node[2]))
    points = [start_rad]

    for element in path_elements:
        # Scale the segment length back to the true radius
        segment_length = element.param * rho
        points = _integrate_segment(points, segment_length, element.steering, element.gear, rho, step_size)

    return points


def main():
    # Define initial and terminal SE(2) state configurations: q = (x, y, theta_in_degrees)
    # The nathanlct library specifically expects angles in degrees
    q0 = (0.0, 0.0, -45.0)
    q1 = (3.5, 2.2, -135.0)

    turning_radius = 2.5

    # Generate the discretized kinematic trajectory
    trajectory = sample_path(q0, q1, turning_radius)
    xs = [q[0] for q in trajectory]
    ys = [q[1] for q in trajectory]

    # Initialize plot topology
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the trajectory
    ax.plot(xs, ys, color='black', linestyle='-', linewidth=1.5, zorder=2)

    # Overlay initial and terminal vehicle geometries
    # Convert degrees to radians explicitly for the drawing function
    draw_vehicle_patch(ax, q0[0], q0[1], math.radians(q0[2]), width=0.5, length=0.75)
    draw_vehicle_patch(ax, q1[0], q1[1], math.radians(q1[2]), width=0.5, length=0.75)

    # Enforce strictly equal aspect ratio to preserve geometric truth
    ax.set_aspect('equal')

    # Strip all plot axes, spines, and ticks for clean export
    ax.axis('off')

    # Export with alpha channel transparency
    output_filename = 'reeds_shepp_trajectory_clean.svg'
    plt.savefig(output_filename, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Successfully generated transparent geometry patch: {output_filename}")


if __name__ == '__main__':
    main()