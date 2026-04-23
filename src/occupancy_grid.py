import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np


def get_cells_on_ray(x0, y0, x1, y1, grid_size):
    """Bresenham's line algorithm to find all cells traversed by a ray."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = int(x0), int(y0)
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    for _ in range(int(n)):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            cells.append((x, y))
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return cells


def create_lidar_grid_svg():
    # Grid parameters
    grid_size = 16

    # Define grid state (0: gray, 1: white, 2: black)
    # Start with all gray
    grid_state = np.zeros((grid_size, grid_size), dtype=int)

    # Car and Sensor parameters
    # Set the car's bounding box roughly on the left
    car_x_start, car_y_start = 2, 7.5
    car_x_width, car_y_height = 3, 2
    sensor_pos = (car_x_start + car_x_width, car_y_start + car_y_height / 2)  # front center

    # Obstacle points and their corresponding cells
    # Coordinates of red points. We'll use these to mark occupied cells.
    occupied_points = [
        (13.5, 11.5), (12.5, 10.5), (11.5, 9.5), (11.5, 8.5), (11.5, 7.5),
         (10.5, 6.5), (11.5, 5.5), (11.5, 4.5), (11.5, 3.5),
        (10.5, 2.5), (9.5, 2.5)
    ]

    # Trace rays to calculate free and occupied cells
    for pt in occupied_points:
        # Trace from sensor to the center of the occupied cell
        ray_cells = get_cells_on_ray(sensor_pos[0], sensor_pos[1], pt[0], pt[1], grid_size)

        # Mark all traversed cells (before the endpoint) as free (white, 1)
        # Note: numpy uses [row, col] so use [cell_y, cell_x]
        for i, (cell_x, cell_y) in enumerate(ray_cells[:-1]):
            # Only mark as free if it is not already black (occupied)
            if grid_state[cell_y, cell_x] != 2:
                grid_state[cell_y, cell_x] = 1

        # Mark the last cell (containing the endpoint) as occupied (black, 2)
        # It's better to explicitly find the cell containing the point
        last_cell_x, last_cell_y = int(pt[0]), int(pt[1])
        if 0 <= last_cell_x < grid_size and 0 <= last_cell_y < grid_size:
            grid_state[last_cell_y, last_cell_x] = 2

    # Figure and custom colormap setup
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: 0=gray, 1=white, 2=black
    custom_cmap = ListedColormap(['#AAAAAA', '#FFFFFF', '#000000'])

    # Plot the grid using imshow with custom colors
    # origin='lower' makes the bottom-left coordinate (0,0)
    ax.imshow(grid_state, cmap=custom_cmap, origin='lower', extent=[0, grid_size, 0, grid_size],
              interpolation='nearest', zorder=1)

    # Add grid lines with no labels
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(which='both', color='#444444', linewidth=0.5, zorder=2)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Plot lidar rays and points
    for pt in occupied_points:
        ax.plot([sensor_pos[0], pt[0]], [sensor_pos[1], pt[1]], color='red', linewidth=1, zorder=3)
        ax.scatter(pt[0], pt[1], color='red', s=10, zorder=4)

    # Plot Car - geometric approximation
    # Base body (silver)
    car_base_coords = [
        (car_x_start, car_y_start),
        (car_x_start + car_x_width, car_y_start),
        (car_x_start + car_x_width + 0.5, car_y_start + car_y_height / 2),  # pointy front
        (car_x_start + car_x_width, car_y_start + car_y_height),
        (car_x_start, car_y_start + car_y_height),
    ]
    car_base = patches.Polygon(car_base_coords, facecolor='#DDDDDD', edgecolor='black', zorder=5)
    ax.add_patch(car_base)

    # Roof (black)
    car_roof = patches.Rectangle(
        (car_x_start + 0.5, car_y_start + car_y_height / 4),
        car_x_width - 1, car_y_height / 2, facecolor='black', edgecolor='black', zorder=6
    )
    ax.add_patch(car_roof)

    # Windows (light grey, transparent)
    windshield = patches.Rectangle(
        (car_x_start + 0.5, car_y_start + car_y_height / 4),
        car_x_width - 1, car_y_height / 2, facecolor='lightgrey', edgecolor='black', alpha=0.5, zorder=7
    )
    ax.add_patch(windshield)

    # Main plot settings
    ax.set_aspect('equal')
    ax.set_frame_on(False)  # remove figure frame

    # Create a separate axis for the legend to place it exactly where we want
    # Subplots_adjust makes room on the right
    fig.subplots_adjust(right=0.75)

    # Add an axis in normal units (0 to 1) for the legend
    legend_ax = fig.add_axes([0.75, 0.45, 0.2, 0.2], frameon=False)
    legend_ax.axis('off')

    # Data for legend
    legend_colors = ['black', 'white', '#AAAAAA']
    legend_labels = [
        r'$p_{z_{k+1}}(O_{k+1}|z_{k+1}) = 0.95$',
        r'$p_{z_{k+1}}(O_{k+1}|z_{k+1}) = 0.05$',
        r'$p_{z_{k+1}}(O_{k+1}|z_{k+1}) = 0.5$'
    ]

    # Draw squares and mathematical text
    for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
        y_pos = 1 - (i + 0.5) / 3  # distribute vertically
        rect = patches.Rectangle((0, y_pos - 0.05), 0.1, 0.1, facecolor=color, edgecolor='black',
                                 transform=legend_ax.transAxes)
        legend_ax.add_patch(rect)

        # Use mathtext (supported by Matplotlib without LaTeX installed) for formulas
        legend_ax.text(0.15, y_pos, label, transform=legend_ax.transAxes, verticalalignment='center', fontsize=12)

    # Save as an SVG for vector output
    plt.savefig('lidar_grid.svg', bbox_inches='tight' )
    print('SVG figure saved to lidar_grid.svg')
    plt.close(fig)  # free up memory


create_lidar_grid_svg()