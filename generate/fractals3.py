import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

def generate_fractal(num_edges=5, recursion_depth=4, seed_shape=None, seed_color=None):
    """Generates a fractal pattern based on the Miyashita et al. algorithm."""

    if seed_shape is None:
        seed_shape = random.randint(2, 8)  # Random number of edges (3-6)
    if seed_color is None:
        seed_color = random.random() #Random color between 0 and 1

    # Initialize polygon vertices
    vertices = []
    for i in range(num_edges):
        angle = 2 * np.pi * i / num_edges
        vertices.append((np.cos(angle), np.sin(angle)))

    def deflect(verts, depth):
        if depth == 0:
            return verts
        new_verts = []
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]  # Wrap around to the first vertex

            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1
            theta = np.arctan2(dy, dx)  # Use arctan2 for correct quadrant

            GA = random.uniform(-0.5, 0.5) # Random deflection amplitude

            new_x = midpoint_x + GA * np.sin(theta)
            new_y = midpoint_y - GA * np.cos(theta)

            new_verts.append((new_x, new_y))

        return deflect(new_verts, depth - 1)

    final_vertices = deflect(vertices, recursion_depth)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')  # Ensure circular shapes appear as circles
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')

    x = [v[0] for v in final_vertices]
    y = [v[1] for v in final_vertices]
    plt.plot(x + [x[0]], y + [y[0]], color=plt.cm.viridis(seed_color), linewidth=2) #Use viridis colormap

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fractal patterns.")
    parser.add_argument("--edges", type=str, default="3-6", help="Number of edges (e.g., 3-6 for a range)")
    parser.add_argument("--depth", type=int, default=2, help="Recursion depth")
    args = parser.parse_args()

    # Parse edge argument
    if "-" in args.edges:
        min_edges, max_edges = map(int, args.edges.split("-"))
        num_edges = random.randint(min_edges, max_edges)
    else:
        num_edges = int(args.edges)

    generate_fractal(num_edges=num_edges, recursion_depth=args.depth)
