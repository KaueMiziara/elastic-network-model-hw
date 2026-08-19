import numpy as np
import matplotlib.pyplot as plt

# Step 1

coordinates = []

with open("data/1AKI.pdb", "r") as file:
    for line in file:
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()

            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                coordinates.append([x, y, z])

ca_matrix = np.array(coordinates)

# Step 2

diff = ca_matrix[:, np.newaxis, :] - ca_matrix[np.newaxis, :, :]

sq_diff = diff**2
sum_sq_diff = np.sum(sq_diff, axis=2)
distance_matrix = np.sqrt(sum_sq_diff)

# Step 3

"""
To build the Elastic Network, we apply a distance cutoff (e.g., $d \\leq 8$ Å).
- If the distance between i and j is less than the threshold, we have a connection.

Mathematically, this is equivalent to applying a binary mask or a threshold to a tensor in DL.
- The result is an adjacency matrix/contact map.

To calculate the physics later, we need to transform the map into a Kirchhoff Matrix.

The Kirchhoff/Hessian Matrix ($\\Gamma$) is defined such that:
- $\\Gamma_{ij}, \forall i \neq j$: If $d_{ij} \\leq 8$, -1; otherwise, 0
- $\\Gamma_{ii}$: Represents the degree (number of connections);
  $$\\Gamma_{ii} = \\sum_{j\neq i} | \\Gamma_{ij} |$$
"""

cutoff = 8.0

contact_map = (distance_matrix <= cutoff).astype(int)

np.fill_diagonal(contact_map, 0)

kirchhoff = -1 * contact_map

degrees = np.sum(contact_map, axis=1)

np.fill_diagonal(kirchhoff, degrees)

print(f"Kirchhoff matrix shape: {kirchhoff.shape}")
print(f"Sample diagonal (connections per atom): {degrees[:5]}")

x_coords = ca_matrix[:, 0]
y_coords = ca_matrix[:, 1]
z_coords = ca_matrix[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    x_coords,
    y_coords,
    z_coords,
    c="teal",
    marker="o",
    s=30,
    alpha=0.8,
)

# Indices where a spring exists (contact_map == 1)
sources, targets = np.nonzero(contact_map)

for src, tgt in zip(sources, targets):
    if src < tgt:
        ax.plot(
            [x_coords[src], x_coords[tgt]],
            [y_coords[src], y_coords[tgt]],
            [z_coords[src], z_coords[tgt]],
            color="gray",
            alpha=0.3,
            linewidth=0.5,
        )

ax.set_title(f"1AKI: 3D Elastic Network Model (Cutoff: {cutoff}Å)", fontsize=14)
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

ax.grid(False)

plt.show()
