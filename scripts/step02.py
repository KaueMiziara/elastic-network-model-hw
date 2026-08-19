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

"""
Now we need to translate the coordinates into a matrix.
This matrix will show the relationship between every single atom.

For this, we calculate the Euclidian Distance between every pair of atmos (i, j):
$$d_{ij} = \\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}$$

Due to the 129 atoms, using standard `for` loops in Python is expensive.
Instead, we can use NumPy broadcasting:
- mathematically similar to manipulating spatial dimensions in a DL tensor transform

By adding "dummy" axes, we can perform operations on all possible pairs simultaneously.
"""

"""
1 - broadcasting the spatial dimensions
- ca_matrix shape is (129, 3); we expand to (129, 1, 3) and (1, 129, 3)
- substracting them creates a (129, 129, 3) tensor of coordinate differences
"""
diff = ca_matrix[:, np.newaxis, :] - ca_matrix[np.newaxis, :, :]

# 2 - Apply the Euclidian Distance
sq_diff = diff**2

sum_sq_diff = np.sum(sq_diff, axis=2)  # sum along the coordinate axis -> (129, 129)

distance_matrix = np.sqrt(sum_sq_diff)

print(f"Distance matrix shape: {distance_matrix.shape}")  # (129, 129)

# 3. Visualize the Distance Matrix
fig, ax = plt.subplots(figsize=(7, 7))

# Plot the matrix. The diagonal will be 0 (an atom is 0 distance from itself)
cax = ax.imshow(distance_matrix, cmap="viridis_r")

cbar = fig.colorbar(cax, shrink=0.8)
cbar.set_label("Distance in Ångströms", rotation=270, labelpad=15)

ax.set_title("1AKI: Distance Matrix Heatmap")
ax.set_xlabel("Atom Index (j)")
ax.set_ylabel("Atom Index (i)")

plt.show()
