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
