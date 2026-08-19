import numpy as np
import matplotlib.pyplot as plt

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

x_coords = ca_matrix[:, 0]
y_coords = ca_matrix[:, 1]
z_coords = ca_matrix[:, 2]
