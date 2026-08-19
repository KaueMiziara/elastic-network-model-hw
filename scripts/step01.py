import numpy as np
import matplotlib.pyplot as plt

coordinates = []

"""
PDB Structure:
- It was designed to represent molecules in the 70s punchcards
- Relevant column rules:
  - Columns 1-6: Record type (we are interested in ATOM)
  - Columns 13-16: Atom name (we want the Carbon-Alpha, CA)
  - Columns 31-38: X coordinate in Ångströms
  - Columns 39-46: Y coordinate in Ångströms
  - Columns 47-54: Z coordinate in Ångströms

"""

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

print(f"Extracted {ca_matrix.shape[0]} C-alpha atoms.")
print(f"Array shape: {ca_matrix.shape}")  # 1AKI: (129, 3)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

x_coords = ca_matrix[:, 0]
y_coords = ca_matrix[:, 1]
z_coords = ca_matrix[:, 2]

ax.scatter(
    x_coords,
    y_coords,
    z_coords,
    c="teal",
    marker="o",
    s=50,
    alpha=0.8,
)

ax.plot(
    x_coords,
    y_coords,
    z_coords,
    color="gray",
    linewidth=1,
    alpha=0.5,
)

ax.set_title("1AKI: C-Alpha Backbone Point Cloud", fontsize=14)
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

plt.show()
