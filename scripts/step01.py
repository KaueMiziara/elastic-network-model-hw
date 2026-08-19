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
