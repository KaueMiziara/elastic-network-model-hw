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

cutoff = 8.0

contact_map = (distance_matrix <= cutoff).astype(int)

np.fill_diagonal(contact_map, 0)

kirchhoff = -1 * contact_map

degrees = np.sum(contact_map, axis=1)

np.fill_diagonal(kirchhoff, degrees)

# Step 4

"""
We'll perform Normal Mode Analysis (NMA) using the Gaussian Network Model (GNM).

To extract the physical movements from the network, we decompose the Kirchhoff matrix. 
- Mathematically, it's identical to finding the principal components in a DL dataset to reduce dimensionality: 
  - we want to find the dominant axes of variance. 
- In physics, these axes are the natural vibrations of the protein.
  - We do this using Eigendecomposition.

When we decompose the Kirchhoff matrix ($K$), we get two things:
- Eigenvalues ($\\lambda$): represent the stiffness or frequency of a specific motion. 
  - Lower values mean large, slow, global movements. 
  - Higher values mean fast, stiff, localized vibrations.
- Eigenvectors ($V$): These are the actual "modes" or shapes of the movement.

The mean-square fluctuation ($\\Delta R_i^2$) of each atom are, essentially, how flexible that part of the protein is
 - it's inversely proportional to the eigenvalues.
 - We sum up the contributions of all the modes to get the total flexibility (often called theoretical B-factors):

$$\\Delta R_i^2 \\propto \\sum_{k=2}^{N} \frac{V_{ik}^2}{\\lambda_k}$$

- We start the sum at $k=2$ because the first eigenvalue is always 0
  - (the entire protein translating in space without changing shape)
"""

eigenvalues, eigenvectors = np.linalg.eigh(kirchhoff)

idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

non_zero_evals = eigenvalues[1:]
non_zero_evecs = eigenvectors[:, 1:]

fluctuations = np.sum((non_zero_evecs**2) / non_zero_evals, axis=1)

print(f"Calculated fluctuations for {len(fluctuations)} atoms.")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    range(1, len(fluctuations) + 1),
    fluctuations,
    color="coral",
    linewidth=2,
    marker="o",
    markersize=3,
)

ax.set_title(
    "1AKI: Theoretical Atom Fluctuations (Gaussian Network Model)", fontsize=14
)
ax.set_xlabel("Residue Index")
ax.set_ylabel("Mean Square Fluctuation (Theoretical)")
ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
