from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
import numpy as np

# Step 1


def extract_calpha_matrix(filepath: str, structure_id="prot"):
    """
    Parses a PDB or mmCIF file and returns a NumPy array of C-alpha coordinates.
    """
    if filepath.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif filepath.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format. Please provide .pdb or .cif")

    structure = parser.get_structure(structure_id, filepath)

    if structure is None:
        raise RuntimeError("Biopython parser returned a NoneType structure.")

    coordinates = []

    for residue in structure.get_residues():
        if residue.id[0] == " ":
            coords = residue["CA"].get_coord()
            coordinates.append(coords)

    if not coordinates:
        raise ValueError(
            f"Zero C-alpha atoms extracted from {filepath}. "
            "Check if the file is a valid PDB/CIF and not an HTML document."
        )

    return np.array(coordinates)


ca_matrix = extract_calpha_matrix("data/1AKI.cif")
print(f"Extracted from mmCIF: {ca_matrix.shape}")

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

eigenvalues, eigenvectors = np.linalg.eigh(kirchhoff)

idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

non_zero_evals = eigenvalues[1:]
non_zero_evecs = eigenvectors[:, 1:]

fluctuations = np.sum((non_zero_evecs**2) / non_zero_evals, axis=1)

print(f"Calculated fluctuations for {len(fluctuations)} atoms.")
