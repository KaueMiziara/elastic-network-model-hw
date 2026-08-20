import numpy as np
import networkx as nx
import plotly.graph_objects as go

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser


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


# Step 1

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

# Step 5

G = nx.from_numpy_array(contact_map)

for i in range(len(ca_matrix)):
    G.nodes[i]["pos"] = ca_matrix[i]

print(
    f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
)

edge_x, edge_y, edge_z = [], [], []
for edge in G.edges():
    x0, y0, z0 = G.nodes[edge[0]]["pos"]
    x1, y1, z1 = G.nodes[edge[1]]["pos"]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(color="rgba(150, 150, 150, 0.5)", width=2),
    hoverinfo="none",
)

node_x, node_y, node_z = [], [], []
for node in G.nodes():
    x, y, z = G.nodes[node]["pos"]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)

fluct_min, fluct_max = np.min(fluctuations), np.max(fluctuations)
normalized_fluct = (fluctuations - fluct_min) / (fluct_max - fluct_min)

node_trace = go.Scatter3d(
    x=node_x,
    y=node_y,
    z=node_z,
    mode="markers",
    marker=dict(
        size=8,
        color=normalized_fluct,
        colorscale="RdYlBu_r",
        colorbar=dict(
            title=dict(text="Flexibility", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
        opacity=0.9,
        line=dict(width=0.5, color="white"),
    ),
    text=[
        f"Residue {i + 1}<br>Flexibility: {fluct:.2f}"
        for i, fluct in zip(G.nodes(), fluctuations)
    ],
    hoverinfo="text",
)

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(color="rgba(150, 150, 150, 0.4)", width=1.5),
    hoverinfo="none",
)

fig = go.Figure(data=[edge_trace, node_trace])

invisible_axis = dict(
    showgrid=False,
    zeroline=False,
    showticklabels=False,
    showline=False,
    showbackground=False,
    title="",
)

fig.update_layout(
    title=dict(
        text="1AKI: Interactive Elastic Network (Colored by Flexibility)",
        font=dict(color="white"),
    ),
    showlegend=False,
    scene=dict(
        xaxis=invisible_axis,
        yaxis=invisible_axis,
        zaxis=invisible_axis,
        bgcolor="rgb(20, 24, 30)",
    ),
    paper_bgcolor="rgb(20, 24, 30)",
    margin=dict(l=0, r=0, b=0, t=40),
)

output_file = "plots/elastic_network_1AKI.html"
fig.write_html(output_file)
print(f"Success! Interactive 3D plot saved to: {output_file}")
print("Double-click this file in your file explorer to view it in your browser.")
