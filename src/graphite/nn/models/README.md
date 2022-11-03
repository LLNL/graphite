# Model-specific documentation

Model- and framework-specific documentation is detailed here.


## NequIP

Here is an example of the model initialization:

```python
from graphite.nn.models.e3nn_nequip import NequIP
# or
from graphite.nn.models.e3nn_nequip_nonperiodic import NequIP_nonperiodic

model = NequIP_nonperiodic(
    irreps_in      = '16x0e',
    irreps_node    = '16x0e',
    irreps_hidden  = '16x0e + 16x1e',
    irreps_edge    = '1x0e + 1x1e + 1x2e',
    irreps_out     = '1x1e'
    num_convs      = 3,
    radial_neurons = [16, 64],
    num_species    = 1,
    max_radius     = 3.0,
    num_neighbors  = 12,
)
```

The important parameters are typically `irreps_hidden`, `irreps_edge`, and `irreps_out`. In NequIP, directional/tensorial information comes from the bond directions in the form of spherical harmonics, the orders of which are specified by `irreps_edge`, into the hidden node features, the orders of which are specified by `irreps_hidden`. Then, what is `irreps_node` for? It is another node feature embedding used for *self interactions* and *SkipInit*. These `irreps_node`-related mechanisms are optional and are absent in the original NequIP paper (I think), but have appeared in a more recent NequIP variant.

For predicting per-particle forces, the output is naturally a vector quantity, hence `irreps_out = '1x1e'`. If you want to output an energy value and a force vector, you can set `irreps_out = '1x0e + 1x1e'`. And since force field prediction probably doesn't require high-order tensor information (l = 3, 4 or higher), l = 2 or less should suffice, hence `irreps_hidden = '16x0e + 16x1e'` and `irreps_edge = '1x0e + 1x1e + 1x2e'`. On the other hand, for classifying phases such as BCC, FCC, and HCP, using high-order tensor information (e.g., l = 4, 6) would be ideal.


## ALIGNN/ALIGNN-d

### Graph construction

To use ALIGNN/ALIGNN-d, the data format has to be special. You need to make use of the custom `AngularGraphPairData` PyG `Data` class that holds a pair of two graphs: one graph for regular atomic structure (atom and bonds) and the other for bond/dihedral angles.

For clarity, here is the ALIGNN/ALIGNN-d graph hierarchy:
- Normal graph (denoted by `G`)
    - Each node represents an atom
    - Each edge represents a bond
- Angular graph (denoted by `A`)
    - Bond angle subgraph (in graph theory, this is actually the *line graph*)
        - Each node represents a bond
            - No additional node features. The edges from `G` and the nodes from `A` share the same embedding.
        - Each edge represents a bond angle
    - Dihedral angle subgraph (I call this the *dihedral graph*)
        - Each node represents a bond
            - No additional node features. The edges from `G` and the nodes from `A` share the same embedding.
        - Each edge represents a dihedral angle
        - If this dihedral graph is missing, then you just have the ALIGNN graph encoding

Here is an illustration of what the ALIGNN/ALIGNN-d graph looks like:
![Illustration of ALIGNN/ALIGNN-d graphs](/media/alignn-and-alignn-d.png "ALGINN and ALIGNN-d")

The example below constructs the ALIGNN/ALIGNN-d graph representation, encoding atom types and bond distances into the normal graph; and bond angles and optionally dihedral angles into the angular graph.

```python
from graphite.graph import *
from graphite.data import AngularGraphPairData

# Construct normal graph G
edge_index_G, x_atm, x_bnd = atoms2graph(atoms, cutoff=3.0, edge_dist=True)

# Construct angular graph A
edge_index_A_bnd_ang = line_graph(edge_index_G)
edge_index_A_dih_ang = dihedral_graph(edge_index_G)
x_bnd_ang    = get_bnd_angs(atoms, edge_index_G, edge_index_A_bnd_ang)
x_dih_ang    = get_dih_angs(atoms, edge_index_G, edge_index_A_dih_ang)
x_ang        = np.concatenate([x_bnd_ang, x_dih_ang])
edge_index_A = np.hstack([edge_index_A_bnd_ang, edge_index_A_dih_ang])
mask_dih_ang = [False]*len(x_bnd_ang) + [True]*len(x_dih_ang)

# Store everything into the custom `AngularGraphPairData` data class
data = AngularGraphPairData(
    edge_index_G = torch.tensor(edge_index_G, dtype=torch.long),
    x_atm        = torch.tensor(x_atm,        dtype=torch.long),
    x_bnd        = torch.tensor(x_bnd,        dtype=torch.float),
    edge_index_A = torch.tensor(edge_index_A, dtype=torch.long),
    x_ang        = torch.tensor(x_ang,        dtype=torch.float),
    mask_dih_ang = torch.tensor(mask_dih_ang, dtype=torch.bool),
)
```

Note that for ALIGNN representation (without the dihedral angles), omit the parts related to dihedral angles. Thus, `edge_index_A` is simply the output of `line_graph(...)`, `x_ang` is the output of `get_bnd_ang(...)`, and `mask_dih_ang` is `None`.


### Running ALIGNN-d

The following example demonstrates a forward pass of the ALIGNN-d model.

```python
from graphite.nn.models import ALIGNN_d
from torch_geometric.data import DataLoader

# Prepare dataloader
dataset = ...  # A list of `AngularGraphPairData` objects
follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(dataset[0], 'x_ang') else ['x_atm']
loader = DataLoader(dataset, batch_size=32, shuffle=True, follow_batch=follow_batch)

# Prepare model
model = ALIGNN_d(dim=100, num_interactions=6, num_species=3, cutoff=3.0)
model.eval()

# Run a forward pass
data = next(iter(loader))
pred = model(data)
```

