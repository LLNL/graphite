# graphite

A repository for implementing graph network models based on atomic structures.


## Installation

It's recommended that the following dependencies are first installed before installing `graphite`:
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`)
    - For implementing graph network operations and models.
- [Optional] Atomic Simulation Environment (`ase`)
    - For reading atomic structures and computing the neighbor list (used to generate graphs).
- [Optional] Euclidean neural networks (`e3nn>=0.4.4`)
    - For implementing *equivariant* networks via tensor products of irreducible representations.

For example:
```bash
conda create -n graphite python=3.9
conda activate graphite
conda install pytorch cudatoolkit=11.3 -c pytorch  # Assuming the CUDA version is 11.3
conda install pyg -c pyg
pip install ase
```

Then, to install `graphite`, clone this repo and run:
```bash
pip install -e /path/to/the/repo
```

The `-e` option signifies an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/), which is well suited for development; in many cases you can edit the source code without having to re-install.

To uninstall:
```bash
pip uninstall graphite
```

[Optional] Other useful packages to install for development:
```bash
pip install jupyterlab sklearn seaborn
```


## Implemented or replicated works

- ALIGNN
    - [*Atomistic Line Graph Neural Network for improved materials property predictions*][ALIGNN paper]
- **ALIGNN-d** (our work)
    - [*Efficient, Interpretable Graph Neural Network Representation for Angle-dependent Properties and its Application to Optical Spectroscopy*][ALIGNN-d paper]
- Edge-gated graph convolution
    - [*Benchmarking Graph Neural Networks*][Edge-gated conv paper]
- Simple equivariant convolution (code implementation adopted from `e3nn`)
    - [e3nn documentation][e3nn basic conv doc]
- NequIP (code implementation adopted from `e3nn`)
    - [*E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials*][NequIP paper]
- Equivariant self-attention (code implementation adopted from `e3nn`)
    - [e3nn documentation][e3nn transformer doc]
    - [*SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks*][SE(3)-transformer paper]


## Usage

`graphite` is meant to be a general toolbox of graph model codes (e.g., helper functions, custom graph convolutions, and template graph models) for atomic structures. Production codes for specific applications should be hosted elsewhere.

- The `graphite` folder contains the source code.
- The `scripts` folder contains template scripts for running or training models defined in `graphite/nn/models`.
    - These scripts may not be up-to-date with the latest changes made to the source code. But they provide more detailed examples of how to use the models defined in `graphite`.


### General examples

#### Convert atomic structures to PyG's `Data` object

Following PyG's format, a standard graph object can be expressed in terms of three variables: `edge_index`, `x_atm`, and `edge_attr`.

- `edge_index` is the graph connectivity in COO format.
- `x_atm` is the input features for the atoms. This can be an array of N x M, where N is the number of atoms and M is the number of channels. But in the example below, this array is N-dimensional, where each element is an integer denoting the atom type.
- `edge_attr` is the input features for the edges. This can be an array of E x C, where E is the number of edges and C is the number of channels. But in the example below, this array is E x 3, where each row is the edge/bond vector.

```python
import torch
import ase.io
from graphite.graph import atoms2graph
from torch_geometric.data import Data

atoms = ase.io.read('path/to/atoms.file')

# Build a graph based on a cutoff radius of 3.5 angstroms.
edge_index, x_atm, edge_attr = atoms2graph(atoms, cutoff=3.5)

# Construct the PyG `Data` object that combines the graph, the node features, and the edge features.
data = Data(
    x          = torch.tensor(x_atm),
    edge_index = torch.tensor(edge_index),
    edge_attr  = torch.tensor(edge_attr),
)
```


#### Save PyG graphs into a PyTorch .pt file

[As documented in the PyG website][PyG dataset doc], the simplest form of a PyG dataset is a plain Python list containing PyG `Data` objects (aka graphs). Below is a data processing example that reads an MD trajectory file, convert each snapshot into PyG `Data`, and save the snapshots into a PyTorch .pt file.

```python
import torch
import ase.io
from graphite.graph import atoms2graph
from torch_geometric.data import Data

traj = ase.io.read('path/to/trajectory.file', index=':')
graphs = [atoms2graph(atoms) for atoms in traj]
dataset = [
    Data(
        x          = torch.tensor(x),
        edge_index = torch.tensor(edge_index),
        edge_attr  = torch.tensor(edge_attr),
    )
    for x, edge_index, edge_attr in graphs
]
torch.save(dataset, 'path/to/savefile.pt')
```


### ALIGNN/ALIGNN-d examples

#### ALIGNN-d graph construction

The example below constructs the ALIGNN-d graph representation, encoding bond angles and dihedral angles into the line graph and the dihedral graph, respectively.

For clarity, here is the ALIGNN-d graph hierarchy:
- Normal graph
    - Node feature: atom type
    - Edge feature: bond length
- Angular graph
    - Bond angle graph, or line graph
        - (No node feature; each node holds the bond index from the normal graph)
        - Edge feature: bond angle
    - Dihedral angle graph, or dihedral graph
        - (No node feature; each node holds the bond index from the normal graph)
        - Edge feature: dihedral angle

```python
from graphite.graph import *
from graphite.data import LineGraphPairData

# Construct original graph G
edge_index_G, x_atm, x_bnd = atoms2graph(atoms, cutoff=3.0, edge_dist=True)

# Construct line/angular graph L
edge_index_L_bnd_ang = line_graph(edge_index_G)
edge_index_L_dih_ang = dihedral_graph(edge_index_G)

# Encode angular values
x_bnd_ang    = get_bnd_angs(atoms, edge_index_G, edge_index_L_bnd_ang)
x_dih_ang    = get_dih_angs(atoms, edge_index_G, edge_index_L_dih_ang)
x_ang        = np.concatenate([x_bnd_ang, x_dih_ang])
edge_index_L = np.hstack([edge_index_L_bnd_ang, edge_index_L_dih_ang])
mask_dih_ang = [False]*len(x_bnd_ang) + [True]*len(x_dih_ang)

# Store everything as a custom PyG object
data = LineGraphPairData(
    edge_index_G = torch.tensor(edge_index_G, dtype=torch.long),   # Required
    x_atm        = torch.tensor(x_atm,        dtype=torch.long),   # Required
    x_bnd        = torch.tensor(x_bnd,        dtype=torch.float),  # Required

    edge_index_L = torch.tensor(edge_index_L, dtype=torch.long),   # Optional angular information
    x_ang        = torch.tensor(x_ang,        dtype=torch.float),  # Optional angular information
    mask_dih_ang = torch.tensor(mask_dih_ang, dtype=torch.bool),   # Optional angular information
)
```

Note that for ALIGNN representation (without the dihedral angles), simply omit the parts related to dihedral angles. Thus, `edge_index_L` is simply the output of `line_graph(...)`, `x_ang` is the output of `get_bnd_ang(...)`, and `mask_dih_ang` is `None`.


#### Running ALIGNN-d model

The following example demonstrates a forward pass of the ALIGNN-d model.

```python
from graphite.nn.models import ALIGNN_d
from torch_geometric.data import DataLoader

# Prepare dataloader
dataset = ...  # Can simply be a list of `LineGraphPairData`
follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(dataset[0], 'x_ang') else ['x_atm']
loader = DataLoader(dataset, batch_size=32, shuffle=True, follow_batch=follow_batch)

# Prepare model
model = ALIGNN_d(dim=100, num_interactions=6, num_species=3, cutoff=3.0)
model.eval()

# Run a forward pass
data = next(iter(loader))
pred, _  = model(data)
```


[ALIGNN paper]: https://www.nature.com/articles/s41524-021-00650-1
[ALIGNN-d paper]: https://arxiv.org/abs/2109.11576
[Edge-gated conv paper]: https://arxiv.org/abs/2003.00982
[e3nn basic conv doc]: https://docs.e3nn.org/en/stable/guide/convolution.html
[NequIP paper]: https://arxiv.org/abs/2101.03164
[SE(3)-transformer paper]: https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html
[e3nn transformer doc]: https://docs.e3nn.org/en/stable/guide/transformer.html
[PyG dataset doc]: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html


## Release

LLNL-CODE-836648