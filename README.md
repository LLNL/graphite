# graphite

A repository for implementing graph network models based on atomic structures.


## Implemented or replicated works

- ALIGNN
    - [*Atomistic Line Graph Neural Network for improved materials property predictions*][ALIGNN paper]
- **ALIGNN-d** (our work)
    - [*Efficient and interpretable graph network representation for angle-dependent properties applied to optical spectroscopy*][ALIGNN-d paper]
- Edge-gated graph convolution
    - [*Benchmarking Graph Neural Networks*][Edge-gated conv paper]
- NequIP (code implementation adopted from `e3nn`)
    - [*E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials*][NequIP paper]


## Installation

The following dependencies need to be installed before installing `graphite`. The installation time is typically within 10 mins on a normal local machine.
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

## Optional but recommended
pip install ase e3nn

## Other useful packages to install for development (optional)
pip install jupyterlab tensorboard seaborn
```

Then, to install `graphite`, clone this repo and run:
```bash
pip install -e /path/to/the/repo
```

The `-e` option signifies an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/), which is well suited for development; this allows you to edit the source code without having to re-install.

To uninstall:
```bash
pip uninstall graphite
```


## How to use

`graphite` is intended to be a general toolbox of graph model codes (e.g., helper functions, custom graph convolutions, and template graph models) for atomic structures. Production codes for specific applications should be hosted elsewhere.

- The `src` folder contains the source code.
- The `notebooks` folder contains Jupyter notebooks as demonstrations for running or training models defined in `src/graphite/nn/models`. Please check out these notebooks if you are new to `graphite`.


### General examples

The general usage examples are shown below. For model-specific documentation, please check out the README at `src/graphite/nn/models`.

#### Convert atomic structures to PyG's `Data` object

Following PyG's format, a standard graph object can be expressed in terms of three variables: `edge_index`, `x_atm`, and `edge_attr`.

- `edge_index` is the graph connectivity in COO format.
- `x_atm` is the input features for the atoms. This can be an N x M array, where N is the number of atoms and M is the number of channels. But in most use cases here, this is just a N-dimensional vector, where each element is an integer denoting the atom type, and the integers should span {0, 1, 2, ...}.
- `edge_attr` is the input features for the edges. This can be an E x C array, where E is the number of edges and C is the number of channels. But in the example below, this is an E x 3 array, where each row is an edge/bond vector.

```python
import torch
import numpy as np
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

    # Optionally, include the atomic positions and the cell parameters (for periodic structures)
    pos        = torch.tensor(atoms.positions, dtype=torch.float),
    cell       = np.array(atoms.cell),
)
```

For non-periodic structures such as molecules, you can choose to store only the atomic positions (besides `x_atm`) and let PyG's `radius_graph` function compute `edge_index`. But if you go this route, the model you're using has to include `radius_graph` and other associated computations.

```python
import torch
import ase.io
from torch_geometric.data import Data

atoms = ase.io.read('path/to/atoms.file')

# Define your atom features
x_atm = ...

data = Data(
    x   = torch.tensor(x_atm, dtype=torch.long),
    pos = torch.tensor(atoms.positions, dtype=torch.float),
)
```


#### Save PyG graphs into a PyTorch .pt file

The simplest form of a PyG dataset is a plain Python list containing PyG `Data` objects (aka graphs). Below is a data processing example that reads an MD trajectory file, convert each snapshot into PyG `Data`, and save the snapshots into a PyTorch .pt file.

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


[ALIGNN paper]: https://www.nature.com/articles/s41524-021-00650-1
[ALIGNN-d paper]: https://www.nature.com/articles/s41524-022-00841-4
[Edge-gated conv paper]: https://arxiv.org/abs/2003.00982
[e3nn basic conv doc]: https://docs.e3nn.org/en/stable/guide/convolution.html
[NequIP paper]: https://arxiv.org/abs/2101.03164
[SE(3)-transformer paper]: https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html
[e3nn transformer doc]: https://docs.e3nn.org/en/stable/guide/transformer.html
[PyG dataset doc]: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html


## Release

LLNL-CODE-836648
