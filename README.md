# graphite

A repository for implementing graph network models based on atomic structures.


## Implemented or replicated works

- ALIGNN
    - [*Atomistic Line Graph Neural Network for improved materials property predictions*][ALIGNN paper]
- **ALIGNN-d** (our work)
    - [*Efficient and interpretable graph network representation for angle-dependent properties applied to optical spectroscopy*][ALIGNN-d paper]
- Edge-gated graph convolution
    - [*Benchmarking Graph Neural Networks*][edge-gated conv paper]
- NequIP (code implementation adopted from `e3nn`)
    - [*E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials*][NequIP paper]
- **Denoising NequIP** (our work, see demo [here](notebooks/denoiser/training-and-inference.ipynb))
    - [*An iterative unbiased geometric approach to identifying crystalline order and disorder via denoising score function model*][denoising paper]
- MeshGraphNet
    - [*Learning Mesh-Based Simulation with Graph Networks*][mgn_paper]


## Installation

The following dependencies need to be installed before installing `graphite`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for implementing graph network operations.
- [Optional] Atomic Simulation Environment (`ase`): for reading/writing atomic structures.
- [Optional] Euclidean neural networks (`e3nn>=0.4.4`): dependency for the NequIP models.

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
- The `notebooks` folder contains Jupyter notebooks as demonstrations for running or training models.


## Release

LLNL-CODE-836648



[ALIGNN paper]: https://www.nature.com/articles/s41524-021-00650-1
[ALIGNN-d paper]: https://www.nature.com/articles/s41524-022-00841-4
[edge-gated conv paper]: https://arxiv.org/abs/2003.00982
[e3nn basic conv doc]: https://docs.e3nn.org/en/stable/guide/convolution.html
[NequIP paper]: https://arxiv.org/abs/2101.03164
[SE(3)-transformer paper]: https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html
[e3nn transformer doc]: https://docs.e3nn.org/en/stable/guide/transformer.html
[PyG dataset doc]: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
[denoising paper]: https://arxiv.org/abs/2212.02421
[mgn_paper]: https://arxiv.org/abs/2010.03409v4


