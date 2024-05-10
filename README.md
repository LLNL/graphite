# graphite

A repository for implementing graph network models based on atomic structures.


## Implemented or replicated works

- ALIGNN
    - [*Atomistic Line Graph Neural Network for improved materials property predictions*][ALIGNN paper]
- **ALIGNN-d** (our work, see [demo](notebooks/alignn/demo.ipynb))
    - [*Efficient and interpretable graph network representation for angle-dependent properties applied to optical spectroscopy*][ALIGNN-d paper]
- Gated GCN
    - [*Benchmarking Graph Neural Networks*][Gated GCN paper]
- NequIP (implementation adopted from `e3nn`; requires `e3nn` installation)
    - [*E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials*][NequIP paper]
- **Atomic Structure denoiser** (our work, see [demo](notebooks/denoiser/demo.ipynb))
    - [*Score-based denoising for atomic structure identification*][Denoiser paper]
- MeshGraphNets
    - [*Learning Mesh-Based Simulation with Graph Networks*][MGN paper]
- **Score dynamics** (our work, see [demo](notebooks/score-dynamics/demo.ipynb))
    - [*Score dynamics: scaling molecular dynamics with picoseconds time steps via conditional diffusion model*][SD paper]
- **Spectroscopy-guided generation of amorphous structures** (our work, see [demo](notebooks/amorph-gen/amorph-gen.ipynb))
    - [*Spectroscopy-guided discovery of three-dimensional structures of disordered materials with diffusion models*][a-C paper]
- Equivariant transformer
    - [*Equivariant pretrained transformer for unified geometric learning on multi-domain 3D molecules*][ET paper]
- Graphormer
    - [*Towards predicting equilibrium distributions for molecular systems with deep learning*][Graphormer paper]


## Gallery

### Spectroscopy-guided amorphous material generation

Unconditional generation of amorphous carbons via diffusion model. Color is meant to help give you a sense of depth.

![](/media/amorph-gen/a-C-denoise-traj-d15.gif)

Conditional generation of amorphous carbons based on a given XANES spectrum.

![](/media/amorph-gen/a-C_generation-with_plot.gif)

Generation of multi-element, amorphous carbon nitrides.

![](/media/amorph-gen/a-C-N-denoise.gif)


### Score dynamics

Generative rollouts of molecular dynamics over picosecond timesteps via conditional diffusion model.

![](/media/score-dynamics/ala-dipep.gif)


### Atomic structure denoiser

Simple and effective atomic denoiser for structure characterization.

<img src="/media/denoiser/denoising-fcc-2d.gif" width="512">
<img src="/media/denoiser/denoising-fcc-3d.gif" width="512">


## Installation

The installation time is typically less than 10 minutes on a normal local machine.

Installation dependencies:
- `pytorch>=2.0.1`
- `torch_geometric`
- `torch-scatter`
- `torch-cluster`

Reasons: many model implementations in this repo are written mostly based on PyTorch, but some operations such as `scatter` and graph pooling are from PyTorch Geometric (PyG), which offers very optimized CUDA implementations. Additionally, most models in this repo treat atomic/molecular data as graphs following the PyG data format (`torch_geometric.data.Data`).

Also, this repo has scattering (via `Torch.scatter_reduce`) and clustering (e.g., `radius_graph` and `knn_graph`) codes such that `torch-scatter` and `torch-cluster` are not strictly required. Still, it is recommended to install `torch-scatter` and `torch-cluster` if you favor CUDA-optimized compute speed.

Lastly, for development purposes, you may want to install packages such as `ase`, `MDAnalysis`, `rdkit`, `lightning`, etc.

Example installation process:
```bash
conda create -n graphite
conda activate graphite
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# For development depending on your use case
pip install ase jupyterlab ipywidgets seaborn lightning tensorboard
```

Then, to install `graphite`, clone this repo and run:
```bash
pip install -e /path/to/the/repo
```

To uninstall:
```bash
pip uninstall graphite
```


## How to use

`graphite` is intended for research and prototyping. It is a general collection of simple codes containing helper functions, custom graph convolutions, model templates, and so on. Full-scale production codes for specific applications and deployments should be hosted elsewhere.

- The `src` folder contains the source code.
- The `notebooks` folder contains Jupyter notebooks that demonstrate running or training models.
    - Some demos require additional packages (e.g., PyTorch Lightning for automated training).
    - The demo notebooks are not always up-to-date. We will try to update the notebooks after every major change to the source code.
- The `media` folder contains media files.


## Release

LLNL-CODE-836648



[ALIGNN paper]: https://www.nature.com/articles/s41524-021-00650-1
[ALIGNN-d paper]: https://www.nature.com/articles/s41524-022-00841-4
[Gated GCN paper]: https://arxiv.org/abs/2003.00982
[e3nn basic conv doc]: https://docs.e3nn.org/en/stable/guide/convolution.html
[NequIP paper]: https://www.nature.com/articles/s41467-022-29939-5
[e3nn transformer doc]: https://docs.e3nn.org/en/stable/guide/transformer.html
[Denoiser paper]: https://arxiv.org/abs/2212.02421
[MGN paper]: https://arxiv.org/abs/2010.03409v4
[ET paper]: https://arxiv.org/abs/2402.12714
[Graphormer paper]: https://arxiv.org/abs/2306.05445
[SD paper]: https://pubs.acs.org/doi/10.1021/acs.jctc.3c01361
[a-C paper]: https://arxiv.org/abs/2312.05472

