# graphite

A repository for implementing graph network models based on atomic structures.


## Implemented or replicated works

- ALIGNN
    - [*Atomistic Line Graph Neural Network for improved materials property predictions*][ALIGNN paper]
- **ALIGNN-d** (our work, see [demo](notebooks/alignn/demo.ipynb))
    - [*Efficient and interpretable graph network representation for angle-dependent properties applied to optical spectroscopy*][ALIGNN-d paper]
- Gated GCN
    - [*Benchmarking Graph Neural Networks*][Gated GCN paper]
- NequIP (code implementation adopted from `e3nn`)
    - [*E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials*][NequIP paper]
- **Atomic Structure denoiser** (our work, see [demo](notebooks/denoiser/demo.ipynb))
    - [*Score-based denoising for atomic structure identification*][Denoiser paper]
- MeshGraphNets
    - [*Learning Mesh-Based Simulation with Graph Networks*][MGN paper]
- **Score dynamics** (our work, see [demo](notebooks/score-dynamics/demo.ipynb))


## Gallery

### Score dynamics

Generative rollouts of molecular dynamics over picosecond timesteps via conditional diffusion model.

![](/media/score-dynamics/ala-dipep.gif)


### Atomic structure denoiser

Simple and effective atomic denoiser for structure characterization.

![](/media/denoiser/denoising-fcc-2d.gif)
![](/media/denoiser/denoising-fcc-3d.gif)


## Installation

The installation time is typically less than 10 minutes on a normal local machine.

Installation dependencies:
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for graph data format processing and batching.
- [Optional] Atomic Simulation Environment (`ase`): for reading/writing atomic structures and efficient neighbor list algorithm.
- [Optional] Euclidean neural networks (`e3nn>=0.4.4`): dependency for the NequIP models.

An example for the installation process:
```bash
conda create -n graphite
conda activate graphite
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg

# Optional install dependency, but required for some of the model implementations
pip install ase e3nn

# Other useful packages for development (optional)
pip install jupyterlab ipywidgets seaborn lightning tensorboard MDAnalysis
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

`graphite` is intended to be a general collection of codes (e.g., helper functions, custom graph convolutions, and template graph models) for research purposes. Production codes for certain applications and deployments should be hosted elsewhere.

- The `src` folder contains the source code.
- The `notebooks` folder contains Jupyter notebooks that demonstrate running or training models.
    - Some demos require additional packages (e.g., PyTorch Lightning for automated training). Please see Installation and the instructions in the demos.


## Release

LLNL-CODE-836648



[ALIGNN paper]: https://www.nature.com/articles/s41524-021-00650-1
[ALIGNN-d paper]: https://www.nature.com/articles/s41524-022-00841-4
[Gated GCN paper]: https://arxiv.org/abs/2003.00982
[e3nn basic conv doc]: https://docs.e3nn.org/en/stable/guide/convolution.html
[NequIP paper]: https://www.nature.com/articles/s41467-022-29939-5
[e3nn transformer doc]: https://docs.e3nn.org/en/stable/guide/transformer.html
[PyG dataset doc]: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
[Denoiser paper]: https://arxiv.org/abs/2212.02421
[MGN paper]: https://arxiv.org/abs/2010.03409v4

