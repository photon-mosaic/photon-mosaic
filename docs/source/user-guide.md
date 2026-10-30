# Usage

## Installation

We currently support Python versions 3.12, 3.13, and 3.14. There is currently no release on PyPI yet, but you can install `photon-mosaic` from Github.
We strongly recommend that you use an environment manager (e.g. `conda` from [Miniforge](https://conda-forge.org/download/)). In your terminal, you can then run:

```bash
conda create -n photon-mosaic
conda activate photon-mosaic
pip install git+https://github.com/photon-mosaic/photon-mosaic@dev
```

### Optional dependencies

Some features need extra packages. Install them through these extras:

| Extra | Needed for |
|---|---|
| `widgets` | interactive widget functionality |
| `suite2p` | suite2p functionality (default for many steps) |
| `deconvolution` | spike deconvolution with OASIS |

Extras can be installed by adding the name of the extra in square brackets to installations command, e.g.

```bash
uv pip install photon-mosaic[suite2p]
```

The example notebooks need `widgets`, `suite2p` and `deconvolution`, so you would run

```bash
pip install "photon-mosaic[widgets,suite2p,deconvolution] @ git+https://github.com/photon-mosaic/photon-mosaic@dev"
```

## Example notebooks

`photon-mosaic` is a Python API, so it is meant to be called from other Python code, such as user scripts, notebooks or downstream Python packages. We provide example notebooks showing typical applications of multi-photon imaging analysis in [the examples folder of the `photon-mosaic` GitHub repository](https://github.com/photon-mosaic/photon-mosaic/tree/dev/examples).
