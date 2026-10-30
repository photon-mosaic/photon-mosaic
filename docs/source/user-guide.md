# Usage

## Installation

We currently support Python versions 3.12 and 3.13.
We recommend installing `photon-mosaic` into a virtual environment managed by [uv](https://docs.astral.sh/uv/getting-started/installation/). In your terminal, you can then run:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install photon-mosaic
```

### Optional dependencies

Some features need extra packages. Install them through these extras:

| Extra | Needed for |
|---|---|
| `widgets` | interactive widget functionality |
| `suite2p` | suite2p functionality (default for many steps) |
| `deconvolution` | spike deconvolution with OASIS |

Extras can be installed by adding the name of the extra in square brackets to the installation command, e.g.

```bash
uv pip install "photon-mosaic[suite2p]"
```

The example notebooks need `widgets`, `suite2p` and `deconvolution`, so you would run

```bash
uv pip install "photon-mosaic[widgets,suite2p,deconvolution]"
```

## Some useful concepts

`photon-mosaic` is built around two Python objects:  `Epoch` objects and  `Imaging` objects. Understanding these conceptually is important to use `photon-mosaic`:

* `Epoch` objects contain spatio-temporally contiguous multiphoton data.
* `Imaging` object group `Epoch` objects together, and keep track of metadata, including about the steps that have been applied to their `Epoch`s. They are used to represent data that is spatially in the same location, but may be acquired at different times or experimental conditions.

## Example notebooks

`photon-mosaic` is a Python API, so it is meant to be called from other Python code, such as user scripts, notebooks or downstream Python packages. We provide introductory notebooks showing typical applications of multi-photon imaging analysis in our {doc}`examples gallery <examples/index>`.

These and more complex examples can also be found in [the examples folder of the `photon-mosaic` GitHub repository](https://github.com/photon-mosaic/photon-mosaic/tree/dev/examples).
