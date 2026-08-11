# For Developers

## Developer installation

Developers can clone `photon-mosaic` from GitHub

```bash
git clone https://github.com/photon-mosaic/photon-mosaic.git
cd photon-mosaic
```

and install with their favourite environment management tool (for example, [the `uv` tool](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## The `photon-mosaic` architecture

This section provides an introduction to the high-level software architecture of the `photon-mosaic` API, for software developers.

### Prerequisites

Initial familiarity with
- object-oriented software development
- lazy evaluation of operations on chunked arrays
- multi-photon times series analysis

### Multi-photon time series analysis

On a conceptual level, analysing multi-photon time series typically means performing a series of image processing steps resulting in a number of regions of interest (ROIs). From each ROI, we then extract its (one-dimensional) temporal signal ("trace"), which can be processed further.

:::{note}
While we can still _conceptually_ think of analysis as a linear sequence of steps, some algorithms simultaneously segment the ROIs and extract their temporal signal, e.g. by minimising a function that depends on both.
:::

For image processing, `photon-mosaic` defines its own data structures to represent multi-photon time series (`Imaging` and `Epoch`) and ROIs (classes derived from `BaseROIs`). Signal processing steps are encoded in `AnalyzerExtensions` which are wrapped into an `Analyzer` class.

We provide notebooks demonstrating how we envision usage of `photon-mosaic` in [our examples folder on GitHub](https://github.com/photon-mosaic/photon-mosaic/tree/dev/examples).

:::{admonition} Analogy to `spikeinterface`
:class: note

For people familiar with the [`spikeinterface` project](https://spikeinterface.readthedocs.io/en/stable/), `Imaging` objects can be thought of as analogous to `Recording` objects in `spikeinterface`, while `Epoch`s can be thought of as analogous to `Segment`s. `Analyzer` and `AnalyzerExtension` classes have [equivalents of the same name](https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html#) in `spikeinterface`.
:::

### Module scopes

The `photon-mosaic` Python package has a [number of submodules](./api_index.rst) with distinct scope:

| Module | Scope |
| --- | --- |
| Core |  Base class definitions, analyzers, core utility functions |
| Extractors | Classes that allow extraction of `photon-mosaic` objects from third party tools |
| Preprocessing | Motion correction and other pre-processing steps (return `Imaging` objects) |
| Sample Data | Provision of public example data for manual testing and examples |
| Segmentation | ROI segmentation algorithms (take `Imaging`, return `ROIs`) |
| Widgets | Interactive IPython widgets for plotting in notebooks |

Note that each submodule has its own `tests/` folder for unit tests.

### The mental model for `Imaging` and `Epoch` objects

`photon-mosaic` takes an object-oriented approach to represent multi-photon time series. Two types of object are central to this: `Imaging` objects represent a collection of  multi-photon time series that are (approximately) contiguous in space. `Epochs` are contained within `Imaging` objects and represent multi-photon time series that are (exactly) contiguous in space _and_ channel _and_ time. This concept is visualised below:

![](_static/single-plane-single-epoch.png)
![](_static/multi-plane-single-epoch.png)

Single- and multi-plane time series are stored in the same epoch...

![](_static/single-plane-single-epoch.png)
![](_static/single-plane-single-epoch.png)

... but different channels, and time series that were interrupted (e.g. consecutive days) belong in different `Epoch`s, but the same `Imaging` object (assuming they were taken from the same anatomical location).

For example, let's say someone acquires a 2-channel time series of a certain brain region. The acquisition is repeated daily, on the same animal and region, for three days. In this case, each day's time series would be two epochs (one per channel) containing all planes, and all the data would live in the same `Imaging` object, which would contain six epochs (3 days by 2 channels).

Time series of different brain regions should be kept in separate `Imaging` objects. Sometimes, adhering to this mental model requires splitting or merging of input data. `photon-mosaic` provides functionality for this.

Beyond grouping `Epoch`s, `Imaging` objects have the additional responsibility for keeping track of metadata around the operations applied to them. They do this via updating their internal dictionary `self._kwargs`.
This enables both provenance tracking and lazy evaluation. Only when calling `.get_series()` on an Imaging object while the stored chain of operations be executed and the requested array data be loaded into memory as needed.
The array data request is delegated to the `Epoch` object(s) in the `Imaging`, but users are expect to mostly interact with `Imaging` objects.

### `ROIs` objects

The [`segmentation` module](./api/segmentation.rst) provides functionality that spatially segments `Imaging` objects and returns a `ROIs` object.

#### `Analyzer` and `AnalyzerExtension` objects

`Analyzer` objects combine `Imaging` and `ROIs`. They provide a `compute` function that takes the name of an `AnalyzerExtension` as a string (and possibly more arguments). `AnalyzerExtension`s may depend on each other (e.g. the `df_over_f` extensions depends on the `fluorescence_trace` extension), and calling `compute` on an extension that depends on another extension that is not yet computed will raise an error.

### Reading and writing data

The [`extractors` module](./api/extractors.rst) reads data written by external software (e.g. ScanImage acquisitions or Suite2p ROIs). We rely on the [`roiextractors` package](https://roiextractors.readthedocs.io/en/latest/index.html), which we dynamically wrap, to support reading a wide range of raw input time series. In accordance with the mental model above, we ensure we return `Imaging` objects with appropriate metadata, containing `Epoch`s that are always four-dimensional arrays (time, height, width, planes), lazily.

We support reading and writing `Imaging` and `ROIs` object natively to `binary` and `zarr` formats.

### Exposing a functional API to users

Outward-facingly, we prefer to expose functions rather than classes to users and libraries depending on `photon-mosaic`. This is because we think this is more elegant and user-friendly.

Pattern

```python
class DoSomething:

do_something = DoSomething
```

For external collections of classes we might want to wrap, this can be done dynamically, using the Python standard library's `inspect` module.
Examples of this include:

* [(static) `read_binary_imaging/BinaryFolderImaging` ](https://github.com/photon-mosaic/photon-mosaic/blob/43cfc029f16639b7a9a8258207e5798803c3486f/src/photon_mosaic/core/binaryimaging.py) (line 257)

* [(dynamic) extractor classes for `roiextractors`](https://github.com/photon-mosaic/photon-mosaic/blob/43cfc029f16639b7a9a8258207e5798803c3486f/src/photon_mosaic/extractors/roiextractors.py) (line 55)
*
