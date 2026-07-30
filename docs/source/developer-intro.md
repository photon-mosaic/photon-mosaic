# For Developers

# Installation

```bash
git clone https://github.com/photon-mosaic/photon-mosaic.git
cd photon-mosaic
pip install -e .
```

Currently, photon-mosaic parallelization depends on an open PR in SpikeInterface ([#4216](https://github.com/SpikeInterface/spikeinterface/pull/4216)). This is installed automatically when installing from source as shown above.


BELOW VERY MUCH DRAFTS



## The `photon-mosaic` mental model for multi-photon time series and its high-level implementation

This section provides an introduction to the high-level software architecture of the `photon-mosaic` API, for software developers.


### Prerequisites

Initial familiarity with
- object-oriented software development in Python
- parallelisation and chunk-based approaches
- multi-photon times series imaging

### `Imaging` and `Epoch` objects

`photon-mosaic` takes an object-oriented approach to represent multi-photon time series. Two types of object are central to this: `Imaging` objects represent two-photon time series that are (approximately) contiguous in space. `Epochs` are two-photon time series that are (exactly) contiguous in space _and_ channel _and_ time.

`Imaging` objects typically contain at least one `Epoch`. For example, let's say someone acquires a volumetric timeseries of a certain brain region. The acquisition is repeated daily, on the same animal and region, for three days. In this case, each day's timeseries would be two epochs (one per channel) containing all planes, and all the data would live in the same `Imaging` object, which would contain six epochs (3 days by 2 channels).

[TODO An illustration of the example might be nice here?]

Different brain regions would be kept in separate `Imaging` objects. Sometimes, adhering to this mental model requires splitting or merging of input data. `photon-mosaic` provides functionality for this.

The base classes for `Imaging` and `Epoch`s are `BaseImaging` and `BaseEpoch`, respectively.

`Imaging` objects have the additional responsibility for keeping track of metadata around the operations applied to them. They do this via updating their internal dictionary `self._kwargs`.

### Providing data on-demand

Imaging objects provide data

### Reading time series from external file formats

We rely on the [`roiextractors` package](https://roiextractors.readthedocs.io/en/latest/index.html).
In the `extractors` module, we dynamically wrap `roiextractor` readers to support reading a wide range of input files. In accordance with the mental model above, we ensure we return `Imaging` objects with appropriate metadata, containing `Epoch`s that are always return four-dimensional arrays (time, height, width, planes), lazily.

### Writing and reading natively from disk

#### When to write to file

### Analyzers and Extensions

### Chunking approaches

How to handle parallelisation

### What about "Session"?

We consider this too ambiguous/experiment-dependent term, so we avoid using it. Users will decide terminology around this as

### Analogy to `spikeinterface`

For people familiar with the `spikeinterface` code base, Imaging` objects can be thought of as analogous to `Recording` objects in `spikeinterface`, while `Epoch`s can be thought of as analogous to `Segment`s.

### Exposing a functional API to users

Outward-facingly, we prefer to expose functions rather than classes to users and libraries depending on `photon-mosaic`. This is because we think this is more elegant and user-friendly.

Pattern

```python
class DoSomething:

do_something = DoSomething
```

For external collection of classes we might want to wrap, this can be done dynamically, using the Python standard library's `inspect` module.

### Advantages of the design

-
- tried and tested through `spikeinterface`, allows code reuse.
