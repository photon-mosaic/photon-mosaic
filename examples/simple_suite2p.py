"""
Simple ``suite2p`` example
==========================

In this notebook, we showcase a very simple example of using ``photon-mosaic`` to run ``suite2p``, and plot results.
"""

# %% [markdown]
# # Simple `suite2p` example
#
# In this notebook, we showcase a very simple example of using `photon-mosaic` to run a full analysis pipeline, and plot result at each. Note that this is a toy example to showcase functionality: its result is intentionally far from perfect, illustrating how users must evaluate and optimise their settings in order to get good results form their data. This is expected.
#
# We start by importing some `photon-mosaic` functions that will come in handy, and set up an interactive matplotlib backend.

# %%
import matplotlib
from photon_mosaic.preprocessing import compute_motion, register_motion
from photon_mosaic.sample_data import download_suite2p_google_drive
from photon_mosaic.segmentation.suite2p_segmentation import Suite2pSegmentationSettings, detect_rois_suite2p
from photon_mosaic.widgets.rois import plot_rois
from photon_mosaic.widgets.series import plot_imaging_series

matplotlib.use("module://ipympl.backend_nbagg")


# %% [markdown]
# The first step is to load the example data. We use `suite2p`'s example data, on just one file with 200 frames. `photon-mosaic` provides a function that downloads and caches it this data from `suite2p`s Google drive, and loads it into an `Imaging` object. We print some information about this `Imaging` object.

# %%
raw_imaging = download_suite2p_google_drive(num_files=1)

print(raw_imaging)

# %% [markdown]
# Printing shows us metadata about the imaging, for example its frame rate and its number of epochs.
# We can also plot it. If you are running this notebook locally or

plot_imaging_series(raw_imaging, backend="ipywidgets", colormap="plasma")

# %% [markdown]
# Next, we compute the motion correction and apply it. Again, we print information and plot the registered data alongside the raw.

# %%
motion = compute_motion(imaging=raw_imaging, method="suite2p")
registered_imaging = register_motion(imaging=raw_imaging, motion=motion)

print(registered_imaging)
plot_imaging_series(
    {"raw": raw_imaging, "motion_corrected": registered_imaging}, backend="ipywidgets", colormap="plasma"
)

# %% [markdown]
# Next, we use `suite2p` with its default settings to detect regions of interest (ROIs) in the registered image.

# %%
detected_rois = detect_rois_suite2p(
    registered_imaging,
    settings=Suite2pSegmentationSettings(algorithm="sourcery"),
)

print(detect_rois_suite2p)
plot_rois(detected_rois)  # type: ignore[arg-type]


# %% [markdown]
# TODO: Finally, we extract signals from the detected ROIs.
