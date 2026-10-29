"""Run a minimal Suite2p analysis workflow."""

# %%
import matplotlib
from photon_mosaic.preprocessing import compute_motion, register_motion
from photon_mosaic.sample_data import download_suite2p_google_drive
from photon_mosaic.segmentation.suite2p_segmentation import Suite2pSegmentationSettings, detect_rois_suite2p
from photon_mosaic.widgets.rois import plot_rois
from photon_mosaic.widgets.series import plot_imaging_series

matplotlib.use("module://ipympl.backend_nbagg")

# %% [markdown]
# # Simple `suite2p` example
#
# In this notebook, we showcase a very simple example of using `photon-mosaic` to run `suite2p`, and plot results. We will run `suite2p` on its own single-plane example data (via a `photon_mosaic.sample_data` function that downloads and caches it from their Google drive), on just one file with 200 frames. Note that this is a toy example to showcase functionality: its result is intentionally far from perfect, illustrating how users must evaluate and optimise their settings in order to get good results form their data. This is expected.
# %% [markdown]
# We start by importing the function that provides the sample data as an `Imaging` object. We then load the data, print some information about it, and plot it.
# %%
first_imaging = download_suite2p_google_drive(1)

plot_imaging_series(first_imaging, backend="ipywidgets", colormap="plasma")

# %% [markdown]
# Next, we compute the motion correction and apply it. Again, we print information and plot the registered data alongside the raw.
# %%
motion = compute_motion(imaging=first_imaging, method="suite2p")
registered_imaging = register_motion(imaging=first_imaging, motion=motion)

print(registered_imaging)
plot_imaging_series(
    {"raw": first_imaging, "motion_corrected": registered_imaging}, backend="ipywidgets", colormap="plasma"
)

# %% [markdown]
# Next, we use `suite2p` with its default settings to detect regions of interest (ROIs) in the registered image.
# %%
detected_rois = detect_rois_suite2p(
    registered_imaging,
    settings=Suite2pSegmentationSettings(algorithm="sourcery"),
)

print(detect_rois_suite2p)
plot_rois(detected_rois)


# %% [markdown]
# TODO: Finally, we extract signals from the detected ROIs.
