"""General-purpose load function for photon-mosaic objects.

Inspired by ``spikeinterface.core.loading.load``, this module provides a
single ``load`` entry-point that auto-detects the object type and format.

Loadable objects
----------------
* ``RoiAnalyzer``  — from binary folder or zarr
* ``BaseImaging``  — delegates to ``spikeinterface.core.loading.load``
* ``BaseRois``     — from binary folder (``BinaryFolderRois``)
                     or zarr (``ZarrRois``)
"""

from pathlib import Path

from spikeinterface.core.core_tools import is_path_remote

_error_msg = (
    "{file_path} is not a recognised photon-mosaic folder. "
    "It should be the result of save() or save_as() on a "
    "RoiAnalyzer, BaseRois, or BaseImaging object."
)


def load(file_or_folder, **kwargs):
    """Load a photon-mosaic (or spikeinterface) object from disk.

    Parameters
    ----------
    file_or_folder : str or Path
        Path to a folder or file.
    **kwargs
        Forwarded to the specific loader.  Recognised keys include:

        * ``load_extensions`` (bool) — for ``RoiAnalyzer``
        * ``format`` (str) — for ``RoiAnalyzer``
        * ``backend_options`` (dict) — for ``RoiAnalyzer`` / zarr
        * ``storage_options`` (dict) — for zarr ROIs
        * ``base_folder`` — for SI extractor json/pkl files

    Returns
    -------
    object
        A ``RoiAnalyzer``, ``BinaryFolderRois``, ``ZarrRois``,
        or a spikeinterface ``BaseExtractor`` (Recording/Sorting).
    """
    path = Path(file_or_folder)

    # --- remote zarr store (e.g. s3://, gcs://) --------------------------------
    if is_path_remote(str(file_or_folder)):
        return _load_zarr(file_or_folder, **kwargs)

    # --- zarr store ---------------------------------------------------------
    if str(path).endswith(".zarr") or (path.is_dir() and (path / ".zgroup").exists()):
        return _load_zarr(path, **kwargs)

    # --- local folder -------------------------------------------------------
    if path.is_dir():
        return _load_folder(path, **kwargs)

    # --- single file (json / pkl) — fall back to spikeinterface -------------
    if path.is_file():
        from spikeinterface.core.loading import load as si_load

        return si_load(file_or_folder, **kwargs)

    raise ValueError(_error_msg.format(file_path=file_or_folder))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_folder(folder, **kwargs):
    """Dispatch a local folder to the right loader."""
    folder = Path(folder)

    # RoiAnalyzer  — identified by photon_mosaic_info.json
    if (folder / "photon_mosaic_info.json").is_file():
        from .roianalyzer import load_roi_analyzer

        return load_roi_analyzer(folder, **kwargs)

    # BinaryFolderRois — identified by binary.json + roi_image_masks.npy
    if (folder / "roi_image_masks.npy").is_file() and (folder / "metadata.json").is_file():
        from .binaryrois import BinaryFolderRois

        return BinaryFolderRois(folder_path=folder)

    # BinaryFolderImaging — identified by binary.json with BinaryImaging class
    if (folder / "binary.json").is_file():
        from .binaryimaging import BinaryFolderImaging

        return BinaryFolderImaging(folder_path=folder)

    # Use spikeinterface load to load from dict/json/pkl
    from spikeinterface.core.loading import load as si_load

    return si_load(str(folder), **kwargs)


def _load_zarr(path, **kwargs):
    """Dispatch a zarr store to the right loader."""
    from spikeinterface.core.zarrextractors import super_zarr_open

    storage_options = kwargs.get("storage_options", kwargs.get("backend_options", {}).get("storage_options", {}))
    zarr_root = super_zarr_open(str(path), mode="r", storage_options=storage_options or {})

    # RoiAnalyzer — identified by photon_mosaic_info
    info = zarr_root.attrs.get("photon_mosaic_info")
    if info is not None and info.get("object") == "RoiAnalyzer":
        from .roianalyzer import load_roi_analyzer

        return load_roi_analyzer(path, **kwargs)

    # Standalone zarr rois — has a "rois" group with roi_image_masks
    if "rois" in zarr_root and "roi_image_masks" in zarr_root["rois"]:
        from .zarrrois import ZarrRois

        return ZarrRois(path, zarr_group_name="rois", storage_options=storage_options)

    # Fall back to spikeinterface
    from spikeinterface.core.loading import load as si_load

    return si_load(str(path), **kwargs)
