"""Build a :class:`MultiFieldImaging` from one or more ScanImage multi-ROI (mesoscope) TIFFs.

Each scan field is loaded as a standalone :class:`BaseImaging` through photon-mosaic's dynamically
generated ScanImage loader, and its lateral placement is read off the underlying roiextractors
extractor. When several files of one recording are passed, each field's imaging spans them all.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
from roiextractors import ScanImageMultiROIImagingExtractor

# Bind the extractors package (not its dynamic attributes) so this module can be imported during
# package init; the generated loaders are resolved lazily at call time, once they exist.
import photon_mosaic.extractors as pm_extractors
from photon_mosaic.core import BaseImaging


def _attach_field_timestamps(
    imaging: BaseImaging,
    extractor: Any,
    timestamps: str,
    full_frame_timestamps: np.ndarray | None,
) -> None:
    """Mirror a field's acquisition timestamps from the extractor onto its imaging object.

    The extractor resolves the base per-frame timeline -- a custom override set here via
    ``set_times``, else the file's native ``frameTimestamps_sec``, else nominal
    ``arange(num_frames) / sampling_frequency`` -- and, in ``"per_field"`` mode, adds the field's
    within-frame scan offset (``get_field_time_offset()``: ScanImage's per-row dwell time times the
    field's start row, modelling that fields are scanned sequentially top to bottom). This copies the
    resolved vector onto the :class:`BaseImaging` time vector, which photon-mosaic reads
    independently of the extractor.

    Parameters
    ----------
    imaging : BaseImaging
        The field to annotate (single epoch).
    extractor : Any
        The underlying roiextractors ``ScanImageMultiROIImagingExtractor``.
    timestamps : str
        ``"per_frame"`` (frame timestamps as-is) or ``"per_field"`` (add the within-frame offset).
    full_frame_timestamps : np.ndarray | None
        Optional custom per-frame base timeline shared by all fields, or None to use the extractor's
        own timestamps.
    """
    num_frames = extractor.get_num_samples()

    # A custom base timeline (shared across fields) overrides the extractor's native timestamps; the
    # per-field offset is applied on read below, not baked into the stored base.
    if full_frame_timestamps is not None:
        if full_frame_timestamps.size < num_frames:
            raise ValueError(
                f"frame_timestamps has {full_frame_timestamps.size} entries but the field has "
                f"{num_frames} frames; need at least as many"
            )
        extractor.set_times(np.asarray(full_frame_timestamps[:num_frames], dtype=float))

    # The extractor resolves the base timeline and adds the per-field offset when requested (raising
    # if "per_field" is asked for but the offset can't be computed); mirror the result onto the
    # imaging object's own (independent) time vector.
    correct_field_offset = timestamps == "per_field"
    field_times = extractor.get_timestamps(correct_field_offset=correct_field_offset)
    imaging.set_times(np.asarray(field_times, dtype=float), segment_index=0, with_warning=False)


def multifield_from_scanimage(
    file_path: str | Path | None = None,
    file_paths: list[str | Path] | None = None,
    timestamps: Literal["per_frame", "per_field"] = "per_frame",
    frame_timestamps: str | Path | np.ndarray | None = None,
) -> list[BaseImaging]:
    """Build a list of imaging fields from one or more ScanImage multi-ROI TIFFs.

    A ScanImage recording is often split across several files. Passing them all ties their samples
    together, so each field's :class:`BaseImaging` spans the whole recording in time.

    Parameters
    ----------
    file_path : str | Path | None
        Path to a single ScanImage TIFF. If the file is part of a multi-file series, this should be
        the first file and the remaining files are detected automatically. Mutually exclusive with
        ``file_paths``.
    file_paths : list[str | Path] | None
        Explicit list of files in temporal order, overriding automatic detection. Use this when the
        files cannot be auto-detected or you want exact control over their order.
    timestamps : "per_frame" | "per_field", default: "per_frame"
        How to time-stamp the fields. ``"per_frame"`` treats each frame as acquired in a single
        instant, so all fields share the same per-frame timeline. ``"per_field"`` adds a per-field
        offset modelling sequential scanning within a frame: ``offset = line_period * roi_row_start``
        (ScanImage's per-row dwell time, from metadata, times the field's start row).
    frame_timestamps : str | Path | np.ndarray | None, default: None
        Optional custom per-frame timestamps, treated as full-frame timestamps shared by all fields
        (then offset per field when ``timestamps="per_field"``). May be an array or a path to a
        ``.npy`` file; it is flattened and cropped to the frames read. If None, each field uses the
        extractor's native per-frame timestamps (the file's ``frameTimestamps_sec``), falling back to
        the sampling frequency when those are unavailable.

    Returns
    -------
    list[BaseImaging]
        One :class:`BaseImaging` per scan field (each spanning all input files), with its lateral
        placement stored in the field's ``geometry`` dict (see :attr:`BaseImaging.geometry`).
    """
    if timestamps not in ("per_frame", "per_field"):
        raise ValueError(f"timestamps must be 'per_frame' or 'per_field', got {timestamps!r}")
    if file_path is None and file_paths is None:
        raise ValueError("provide either file_path or file_paths")
    if file_paths is not None and len(file_paths) == 0:
        raise ValueError("file_paths must not be empty")

    # Load custom full-frame timestamps once (shared across all fields), if given.
    full_frame_timestamps: np.ndarray | None = None
    if frame_timestamps is not None:
        if isinstance(frame_timestamps, (str, Path)):
            frame_timestamps = np.load(Path(frame_timestamps).expanduser())
        full_frame_timestamps = np.asarray(frame_timestamps, dtype=float).ravel()

    # read_scan_image_multi_roi_imaging is generated dynamically at import time (see
    # extractors/roiextractors.py), so it is invisible to static analysis; resolve it at call time.
    read_field = pm_extractors.read_scan_image_multi_roi_imaging  # type: ignore[attr-defined]

    # The loader forwards these straight to the underlying extractor, which concatenates the files.
    # file_paths (explicit order) takes precedence over file_path (single / series head).
    loader_kwargs: dict[str, Any]
    if file_paths is not None:
        loader_kwargs = {"file_paths": [Path(path) for path in file_paths]}
        first_file = Path(file_paths[0])
    else:
        assert file_path is not None  # guaranteed by the validation above
        loader_kwargs = {"file_path": Path(file_path)}
        first_file = Path(file_path)

    # Cheap metadata read: how many fields are stacked. The ROI layout is identical across files in
    # a series, so the first file suffices. get_num_rois is a static method on the underlying
    # roiextractors extractor, not on the photon-mosaic wrapper.
    num_fields = ScanImageMultiROIImagingExtractor.get_num_rois(first_file)

    fields: list[BaseImaging] = []
    for field_index in range(num_fields):
        # Each field is wrapped as a standalone BaseImaging via the dynamic loader; passing the
        # file(s) makes the field span the full recording across files.
        imaging = read_field(roi_index=field_index, **loader_kwargs)
        # Reach through the wrapper to the underlying extractor for the physical geometry.
        extractor = imaging.epochs[0].roiextractor_extractor
        # The geometry dict is stored as a JSON-serializable annotation (see BaseImaging.geometry), so
        # the affine is a 3x3 list of lists rather than an ndarray; wrap in np.asarray at use.
        affine = np.asarray(extractor.roi_affine, dtype=float).tolist() if extractor.roi_affine is not None else None
        # Microns per scanner-angle unit: ScanImage's objective resolution, read from the frame
        # metadata. Lets downstream code express the geometry in micrometers as well as scanner units.
        objective_resolution = extractor._metadata.get("SI.objectiveResolution")
        microns_per_scanner_unit = float(objective_resolution) if objective_resolution is not None else None
        # Lateral placement and identity, stored directly on the field (see BaseImaging.geometry).
        imaging.geometry = dict(
            name=extractor.roi_name,
            uuid=extractor.roi_uuid,
            center_xy=list(extractor.roi_center_xy) if extractor.roi_center_xy is not None else None,
            size_xy=list(extractor.roi_size_xy) if extractor.roi_size_xy is not None else None,
            affine=affine,
            microns_per_scanner_unit=microns_per_scanner_unit,
        )
        # Attach per-plane physical depth (z) when the extractor exposes it. ScanImage stores the
        # ROI's z-plane(s) as ``zs`` -- a scalar for a planar field, one value per plane for a
        # (future) volumetric z-stack. Depth belongs on the field's own plane axis, so it is set as
        # a per-plane property; skip if the count does not match the field's planes (no fabrication).
        roi_zs = getattr(extractor, "roi_zs", None)
        if roi_zs is not None:
            plane_depths = np.atleast_1d(np.asarray(roi_zs, dtype=float))
            if plane_depths.size == imaging.num_planes:
                imaging.set_property("z", plane_depths)
        # Attach the time vector (shared per-frame, optionally offset per field).
        _attach_field_timestamps(imaging, extractor, timestamps, full_frame_timestamps)
        fields.append(imaging)

    # Fields are expected to be co-acquired (one shared timebase); flag a mixed set rather than fail.
    sampling_frequencies = {round(field.sampling_frequency, 4) for field in fields}
    if len(sampling_frequencies) > 1:
        warnings.warn(
            f"fields have heterogeneous sampling frequencies: {sorted(sampling_frequencies)}",
            UserWarning,
        )

    return fields
