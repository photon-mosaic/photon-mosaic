"""Binary folder format for ROI data.

Classes
-------
BinaryRois
    ROI extractor that loads image masks from a .npy file on disk.
BinaryFolderRois
    ROI extractor that loads from a folder containing masks, metadata, and provenance.
"""

import json
from pathlib import Path

import numpy as np
import sparse

from .baserois import BaseRois


class BinaryRois(BaseRois):
    """ROI extractor backed by .npy or .npz files on disk.

    Parameters
    ----------
    file_path : str or Path
        Path to the file containing image masks (num_rois, height, width[, planes]) --
        ``.npy`` for a dense array, ``.npz`` for a sparse `pydata/sparse` array.
    sampling_frequency : float
        Sampling frequency of the ROIs in Hz.
    roi_ids : ArrayLike
        Array of ROI IDs.
    shape : tuple
        Shape of the ROI masks (height, width[, planes]).
    """

    def __init__(self, file_path, sampling_frequency, roi_ids, shape):
        file_path = Path(file_path)
        roi_ids = np.asarray(roi_ids)

        BaseRois.__init__(
            self,
            sampling_frequency=sampling_frequency,
            shape=shape,
            roi_ids=roi_ids,
        )

        self._file_path = file_path
        self._kwargs = {
            "file_path": str(file_path.absolute()),
            "sampling_frequency": sampling_frequency,
            "roi_ids": roi_ids,
            "shape": shape,
        }

    def get_roi_image_masks(self, roi_ids=None):
        if self._file_path.suffix == ".npz":
            masks = sparse.load_npz(self._file_path)
        else:
            masks = np.load(self._file_path)
        if roi_ids is None:
            return masks
        roi_indices = self.ids_to_indices(roi_ids)
        return masks[roi_indices]


class BinaryFolderRois(BinaryRois):
    """ROI extractor that loads from a folder saved by ``BaseRois.save()``.

    The folder must contain:
    - ``roi_image_masks.npy`` (dense) or ``roi_image_masks.npz`` (sparse) — the mask data
    - ``roi_ids.npy`` — the ROI IDs
    - ``metadata.json`` — sampling_frequency and shape
    - ``binary.json`` — provenance file (written by ``BaseRois.save()``)

    Optionally:
    - ``properties/`` folder with per-ROI property .npy files
    - ``annotations.json`` with annotations

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder.
    """

    def __init__(self, folder_path):
        folder_path = Path(folder_path)

        with open(folder_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        roi_ids = np.load(folder_path / "roi_ids.npy")
        npz_path = folder_path / "roi_image_masks.npz"
        file_path = npz_path if npz_path.is_file() else folder_path / "roi_image_masks.npy"

        BinaryRois.__init__(
            self,
            file_path=file_path,
            sampling_frequency=metadata["sampling_frequency"],
            roi_ids=roi_ids,
            shape=tuple(metadata["shape"]),
        )

        self.load_metadata_from_folder(folder_path)

        # Override _kwargs so serialisation round-trips through BinaryFolderRois
        self._kwargs = dict(folder_path=str(folder_path.absolute()))
