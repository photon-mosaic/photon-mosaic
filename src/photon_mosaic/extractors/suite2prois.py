from pathlib import Path

import numpy as np

from photon_mosaic.core import BaseRois


class Suite2pRois(BaseRois):
    """Suite2p ROIs extractor."""

    def __init__(self, folder_path: str | Path):
        """Create a Suite2pRois extractor from a Suite2p folder."""
        ops_file = Path(folder_path) / "ops.npy"
        stat_file = Path(folder_path) / "stat.npy"
        if not ops_file.is_file():
            raise FileNotFoundError(f"No ops.npy file found in {folder_path}")
        if not stat_file.is_file():
            raise FileNotFoundError(f"No stat.npy file found in {folder_path}")

        ops = np.load(ops_file, allow_pickle=True).item()
        height = ops["Lyc"]
        width = ops["Lxc"]
        sampling_frequency = ops["fs"]

        self.stats = np.load(stat_file, allow_pickle=True)
        roi_ids = np.arange(len(self.stats))

        BaseRois.__init__(
            self,
            sampling_frequency=sampling_frequency,
            shape=(height, width),
            roi_ids=roi_ids,
        )

        # set properties
        available_properties = list(self.stats[0].keys())

        skip_properties = ["xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"]

        for prop in available_properties:
            if prop in skip_properties:
                continue
            values = [stat[prop] for stat in self.stats]
            try:
                self.set_property(prop, values)
            except Exception as e:
                print(f"Error setting property {prop}: {e}")

        is_cell_file = Path(folder_path) / "iscell.npy"
        if is_cell_file.is_file():
            iscell = np.load(is_cell_file)
            iscell_bool = iscell[:, 0] == 1
            iscell_prob = iscell[:, 1]
            self.set_property("iscell", iscell_bool)
            self.set_property("iscell_probability", iscell_prob)

        self._kwargs = {"suite2p_folder": folder_path}

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_ids = self.roi_ids
        masks = []
        for roi_id in roi_ids:
            stat = self.stats[roi_id]
            mask = np.zeros(self.shape[:2], dtype=bool)
            ypix = stat["ypix"]
            xpix = stat["xpix"]
            mask[ypix, xpix] = True
            masks.append(mask)
        return np.array(masks)


read_suite2p_rois = Suite2pRois
