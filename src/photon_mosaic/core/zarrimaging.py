from pathlib import Path

import numpy as np
import zarr
from spikeinterface.core.chunkable_tools import write_chunkable_to_zarr
from spikeinterface.core.core_tools import check_json, retrieve_importing_provenance
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.zarrextractors import (
    add_properties_and_annotations,
    get_default_zarr_compressor,
    resolve_zarr_path,
    super_zarr_open,
)

from photon_mosaic.core.baseimaging import BaseImaging, BaseImagingEpoch


class ZarrImaging(BaseImaging):
    """An Imaging object that is backed by a zarr array on disk.
    This allows for memory efficient access to large imaging datasets that do not fit in memory.

    Parameters
    ----------
    folder_path: str | Path
        The path to the folder containing the zarr group with the imaging data.
    storage_options: dict | None
        Optional storage options for opening the zarr group.
    """

    def __init__(
        self, folder_path: str | Path, storage_options: dict | None = None, load_compression_ratio: bool = False
    ):
        folder_path, folder_path_kwarg = resolve_zarr_path(folder_path)

        self._root = super_zarr_open(folder_path, mode="r", storage_options=storage_options)

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "shape" in self._root.keys(), "'shape' dataset not found!"
        shapes = self._root["shape"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attribute not found!"
        assert num_segments is not None, "'num_segments' attribute not found!"

        BaseImaging.__init__(self, sampling_frequency=sampling_frequency, shape=shapes)

        t_starts = self._root.get("t_starts", None)
        if load_compression_ratio:
            total_nbytes = 0
            total_nbytes_stored = 0
            cr_by_segment = {}
        for segment_index in range(num_segments):
            video_name = f"video_seg{segment_index}"

            time_kwargs = {}
            time_vector = self._root.get(f"times_seg{segment_index}", None)
            if time_vector is not None:
                time_kwargs["time_vector"] = time_vector
            else:
                if t_starts is None:
                    t_start = None
                else:
                    t_start = t_starts[segment_index]
                    if np.isnan(t_start):
                        t_start = None
                time_kwargs["t_start"] = t_start
            time_kwargs["sampling_frequency"] = sampling_frequency

            epoch = ZarrImagingEpoch(self._root, video_name, **time_kwargs)
            self.add_epoch(epoch)

            if load_compression_ratio:
                nbytes_segment = self._root[video_name].nbytes
                nbytes_stored_segment = self._root[video_name].nbytes_stored
                if nbytes_stored_segment > 0:
                    cr_by_segment[segment_index] = nbytes_segment / nbytes_stored_segment
                else:
                    cr_by_segment[segment_index] = np.nan

                total_nbytes += nbytes_segment
                total_nbytes_stored += nbytes_stored_segment

        # load properties
        if "properties" in self._root:
            prop_group = self._root["properties"]
            for key in prop_group.keys():
                values = self._root["properties"][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)
        if load_compression_ratio:
            # annotate compression ratios
            if total_nbytes_stored > 0:
                cr = total_nbytes / total_nbytes_stored
            else:
                cr = np.nan
            self.annotate(compression_ratio=cr, compression_ratio_segments=cr_by_segment)

        self._kwargs = {
            "folder_path": folder_path_kwarg,
            "storage_options": storage_options,
            "load_compression_ratio": load_compression_ratio,
        }


class ZarrImagingEpoch(BaseImagingEpoch):
    def __init__(self, root, dataset_name, **time_kwargs):
        BaseImagingEpoch.__init__(self, **time_kwargs)
        self._video = root[dataset_name]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self._video.shape[0]

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: list[int] | None = None,
    ) -> np.ndarray:
        video = self._video[start_frame:end_frame]
        if plane_indices is not None:
            video = video[..., plane_indices]
        return video


def add_imaging_to_zarr_group(
    imaging: BaseImaging, zarr_group: zarr.hierarchy.Group, verbose=False, dtype=None, **kwargs
):
    zarr_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if imaging.check_if_json_serializable():
        zarr_group.attrs["provenance"] = check_json(imaging.to_dict(recursive=True))
    else:
        zarr_group.attrs["provenance"] = None

    # save data (done the subclass)
    zarr_group.attrs["sampling_frequency"] = float(imaging.get_sampling_frequency())
    zarr_group.attrs["num_segments"] = int(imaging.get_num_segments())
    zarr_group.create_dataset(name="shape", data=imaging.shape, compressor=None)
    dataset_paths = [f"video_seg{i}" for i in range(imaging.get_num_segments())]

    dtype = imaging.get_dtype() if dtype is None else dtype
    extra_chunks = zarr_kwargs.get("extra_chunks", None)
    global_compressor = zarr_kwargs.pop("compressor", get_default_zarr_compressor())
    compressor_by_dataset = zarr_kwargs.pop("compressor_by_dataset", {})
    global_filters = zarr_kwargs.pop("filters", None)
    filters_by_dataset = zarr_kwargs.pop("filters_by_dataset", {})
    compressor_videos = compressor_by_dataset.get("videos", global_compressor)
    filters_videos = filters_by_dataset.get("videos", global_filters)
    compressor_times = compressor_by_dataset.get("times", global_compressor)
    filters_times = filters_by_dataset.get("times", global_filters)

    write_chunkable_to_zarr(
        chunkable=imaging,
        zarr_group=zarr_group,
        dataset_paths=dataset_paths,
        compressor_data=compressor_videos,
        filters_data=filters_videos,
        compressor_times=compressor_times,
        filters_times=filters_times,
        dtype=dtype,
        extra_chunks=extra_chunks,
        verbose=verbose,
        **job_kwargs,
    )

    add_properties_and_annotations(zarr_group, imaging)
    zarr_group.attrs["zarr_class_info   "] = retrieve_importing_provenance(ZarrImaging)
