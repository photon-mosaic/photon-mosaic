from __future__ import annotations

from photon_mosaic.core import BaseImaging, BaseImagingEpoch


class BasePreprocessor(BaseImaging):
    def __init__(self, imaging, sampling_frequency=None, dtype=None):
        assert isinstance(imaging, BaseImaging), "'imaging' must be a BaseImaging"

        if sampling_frequency is None:
            sampling_frequency = imaging.sampling_frequency

        if dtype is None:
            dtype = imaging.get_dtype()

        BaseImaging.__init__(self, sampling_frequency=sampling_frequency, shape=imaging.shape)
        imaging.copy_metadata(self, only_main=False)
        self._parent = imaging

        # self._kwargs have to be handled in subclass


class BasePreprocessorEpoch(BaseImagingEpoch):
    def __init__(self, parent_imaging_epoch):
        BaseImagingEpoch.__init__(self, **parent_imaging_epoch.get_times_kwargs())
        self.parent_imaging_epoch = parent_imaging_epoch

    def get_num_samples(self):
        return self.parent_imaging_epoch.get_num_samples()

    def get_series(self, start_frame, end_frame, plane_indices=None):
        raise NotImplementedError
