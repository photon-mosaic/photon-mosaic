from __future__ import annotations

from typing import Any, Sequence

from numpy.typing import DTypeLike, NDArray

from photon_mosaic.core import BaseImaging, BaseImagingEpoch


class BasePreprocessor(BaseImaging):
    def __init__(
        self,
        imaging: BaseImaging,
        sampling_frequency: float | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Wrap a `BaseImaging` object with preprocessing metadata.

        Parameters
        ----------
        imaging : BaseImaging
            Parent imaging object providing frames and metadata.
        sampling_frequency : float | None, optional
            Override for the output sampling frequency. Defaults to the parent's value.
        dtype : DTypeLike | None, optional
            Desired dtype for downstream processing. Defaults to parent's dtype.
        """

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
    def __init__(self, parent_imaging_epoch: BaseImagingEpoch) -> None:
        """Epoch wrapper that delegates metadata to its parent imaging epoch."""
        BaseImagingEpoch.__init__(self, **parent_imaging_epoch.get_times_kwargs())
        self.parent_imaging_epoch = parent_imaging_epoch

    def get_num_samples(self) -> int:
        """Return the number of samples in the parent epoch."""
        return self.parent_imaging_epoch.get_num_samples()

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: int | slice | Sequence[int] | None = None,
    ) -> NDArray[Any]:
        """Return a frame series for the requested interval and planes.

        Subclasses must override this to apply their specific preprocessing before
        returning the requested frames.
        """

        raise NotImplementedError
