"""Toy example ImagingExtractor and SegmentationExtractor for testing.

Functions
---------
toy_example
    Create a toy example of an ImagingExtractor and a SegmentationExtractor.
"""

import numpy as np

from .baseimaging import BaseImaging, BaseImagingSegment
from .numpyimaging import NumpyImaging


def _gaussian(x, mu, sigma):
    """Compute classical gaussian with parameters x, mu, sigma."""
    return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-((x - mu) ** 2) / sigma)


def _generate_rois(
    num_units=10,
    height=100,
    width=100,
    roi_size=4,
    min_dist=5,
    mode="uniform",
    seed=None,
):  # TODO: mode --> literal type
    """Generate ROIs with given parameters.

    Parameters
    ----------
    num_units: int
        Number of ROIs
    size_x: int
        Size of x dimension (pixels)
    size_y: int
        Size of y dimension (pixels)
    roi_size: int
        Size of ROI in x and y dimension (pixels)
    min_dist: int
        Minimum distance between ROI centers (pixels)
    mode: str
        'uniform' or 'gaussian'.
        If 'uniform', ROI values are uniform and equal to 1.
        If 'gaussian', ROI values are gaussian modulated

    Returns
    -------
    roi_pixels: list
        List of pixel coordinates for each ROI
    image: np.ndarray
        Image with ROIs
    means: list
        List of mean coordinates for each ROI
    """
    image = np.zeros((height, width))
    max_iter = 1000

    count = 0
    it = 0
    means = []

    rng = np.random.default_rng(seed=seed)

    while count < num_units:
        mean_x = rng.integers(0, height - 1)
        mean_y = rng.integers(0, width - 1)

        mean_ = np.array([mean_x, mean_y])

        if len(means) == 0:
            means.append(mean_)
            count += 1
        else:
            dists = np.array([np.linalg.norm(mean_ - m) for m in means])

            if np.all(dists > min_dist):
                means.append(mean_)
                count += 1

        it += 1

        if it >= max_iter:
            raise Exception("Could not fit ROIs given 'min_dist'")

    roi_pixels = []
    for m, mean in enumerate(means):
        # print(f"ROI {m + 1}/{num_units}")
        pixels = []
        for i in np.arange(height):
            for j in np.arange(width):
                p = np.array([i, j])

                if np.linalg.norm(p - mean) < roi_size:
                    pixels.append(p)
                    if mode == "uniform":
                        image[i, j] = 1
                    elif mode == "gaussian":
                        image[i, j] = _gaussian(i, mean[0], roi_size) + _gaussian(j, mean[1], roi_size)
                    else:
                        raise Exception("'mode' can be 'uniform' or 'gaussian'")
        roi_pixels.append(np.array(pixels))

    return roi_pixels, image, means


class NoiseGeneratorImaging(BaseImaging):

    def __init__(
        self, sampling_frequency=30, durations=[10], height=100, width=100, noise_std=0.05, seed=None, **noise_kwargs
    ):
        from spikeinterface.core.generate import NoiseGeneratorRecording

        super().__init__(sampling_frequency=sampling_frequency, shape=(height, width))
        self.noise_generator_recording = NoiseGeneratorRecording(
            num_channels=self.get_num_pixels(),
            sampling_frequency=sampling_frequency,
            durations=durations,
            noise_levels=noise_std,
            seed=seed,
            noise_block_size=100,
            **noise_kwargs,
        )
        seed = self.noise_generator_recording._kwargs["seed"]

        for segment_index in range(len(durations)):
            self.add_imaging_segment(
                NoiseGeneratorImagingSegment(
                    sampling_frequency=sampling_frequency,
                    noise_generator=self.noise_generator_recording,
                    segment_index=segment_index,
                    image_shape=self.image_shape,
                )
            )

        self._kwargs = dict(
            sampling_frequency=sampling_frequency,
            durations=durations,
            height=height,
            width=width,
            noise_std=noise_std,
            seed=seed,
        )
        self._kwargs.update(noise_kwargs)


class NoiseGeneratorImagingSegment(BaseImagingSegment):

    def __init__(self, sampling_frequency, noise_generator, segment_index, image_shape):
        super().__init__(sampling_frequency=sampling_frequency)
        self.noise_generator = noise_generator
        self.segment_index = segment_index
        self.image_shape = image_shape

    def get_num_samples(self):
        return self.noise_generator.get_num_samples(segment_index=self.segment_index)

    def get_series(self, start_frame, end_frame):
        traces = self.noise_generator.get_traces(
            start_frame=start_frame, end_frame=end_frame, segment_index=self.segment_index
        )
        video_shape = (traces.shape[0], self.image_shape[0], self.image_shape[1])
        return traces.reshape(video_shape)


class GroundTruthImaging(BaseImaging):

    def __init__(
        self,
        durations=[10],
        num_rois=10,
        height=100,
        width=100,
        roi_size=4,
        min_dist=5,
        mode="uniform",
        sampling_frequency=30.0,
        sorting_sampling_frequency=30_000,
        decay_time=0.5,
        noise_std=0.05,
        seed=None,
    ):
        from spikeinterface.core.generate import generate_sorting

        if seed is None:
            seed = np.random.default_rng(seed=None).integers(0, 2**63)

        assert (
            np.mod(sorting_sampling_frequency, sampling_frequency) == 0
        ), "sorting_sampling_frequency needs to be a multiple of sorting_frequency"

        # generate ROIs
        num_rois = int(num_rois)
        roi_pixels, roi_values, _ = _generate_rois(
            num_units=num_rois, height=height, width=width, roi_size=roi_size, min_dist=min_dist, mode=mode, seed=seed
        )

        self.noise_generator = NoiseGeneratorImaging(
            durations=durations, height=height, width=width, noise_std=noise_std, seed=seed
        )

        self.sorting = generate_sorting(
            durations=durations, num_units=num_rois, sampling_frequency=sorting_sampling_frequency, seed=seed
        )

        self.noise_generator = NoiseGeneratorImaging(
            durations=durations,
            height=height,
            width=width,
            noise_std=noise_std,
        )

        super().__init__(sampling_frequency=sampling_frequency, shape=(height, width))

        # create decaying response
        resp_samples = int(decay_time * sorting_sampling_frequency)
        resp_tau = resp_samples / 5
        tresp = np.arange(resp_samples)
        kernel = np.exp(-tresp / resp_tau)

        # add single segment
        for segment_index, noise_segment in enumerate(self.noise_generator._imaging_segments):
            gt_segment = GroundTruthImagingSegment(
                sampling_frequency=sampling_frequency,
                noise_segment=noise_segment,
                segment_index=segment_index,
                sorting=self.sorting,
                roi_pixels=roi_pixels,
                roi_values=roi_values,
                kernel=kernel,
            )
            self.add_imaging_segment(gt_segment)

        self._kwargs = dict(
            durations=durations,
            num_rois=num_rois,
            height=height,
            width=width,
            roi_size=roi_size,
            min_dist=min_dist,
            mode=mode,
            sampling_frequency=sampling_frequency,
            sorting_sampling_frequency=sorting_sampling_frequency,
            decay_time=decay_time,
            noise_std=noise_std,
        )


class GroundTruthImagingSegment(BaseImagingSegment):
    def __init__(
        self, sampling_frequency, segment_index, noise_segment, sorting, roi_pixels, roi_values, kernel, min_samples=10
    ):
        super().__init__(sampling_frequency=sampling_frequency)
        self.noise_segment = noise_segment
        self.segment_index = segment_index
        self.sorting = sorting
        self.roi_pixels = roi_pixels
        self.roi_values = roi_values
        self.kernel_high_res = kernel
        self.decimation_factor = int(self.sorting.sampling_frequency // self.sampling_frequency)
        self.min_samples = min_samples

    def get_num_samples(self):
        return self.noise_segment.get_num_samples()

    def get_series(self, start_frame, end_frame):
        if (end_frame - start_frame) < self.min_samples:
            end_frame_ = self.min_samples - (end_frame - start_frame)
        else:
            end_frame_ = end_frame

        noise = self.noise_segment.get_series(start_frame, end_frame_)
        fluo_hr = np.zeros((len(self.roi_pixels), (end_frame_ - start_frame) * self.decimation_factor))
        t_start = max(0, start_frame / self.sampling_frequency)  # - self.kernel_duration)
        t_stop = end_frame_ / self.sampling_frequency
        sorting_sliced = self.sorting.time_slice(t_start, t_stop)
        resp = self.kernel_high_res
        num_hr_frames = fluo_hr.shape[1]
        for u_i, unit_id in enumerate(sorting_sliced.unit_ids):
            spike_train = sorting_sliced.get_unit_spike_train(
                unit_id,
                segment_index=self.segment_index,
            )
            for spike_index in spike_train:  # TODO build a local function that generates frames with spikes
                if spike_index + len(resp) < num_hr_frames:
                    fluo_hr[u_i, spike_index : spike_index + len(resp)] += resp
                else:
                    fluo_hr[u_i, spike_index:] = resp[: num_hr_frames - spike_index]

        # now decimate
        fluo_traces = fluo_hr[:, :: self.decimation_factor]

        # generate video
        fluo_video = np.zeros((noise.shape))
        for roi_pixels, fluo_trace in zip(self.roi_pixels, fluo_traces):
            for pixel in roi_pixels:
                fluo_video[:, *pixel] += fluo_trace * self.roi_values[*pixel]

        video = fluo_video + noise
        return video[: (end_frame - start_frame)]


generate_gt_video = GroundTruthImaging


def generate_ground_truth_video(
    duration=10,
    num_rois=10,
    size_x=100,
    size_y=100,
    roi_size=4,
    min_dist=5,
    mode="uniform",
    sampling_frequency=30.0,
    decay_time=0.5,
    noise_std=0.05,
):
    """Create a toy example of an ImagingExtractor and a SegmentationExtractor.

    Parameters
    ----------
    duration: float
        Duration in s
    num_rois: int
        Number of ROIs
    size_x: int
        Size of x dimension (pixels)
    size_y: int
        Size of y dimension (pixels)
    roi_size: int
        Size of ROI in x and y dimension (pixels)
    min_dist: int
        Minimum distance between ROI centers (pixels)
    mode: str
        'uniform' or 'gaussian'.
        If 'uniform', ROI values are uniform and equal to 1.
        If 'gaussian', ROI values are gaussian modulated
    sampling_frequency: float
        The sampling rate
    decay_time: float
        Decay time of fluorescence response
    noise_std: float
        Standard deviation of added gaussian noise

    Returns
    -------
    imag: NumpyImagingExtractor
        The output imaging extractor
    seg: NumpySegmentationExtractor
        The output segmentation extractor
    """
    # generate ROIs
    num_rois = int(num_rois)
    roi_pixels, im, means = _generate_rois(
        num_units=num_rois,
        size_x=size_x,
        size_y=size_y,
        roi_size=roi_size,
        min_dist=min_dist,
        mode=mode,
    )

    from spikeinterface.core import generate_sorting

    sort = generate_sorting(durations=[duration], num_units=num_rois, sampling_frequency=sampling_frequency)

    # create decaying response
    resp_samples = int(decay_time * sampling_frequency)
    resp_tau = resp_samples / 5
    tresp = np.arange(resp_samples)
    resp = np.exp(-tresp / resp_tau)

    num_frames = int(sampling_frequency * duration)

    # convolve response with ROIs
    # TODO: optimize this and make it lazy
    raw = np.zeros((num_rois, num_frames))  # TODO Change to new standard formatting with time in first axis
    deconvolved = np.zeros((num_rois, num_frames))  # TODO Change to new standard formatting with time in first axis
    frames = num_frames
    for u_i in range(num_rois):
        unit_id = sort.unit_ids[u_i]
        for s in sort.get_unit_spike_train(unit_id):  # TODO build a local function that generates frames with spikes
            if s < num_frames:
                if s + len(resp) < frames:
                    raw[u_i, s : s + len(resp)] += resp
                else:
                    raw[u_i, s:] = resp[: frames - s]
                deconvolved[u_i, s] = 1

    # generate video
    video = np.zeros((frames, size_x, size_y))
    for rp, t in zip(roi_pixels, raw):
        for r in rp:
            video[:, r[0], r[1]] += t * im[r[0], r[1]]

    # normalize video
    video /= np.max(video)

    # add noise
    video += noise_std * np.abs(np.random.randn(*video.shape))

    # instantiate imaging and segmentation extractors
    imag = NumpyImaging(timeseries=video, sampling_frequency=sampling_frequency)

    # create image masks
    image_masks = np.zeros((size_x, size_y, num_rois))
    for rois_i, roi in enumerate(roi_pixels):
        for r in roi:
            image_masks[r[0], r[1], rois_i] += im[r[0], r[1]]

    # seg = NumpySegmentationExtractor(
    #     image_masks=image_masks,
    #     raw=raw,
    #     deconvolved=deconvolved,
    #     neuropil=neuropil,
    #     sampling_frequency=float(sampling_frequency),
    # )
    seg = None

    return imag, seg
