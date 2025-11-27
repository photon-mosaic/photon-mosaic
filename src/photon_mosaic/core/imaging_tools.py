from pathlib import Path

import numpy as np
from spikeinterface.core.core_tools import add_suffix
from spikeinterface.core.recording_tools import get_random_slices
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs

from .job_tools import ChunkExecutor
from .utils import DTypeLike, PathType


def get_random_data_chunks(imaging, concatenated=True, **random_slices_kwargs):
    """
    Extract random chunks across segments.

    Internally, it uses `get_random_slices()` and retrieves the traces chunk as a list
    or a concatenated unique array.

    Please read `get_random_slices()` for more details on parameters.


    Parameters
    ----------
    imaging : BaseImaging
        The imaging extractor to get random chunks from
    num_chunks_per_segment : int, default: 20
        Number of chunks per segment
    concatenated : bool, default: True
        If True chunk are concatenated along time axis
    **random_slices_kwargs : dict
        Options transmited to  get_random_slices(), please read documentation from this
        function for more details.

    Returns
    -------
    chunk_list : np.array | list of np.array
        Array of concatenate chunks per segment
    """
    slices = get_random_slices(imaging, **random_slices_kwargs)

    chunk_list = []
    for segment_index, start_frame, end_frame in slices:
        series_chunk = imaging.get_series(
            start_frame=start_frame,
            end_frame=end_frame,
            segment_index=segment_index,
        )
        chunk_list.append(series_chunk)

    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list


# used by write_binary_recording + ChunkRecordingExecutor
def _init_binary_worker(imaging, file_path_dict, dtype, byte_offest):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["imaging"] = imaging
    worker_ctx["byte_offset"] = byte_offest
    worker_ctx["dtype"] = np.dtype(dtype)

    file_dict = {segment_index: open(file_path, "rb+") for segment_index, file_path in file_path_dict.items()}
    worker_ctx["file_dict"] = file_dict

    return worker_ctx


# used by write_binary_recording + ChunkRecordingExecutor
def _write_binary_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    imaging = worker_ctx["imaging"]
    dtype = worker_ctx["dtype"]
    byte_offset = worker_ctx["byte_offset"]
    file = worker_ctx["file_dict"][segment_index]

    num_pixels = imaging.get_num_pixels()
    dtype_size_bytes = np.dtype(dtype).itemsize

    # Calculate byte offsets for the start frames relative to the entire recording
    start_byte = byte_offset + start_frame * num_pixels * dtype_size_bytes

    video = imaging.get_series(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    video = video.astype(dtype, order="c", copy=False)

    file.seek(start_byte)
    file.write(video.data)
    # flush is important!!
    file.flush()


def write_binary_imaging(
    imaging: "BaseImaging",
    file_paths: list[PathType] | PathType,
    dtype: DTypeLike | None = None,
    add_file_extension: bool = True,
    byte_offset: int = 0,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Save the video of an imaging extractor in several binary .dat format.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object to be saved in .dat format
    file_path : str or list[str]
        The path to the file.
    dtype : dtype or None, default: None
        Type of the saved data
    add_file_extension, bool, default: True
        If True, and  the file path does not end in "raw", "bin", or "dat" then "raw" is added as an extension.
    byte_offset : int, default: 0
        Offset in bytes for the binary file (e.g. to write a header). This is useful in case you want to append data
        to an existing file where you wrote a header or other data before.
    verbose : bool
        This is the verbosity of the ChunkRecordingExecutor
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    file_path_list = [file_paths] if not isinstance(file_paths, list) else file_paths
    num_segments = imaging.get_num_segments()
    if len(file_path_list) != num_segments:
        raise ValueError("'file_paths' must be a list of the same size as the number of segments in the recording")

    file_path_list = [Path(file_path) for file_path in file_path_list]
    if add_file_extension:
        file_path_list = [add_suffix(file_path, ["raw", "bin", "dat"]) for file_path in file_path_list]

    dtype = dtype if dtype is not None else imaging.get_dtype()

    dtype_size_bytes = np.dtype(dtype).itemsize
    num_pixels = imaging.get_num_pixels()

    file_path_dict = {segment_index: file_path for segment_index, file_path in enumerate(file_path_list)}
    for segment_index, file_path in file_path_dict.items():
        num_frames = imaging.get_num_samples(segment_index=segment_index)
        data_size_bytes = dtype_size_bytes * num_frames * num_pixels
        file_size_bytes = data_size_bytes + byte_offset

        # Create an empty file with file_size_bytes
        with open(file_path, "wb+") as file:
            # The previous implementation `file.truncate(file_size_bytes)` was slow on Windows (#3408)
            file.seek(file_size_bytes - 1)
            file.write(b"\0")

        assert Path(file_path).is_file()

    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (imaging, file_path_dict, dtype, byte_offset)
    executor = ChunkExecutor(
        imaging,
        func,
        init_func,
        init_args,
        job_name="write_binary_imaging",
        verbose=verbose,
        **job_kwargs,
    )
    executor.run()


write_binary_imaging.__doc__ = write_binary_imaging.__doc__.format(_shared_job_kwargs_doc)
