import numpy as np
import pytest

from photon_mosaic.core.arrayimaging import ArrayImaging, NumpyImagingSegment
import dask.array as da

@pytest.fixture(scope="module")
def np_video():
    shape = (1000,256,256,50)
    return np.random.randint(0, 4000, size=shape, dtype=np.uint16)


def test_compare_numpy_imaging_segment(benchmark, np_video):
    np_imaging = ArrayImaging(np_video, sampling_frequency = 30)
    np_plane_3 = benchmark(np_imaging.get_series, **{"start_frame": 0, "end_frame": 100, "plane_ids" :[3]})
    
    assert np_plane_3.shape ==  (100,) + np_video.shape[1:3] + (1,)


def test_compare_dask_imaging_segment(benchmark, np_video):
    da_video = da.from_array(np_video, chunks=(100, -1, -1, 1)) 

    da_imaging = ArrayImaging(da_video, sampling_frequency = 30)
    
    da_plane_3 = benchmark(lambda : da_imaging.get_series(0, 100, plane_ids=[3]).compute())
    assert da_plane_3.shape == (100,) + np_video.shape[1:3] + (1,)