import numpy as np

from photon_mosaic.core.baseimaging import BaseImaging


def assert_imaging_equal(imaging1: BaseImaging, imaging2: BaseImaging):
    assert imaging1.sampling_frequency == imaging2.sampling_frequency
    assert imaging1.shape == imaging2.shape
    assert len(imaging1.epochs) == len(imaging2.epochs)
    assert imaging1.get_num_epochs() == imaging2.get_num_epochs()
    assert imaging1.get_num_planes() == imaging2.get_num_planes()
    assert imaging1.shape == imaging2.shape
    for epoch_index in range(imaging1.get_num_epochs()):
        np.testing.assert_array_equal(
            imaging1.get_series(epoch_index=epoch_index), imaging2.get_series(epoch_index=epoch_index)
        )
        # slice and assert
        start = 2
        end = 6
        imaging1_slice = imaging1.get_series(start_frame=start, end_frame=end, epoch_index=epoch_index)
        imaging2_slice = imaging2.get_series(start_frame=start, end_frame=end, epoch_index=epoch_index)
        np.testing.assert_array_equal(
            imaging1_slice,
            imaging2_slice,
        )
        shape1 = imaging1.shape
        assert imaging1_slice.shape == (end - start, shape1[0], shape1[1], shape1[2])
        assert imaging2_slice.shape == (end - start, shape1[0], shape1[1], shape1[2])
