import numpy as np
import pytest
from pydantic import ValidationError

from photon_mosaic.core import NumpyImaging
from photon_mosaic.core.generators import generate_imaging_with_rois
from photon_mosaic.extractors import Suite2pRois
from photon_mosaic.segmentation import (
    EpochSegmentations,
    read_segmentation_output,
    run_segmentation,
)
from photon_mosaic.segmentation import suite2p_segmentation as seg


def _install_fake_inner(monkeypatch):
    """Mock the streaming binner and the algorithm dispatcher.

    Returns ``(bin_calls, detect_calls)`` so tests can assert which epochs,
    ranges, and badframes the public API forwarded to each layer.
    """
    bin_calls: list[dict] = []
    detect_calls: list[dict] = []

    def fake_stream_bin_movie(
        imaging,
        plane_index,
        epoch_indices,
        *,
        bin_size,
        yrange,
        xrange,
        badframes,
        nbins,
    ):
        bin_calls.append(
            {
                "plane_index": plane_index,
                "epoch_indices": list(epoch_indices),
                "yrange": list(yrange),
                "xrange": list(xrange),
                "badframes": None if badframes is None else np.array(badframes, copy=True),
                "bin_size": bin_size,
                "nbins": nbins,
            }
        )
        n = sum(imaging.get_num_samples(segment_index=i) for i in epoch_indices)
        Lyc = yrange[1] - yrange[0]
        Lxc = xrange[1] - xrange[0]
        return np.zeros((n, Lyc, Lxc), dtype=np.float32)

    def fake_detect_rois_from_mov(mov, *, cfg, Ly, Lx, yrange, xrange):
        detect_calls.append(
            {
                "mov_shape": mov.shape,
                "Ly": Ly,
                "Lx": Lx,
                "yrange": list(yrange),
                "xrange": list(xrange),
                "algorithm": cfg.algorithm,
            }
        )
        return [
            {
                "ypix": np.array([0]),
                "xpix": np.array([1]),
                "lam": np.array([1.0]),
                "npix": 1,
            }
        ]

    monkeypatch.setattr(seg, "_stream_bin_movie", fake_stream_bin_movie)
    monkeypatch.setattr(seg, "_detect_rois_from_mov", fake_detect_rois_from_mov)
    return bin_calls, detect_calls


def _make_test_imaging():
    epoch0 = np.ones((2, 4, 5, 1), dtype=np.float32)
    epoch1 = np.full((3, 4, 5, 1), 2.0, dtype=np.float32)
    return NumpyImaging([epoch0, epoch1], sampling_frequency=7.0)


# ---------------------------------------------------------------------------
# run_segmentation(method="suite2p")
# ---------------------------------------------------------------------------


class TestRunSuite2p:
    @pytest.fixture()
    def imaging(self):
        return _make_test_imaging()

    @pytest.fixture()
    def fake_inner(self, monkeypatch):
        return _install_fake_inner(monkeypatch)

    def test_all_epochs_concatenates_selected_epochs(self, imaging, fake_inner, tmp_path):
        bin_calls, detect_calls = fake_inner
        rois = run_segmentation(imaging, method="suite2p", folder=tmp_path / "seg", scope="all_epochs")

        assert isinstance(rois, Suite2pRois)
        assert len(bin_calls) == 1
        assert bin_calls[0]["epoch_indices"] == [0, 1]
        assert bin_calls[0]["yrange"] == [0, 4]
        assert bin_calls[0]["xrange"] == [0, 5]
        assert bin_calls[0]["badframes"] is None
        assert len(detect_calls) == 1
        assert detect_calls[0]["mov_shape"] == (5, 4, 5)
        assert rois.get_num_rois() == 1

    def test_per_epoch_returns_container(self, imaging, fake_inner, tmp_path):
        bin_calls, detect_calls = fake_inner
        rois_by_epoch = run_segmentation(
            imaging,
            method="suite2p",
            folder=tmp_path / "seg",
            scope="per_epoch",
            badframes=[np.array([False, True]), np.array([True, False, False])],
        )

        assert isinstance(rois_by_epoch, EpochSegmentations)
        assert rois_by_epoch.epoch_indices == [0, 1]
        assert isinstance(rois_by_epoch[0], Suite2pRois)
        assert isinstance(rois_by_epoch[1], Suite2pRois)
        assert len(bin_calls) == 2
        assert bin_calls[0]["epoch_indices"] == [0]
        assert bin_calls[1]["epoch_indices"] == [1]
        np.testing.assert_array_equal(bin_calls[0]["badframes"], np.array([False, True]))
        np.testing.assert_array_equal(bin_calls[1]["badframes"], np.array([True, False, False]))
        assert detect_calls[0]["mov_shape"] == (2, 4, 5)
        assert detect_calls[1]["mov_shape"] == (3, 4, 5)

    def test_output_folder_round_trip(self, imaging, fake_inner, tmp_path):
        folder = tmp_path / "seg"
        rois = run_segmentation(imaging, method="suite2p", folder=folder)

        plane_folder = folder / "segmentation_output" / "plane0"
        assert Suite2pRois(plane_folder).get_num_rois() == rois.get_num_rois()
        reloaded = read_segmentation_output(folder)
        assert isinstance(reloaded, Suite2pRois)
        np.testing.assert_array_equal(reloaded.get_roi_image_masks().todense(), rois.get_roi_image_masks().todense())

    def test_all_epochs_rejects_mismatched_epoch_ranges(self, imaging, fake_inner, tmp_path):
        with pytest.raises(ValueError, match="yrange differs across selected epochs"):
            run_segmentation(
                imaging, method="suite2p", folder=tmp_path / "seg", scope="all_epochs", yrange=[[0, 4], [1, 4]]
            )

    def test_all_epochs_accepts_per_epoch_badframes(self, imaging, fake_inner, tmp_path):
        bin_calls, _ = fake_inner
        run_segmentation(
            imaging,
            method="suite2p",
            folder=tmp_path / "seg",
            scope="all_epochs",
            badframes=[np.array([False, True]), np.array([True, False, False])],
        )

        np.testing.assert_array_equal(bin_calls[0]["badframes"], np.array([False, True, True, False, False]))


class TestRunSuite2pUnmocked:
    """Run suite2p's own detection end-to-end (no mocks)."""

    def test_sparsery_with_defaults(self, tmp_path):
        # The only test where suite2p's detection runs: the others mock _detect_rois_from_mov,
        # and the cellpose test below fails earlier, at settings validation.
        _, imaging, _ = generate_imaging_with_rois(
            num_frames=300,
            height=64,
            width=64,
            num_rois=10,
            radius_range=(4, 6),
            decay_time=2.0,
            event_rate=1.0,
            seed=0,
        )

        rois = run_segmentation(imaging, method="suite2p", folder=tmp_path / "seg")

        assert rois.get_num_rois() > 0

    def test_cellpose_rejects_zero_diameter(self):
        # suite2p 1.x has no diameter=0 auto-estimate; it crashes inside roi_detect.
        _, imaging, _ = generate_imaging_with_rois(
            num_frames=300, height=64, width=64, num_rois=10, radius_range=(4, 6), seed=0
        )

        with pytest.raises(ValidationError, match="diameter"):
            run_segmentation(imaging, method="suite2p", settings={"algorithm": "cellpose", "diameter": [0, 0]})


# ---------------------------------------------------------------------------
# _stream_bin_movie
# ---------------------------------------------------------------------------


class TestStreamBinMovie:
    """Exercise the streaming binner end-to-end (no mocks)."""

    def test_bin_size_one_returns_full_movie_concatenated(self):
        ep0 = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5, 1)
        ep1 = (np.arange(3 * 4 * 5, dtype=np.uint16) + 1000).reshape(3, 4, 5, 1)
        imaging = NumpyImaging([ep0, ep1], sampling_frequency=7.0)

        mov = seg._stream_bin_movie(
            imaging,
            plane_index=0,
            epoch_indices=[0, 1],
            bin_size=1,
            yrange=[0, 4],
            xrange=[0, 5],
            badframes=None,
            nbins=5000,
        )

        expected = np.concatenate([ep0[:, :, :, 0], ep1[:, :, :, 0]], axis=0).astype(np.float32)
        np.testing.assert_array_equal(mov, expected)

    def test_yrange_xrange_crops_output(self):
        ep0 = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5, 1)
        imaging = NumpyImaging([ep0], sampling_frequency=7.0)

        mov = seg._stream_bin_movie(
            imaging,
            plane_index=0,
            epoch_indices=[0],
            bin_size=1,
            yrange=[1, 3],
            xrange=[2, 4],
            badframes=None,
            nbins=5000,
        )

        expected = ep0[:, 1:3, 2:4, 0].astype(np.float32)
        np.testing.assert_array_equal(mov, expected)


# ---------------------------------------------------------------------------
# Suite2pSegmentationSettings
# ---------------------------------------------------------------------------


class TestMergedDefaults:
    def test_overlays_overrides_on_suite2p_defaults(self):
        cfg = seg.Suite2pSegmentationSettings(algorithm="sourcery", threshold_scaling=2.5)
        merged = cfg.merged_with_suite2p_defaults()
        assert merged["algorithm"] == "sourcery"
        assert merged["threshold_scaling"] == 2.5
        # Suite2p-only sub-dicts come through untouched
        assert "sparsery_settings" in merged
        assert "sourcery_settings" in merged
        assert "cellpose_settings" in merged


class TestDiameter:
    def test_accepts_diameter_of_at_least_one(self):
        assert seg.Suite2pSegmentationSettings(diameter=[1, 12]).diameter == [1.0, 12.0]

    @pytest.mark.parametrize("diameter", [[0, 0], [0, 12], [12, -1]])
    def test_rejects_diameter_below_one(self, diameter):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            seg.Suite2pSegmentationSettings(diameter=diameter)
