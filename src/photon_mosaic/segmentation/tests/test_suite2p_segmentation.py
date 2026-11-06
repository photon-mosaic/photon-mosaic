import numpy as np
import pytest
from pydantic import ValidationError

from photon_mosaic.core import NumpyImaging, load, split_imaging_by_epochs
from photon_mosaic.core.generators import generate_imaging_with_rois
from photon_mosaic.extractors import Suite2pRois
from photon_mosaic.segmentation import (
    run_segmentation,
)
from photon_mosaic.segmentation import suite2p_segmentation as seg


def _install_fake_inner(monkeypatch):
    """Mock the streaming binner and the algorithm dispatcher.

    Returns ``(bin_calls, detect_calls)`` so tests can assert which frames,
    ranges, and badframes the public API forwarded to each layer.
    """
    bin_calls: list[dict] = []
    detect_calls: list[dict] = []

    def fake_stream_bin_movie(
        imaging,
        *,
        bin_size,
        yrange,
        xrange,
        badframes,
        nbins,
    ):
        bin_calls.append(
            {
                "num_frames": imaging.get_total_frames(),
                "yrange": list(yrange),
                "xrange": list(xrange),
                "badframes": None if badframes is None else np.array(badframes, copy=True),
                "bin_size": bin_size,
                "nbins": nbins,
            }
        )
        n = imaging.get_total_frames()
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
                # fields suite2p's roi_stats computes and its cell classifier reads
                "npix_norm": 1.0,
                "compact": 1.0,
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

    def test_concatenates_all_epochs(self, imaging, fake_inner, tmp_path):
        bin_calls, detect_calls = fake_inner
        rois = run_segmentation(imaging, method="suite2p", folder=tmp_path / "seg")

        assert isinstance(rois, Suite2pRois)
        assert len(bin_calls) == 1
        assert bin_calls[0]["num_frames"] == 5
        assert bin_calls[0]["yrange"] == [0, 4]
        assert bin_calls[0]["xrange"] == [0, 5]
        assert bin_calls[0]["badframes"] is None
        assert len(detect_calls) == 1
        assert detect_calls[0]["mov_shape"] == (5, 4, 5)
        assert rois.get_num_rois() == 1

    def test_split_epochs_returns_one_rois_per_epoch(self, imaging, fake_inner, tmp_path):
        bin_calls, detect_calls = fake_inner
        rois_by_epoch = [
            run_segmentation(epoch, method="suite2p", folder=None) for epoch in split_imaging_by_epochs(imaging)
        ]

        assert all(isinstance(rois, Suite2pRois) for rois in rois_by_epoch)
        assert [call["num_frames"] for call in bin_calls] == [2, 3]
        assert detect_calls[0]["mov_shape"] == (2, 4, 5)
        assert detect_calls[1]["mov_shape"] == (3, 4, 5)

    def test_output_folder_round_trip(self, imaging, fake_inner, tmp_path):
        folder = tmp_path / "seg"
        rois = run_segmentation(imaging, method="suite2p", folder=folder)

        assert isinstance(rois, Suite2pRois)
        reloaded = load(folder / "rois")
        np.testing.assert_array_equal(reloaded.get_roi_image_masks().todense(), rois.get_roi_image_masks().todense())
        # suite2p's scalar stat fields are saved as ROI properties
        np.testing.assert_array_equal(reloaded.get_property("npix"), rois.get_property("npix"))
        # cell classifier output, under the names Suite2pRois uses when loading iscell.npy
        assert rois.get_property("iscell").dtype == bool
        for key in ("iscell", "iscell_probability"):
            np.testing.assert_array_equal(reloaded.get_property(key), rois.get_property(key))

    def test_rejects_multi_plane_imaging(self, fake_inner):
        two_planes = NumpyImaging([np.ones((3, 4, 5, 2), dtype=np.float32)], sampling_frequency=7.0)
        with pytest.raises(ValueError, match="does not support multi-plane imaging"):
            run_segmentation(two_planes, method="suite2p", folder=None)

    def test_all_epochs_rejects_mismatched_epoch_ranges(self, imaging, fake_inner, tmp_path):
        with pytest.raises(ValueError, match="yrange differs across epochs"):
            run_segmentation(imaging, method="suite2p", folder=tmp_path / "seg", yrange=[[0, 4], [1, 4]])

    def test_all_epochs_accepts_per_epoch_badframes(self, imaging, fake_inner, tmp_path):
        bin_calls, _ = fake_inner
        run_segmentation(
            imaging,
            method="suite2p",
            folder=tmp_path / "seg",
            badframes=[np.array([False, True]), np.array([True, False, False])],
        )

        np.testing.assert_array_equal(bin_calls[0]["badframes"], np.array([False, True, True, False, False]))

    def test_bin_size_setting_is_used(self, imaging, fake_inner):
        bin_calls, _ = fake_inner
        run_segmentation(imaging, method="suite2p", folder=None, bin_size=2)
        assert bin_calls[0]["bin_size"] == 2

    def test_unsupported_setting_raises(self, imaging, fake_inner):
        with pytest.raises(ValueError, match="Not supported"):
            run_segmentation(imaging, method="suite2p", folder=None, chan2_threshold=0.5)


class TestRunSuite2pUnmocked:
    """Run suite2p's own detection end-to-end (no mocks)."""

    @pytest.fixture(scope="class")
    def imaging(self):
        # 10 ROIs; the other run tests mock _detect_rois_from_mov
        return generate_imaging_with_rois(
            num_frames=300,
            height=64,
            width=64,
            num_rois=10,
            radius_range=(4, 6),
            decay_time=2.0,
            event_rate=1.0,
            seed=0,
        )[1]

    def test_sparsery_with_defaults(self, imaging, tmp_path):
        assert run_segmentation(imaging, method="suite2p", folder=tmp_path / "seg").get_num_rois() > 0

    def test_max_rois_reaches_sparsery(self, imaging):
        assert 0 < run_segmentation(imaging, method="suite2p", folder=None, max_ROIs=3).get_num_rois() <= 3

    def test_npix_norm_max_reaches_roi_stats(self, imaging):
        rois = run_segmentation(imaging, method="suite2p", folder=None, npix_norm_max=1.0)
        assert (rois.get_property("npix_norm") <= 1.0).all()

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


class TestSettings:
    def test_env_var_reaches_a_setting(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SUITE2P_SEGMENTATION_SPATIAL_SCALE", "1")
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("UNRELATED_KEY=1\n")  # a .env file is not read
        assert seg.Suite2pSegmentationSettings().spatial_scale == 1

    def test_unknown_name_raises(self):
        with pytest.raises(ValidationError):
            seg.Suite2pSegmentationSettings(sparsery_settings={"spatial_scale": 1})


class TestDiameter:
    def test_accepts_diameter_of_at_least_one(self):
        assert seg.Suite2pSegmentationSettings(diameter=[1, 12]).diameter == [1.0, 12.0]

    @pytest.mark.parametrize("diameter", [[0, 0], [0, 12], [12, -1]])
    def test_rejects_diameter_below_one(self, diameter):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            seg.Suite2pSegmentationSettings(diameter=diameter)
