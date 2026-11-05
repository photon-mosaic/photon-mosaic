import json

import numpy as np
import pytest
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from photon_mosaic.core import NumpyImaging
from photon_mosaic.extractors import Suite2pRois
from photon_mosaic.segmentation import (
    BaseSegmentation,
    EpochSegmentations,
    basesegmentation,
    read_segmentation_output,
    register_segmentation_class,
    run_segmentation,
)


class DummySettings(BaseSettings):
    num_rois: int = Field(default=2, description="Number of single-pixel ROIs to emit.")
    fail: bool = Field(default=False, description="Raise inside run.")


class DummySegmentation(BaseSegmentation):
    """Backend emitting single-pixel ROIs, without any optional dependency.

    A span starting at epoch ``i`` gets ``num_rois + i`` ROIs, so per-epoch results can be told apart.
    """

    method_name = "dummy"
    settings_class = DummySettings

    @classmethod
    def is_installed(cls):
        return True

    @classmethod
    def get_method_version(cls):
        return "0.0.0"

    @classmethod
    def run(cls, imaging, epoch_indices, yrange, xrange, badframes, settings):
        if settings.fail:
            raise RuntimeError("dummy failure")
        H, W, _ = imaging.shape
        num_rois = settings.num_rois + epoch_indices[0]
        stats = [{"ypix": np.array([i]), "xpix": np.array([0]), "lam": np.array([1.0])} for i in range(num_rois)]
        return Suite2pRois.from_stat(stats, shape=(H, W, 1), sampling_frequency=imaging.sampling_frequency)


@pytest.fixture(autouse=True)
def dummy_method(monkeypatch):
    monkeypatch.setitem(basesegmentation._registered_segmentation_classes, "dummy", DummySegmentation)


@pytest.fixture()
def imaging():
    epoch0 = np.ones((2, 4, 5, 1), dtype=np.float32)
    epoch1 = np.full((3, 4, 5, 1), 2.0, dtype=np.float32)
    return NumpyImaging([epoch0, epoch1], sampling_frequency=7.0)


def test_unknown_method_lists_available(imaging, tmp_path):
    with pytest.raises(ValueError, match="Available methods: .*'dummy'.*'suite2p'"):
        run_segmentation(imaging, method="nope", folder=tmp_path / "seg")


def test_register_requires_method_name():
    class Nameless(BaseSegmentation):
        pass

    with pytest.raises(ValueError, match="non-empty 'method_name'"):
        register_segmentation_class(Nameless)


def test_params_override_settings(imaging, tmp_path):
    rois = run_segmentation(imaging, method="dummy", folder=tmp_path / "seg", settings={"num_rois": 1}, num_rois=3)

    assert rois.get_num_rois() == 3
    params = json.loads((tmp_path / "seg" / "segmentation_params.json").read_text())
    assert params["method"] == "dummy"
    assert params["settings"] == {"num_rois": 3, "fail": False}


def test_unknown_param_rejected(imaging, tmp_path):
    with pytest.raises(ValueError, match="Unknown parameters \\['num_roi'\\]"):
        run_segmentation(imaging, method="dummy", folder=tmp_path / "seg", num_roi=3)


def test_invalid_param_value_rejected(imaging, tmp_path):
    with pytest.raises(ValidationError):
        run_segmentation(imaging, method="dummy", folder=tmp_path / "seg", num_rois="many")


def test_folder_contents_and_imaging_registered(imaging, tmp_path):
    folder = tmp_path / "seg"
    rois = run_segmentation(imaging, method="dummy", folder=folder, badframes=np.zeros(5, dtype=bool))

    assert rois.imaging is imaging
    assert (folder / "segmentation_params.json").is_file()
    assert (folder / "badframes.npz").is_file()
    assert (folder / "segmentation_output" / "metadata.json").is_file()
    log = json.loads((folder / "segmentation_log.json").read_text())
    assert log["method_version"] == "0.0.0"
    assert log["error"] is False
    assert log["run_time"] is not None


def test_registered_imaging_provenance_is_json(imaging, tmp_path):
    from photon_mosaic.preprocessing import RegisterImaging
    from photon_mosaic.preprocessing.suite2p_registration import Suite2PMotion

    motion = Suite2PMotion(imaging=imaging, displacements=[np.zeros((2, 1, 2)), np.zeros((3, 1, 2))], ops={})
    registered = RegisterImaging(imaging, motion)

    run_segmentation(registered, method="dummy", folder=tmp_path / "seg")

    assert (tmp_path / "seg" / "imaging.json").is_file()


def test_imaging_provenance_falls_back_to_pickle(imaging, tmp_path, monkeypatch):
    def failing_dump_to_json(*args, **kwargs):
        raise TypeError("Object of type Foo is not JSON serializable")

    monkeypatch.setattr(imaging, "dump_to_json", failing_dump_to_json)

    rois = run_segmentation(imaging, method="dummy", folder=tmp_path / "seg")

    assert rois.get_num_rois() == 2
    assert not (tmp_path / "seg" / "imaging.json").exists()
    assert (tmp_path / "seg" / "imaging.pickle").is_file()


def test_existing_folder(imaging, tmp_path):
    folder = tmp_path / "seg"
    folder.mkdir()
    with pytest.raises(FileExistsError):
        run_segmentation(imaging, method="dummy", folder=folder)

    rois = run_segmentation(imaging, method="dummy", folder=folder, remove_existing_folder=True)
    assert rois.get_num_rois() == 2


def test_error_is_logged(imaging, tmp_path):
    folder = tmp_path / "seg"
    with pytest.raises(RuntimeError, match="dummy failure"):
        run_segmentation(imaging, method="dummy", folder=folder, fail=True)
    log = json.loads((folder / "segmentation_log.json").read_text())
    assert log["error"] is True
    assert "dummy failure" in log["error_trace"]

    result = run_segmentation(
        imaging, method="dummy", folder=folder, fail=True, remove_existing_folder=True, raise_error=False
    )
    assert result is None


@pytest.mark.parametrize("scope", ["all_epochs", "per_epoch"])
def test_read_segmentation_output_round_trip(imaging, tmp_path, scope):
    folder = tmp_path / "seg"
    result = run_segmentation(imaging, method="dummy", folder=folder, scope=scope)
    reloaded = read_segmentation_output(folder, imaging=imaging)

    if scope == "all_epochs":
        pairs = [(result, reloaded)]
    else:
        assert isinstance(result, EpochSegmentations)
        assert isinstance(reloaded, EpochSegmentations)
        assert reloaded.epoch_indices == result.epoch_indices == [0, 1]
        assert [result[i].get_num_rois() for i in (0, 1)] == [2, 3]
        pairs = [(result[i], reloaded[i]) for i in (0, 1)]

    for original, loaded in pairs:
        assert isinstance(original, Suite2pRois)
        assert loaded.imaging is imaging
        np.testing.assert_array_equal(
            np.asarray(loaded.get_roi_image_masks().todense()), np.asarray(original.get_roi_image_masks().todense())
        )


@pytest.mark.parametrize("save", [True, False])
def test_segmentation_info_is_attached_and_saved(imaging, tmp_path, save):
    folder = tmp_path / "seg" if save else None
    rois = run_segmentation(imaging, method="dummy", folder=folder, num_rois=3)

    info = rois.segmentation_info
    assert info["params"] == {"num_rois": 3, "fail": False}
    assert info["log"]["method"] == "dummy"
    assert info["log"]["error"] is False
    assert info["imaging"]["class"].endswith("NumpyImaging")

    if save:
        assert read_segmentation_output(folder).segmentation_info == info


def test_no_folder_returns_rois_and_writes_nothing(imaging, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rois = run_segmentation(imaging, method="dummy", folder=None)

    assert isinstance(rois, Suite2pRois)
    assert rois.imaging is imaging
    assert list(tmp_path.iterdir()) == []


def test_registered_imaging_defaults_to_motion_ranges_and_badframes_of_selected_epochs(imaging, tmp_path):
    from photon_mosaic.core import Motion
    from photon_mosaic.preprocessing import RegisterImaging

    motion = Motion(
        imaging=imaging,
        displacements=[np.zeros((2, 1, 2)), np.zeros((3, 1, 2))],
        yranges=[[(0, 4)], [(1, 3)]],
        xranges=[[(0, 5)], [(2, 4)]],
        corrected_badframes=[np.array([True, False]), np.array([False, True, False])],
    )
    registered = RegisterImaging(imaging, motion)
    folder = tmp_path / "seg"

    run_segmentation(registered, method="dummy", folder=folder, scope="per_epoch", epoch_indices=[1])

    params = json.loads((folder / "segmentation_params.json").read_text())
    assert params["run_kwargs"]["yrange"] == [[1, 3]]
    assert params["run_kwargs"]["xrange"] == [[2, 4]]
    saved = np.load(folder / "badframes.npz")
    assert list(saved) == ["epoch1"]
    np.testing.assert_array_equal(saved["epoch1"], [False, True, False])


def test_registered_imaging_without_motion_ranges_falls_back_to_full_frame(imaging, tmp_path):
    from photon_mosaic.core import Motion
    from photon_mosaic.preprocessing import RegisterImaging

    motion = Motion(imaging=imaging, displacements=[np.zeros((2, 1, 2)), np.zeros((3, 1, 2))])
    run_segmentation(RegisterImaging(imaging, motion), method="dummy", folder=tmp_path / "seg")
    assert not (tmp_path / "seg" / "badframes.npz").exists()
