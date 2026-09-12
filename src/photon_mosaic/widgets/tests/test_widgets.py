import threading
from unittest.mock import patch

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from photon_mosaic.core.generators import generate_random_imaging, generate_rois  # noqa: E402
from photon_mosaic.widgets.rois import RoisWidget  # noqa: E402
from photon_mosaic.widgets.series import ImagingSeriesWidget  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_imaging():
    return generate_random_imaging(num_frames=50, height=32, width=32, seed=42)


@pytest.fixture
def small_rois(small_imaging):
    rois = generate_rois(num_rois=3, height=32, width=32, sampling_frequency=30.0, seed=42)
    rois.register_imaging(small_imaging)
    return rois


@pytest.fixture
def small_rois_sparse(small_imaging):
    rois = generate_rois(num_rois=3, height=32, width=32, sampling_frequency=30.0, seed=42, sparse=True)
    rois.register_imaging(small_imaging)
    return rois


@pytest.fixture
def small_rois_no_imaging():
    return generate_rois(num_rois=3, height=32, width=32, sampling_frequency=30.0, seed=42)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_exports():
    from photon_mosaic.widgets import plot_imaging_series, plot_rois

    assert plot_rois is RoisWidget
    assert plot_imaging_series is ImagingSeriesWidget


# ---------------------------------------------------------------------------
# RoisWidget – matplotlib backend
# ---------------------------------------------------------------------------


class TestRoisWidgetMatplotlib:
    def test_basic(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, backend="matplotlib")
        assert w.figure is not None
        assert w.ax is not None
        assert "n=3" in w.ax.get_title()

    def test_with_sparse_masks(self, small_rois_sparse, small_imaging):
        """Sparse-backed ROIs should densify internally and plot the same as dense ones."""
        w = RoisWidget(small_rois_sparse, imaging=small_imaging, backend="matplotlib")
        assert w.figure is not None
        images = w.ax.get_images()
        assert len(images) >= 2

    def test_with_background(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, backend="matplotlib")
        # There should be at least 2 imshow calls: background + overlay
        images = w.ax.get_images()
        assert len(images) >= 2

    def test_without_explicit_imaging_uses_registered(self, small_rois):
        """When no imaging arg is passed, widget uses rois.imaging (registered)."""
        w = RoisWidget(small_rois, backend="matplotlib")
        images = w.ax.get_images()
        assert len(images) >= 2

    def test_show_roi_ids(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, show_roi_ids=True, backend="matplotlib")
        texts = w.ax.texts
        assert len(texts) == 3  # one label per ROI

    def test_custom_colormap_string(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, colors="viridis", backend="matplotlib")
        assert w.figure is not None

    def test_custom_alpha(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, alpha=0.8, backend="matplotlib")
        assert w.data_plot["alpha"] == 0.8

    def test_data_plot_contents(self, small_rois, small_imaging):
        w = RoisWidget(small_rois, imaging=small_imaging, backend="matplotlib")
        dp = w.data_plot
        assert dp["num_rois"] == 3
        assert len(dp["roi_ids"]) == 3
        assert dp["alpha"] == 0.5
        assert dp["show_roi_ids"] is False
        assert dp["background"] is not None
        assert dp["background"].shape[:2] == (32, 32)


# ---------------------------------------------------------------------------
# ImagingSeriesWidget – __init__ tests (no plotting)
# ---------------------------------------------------------------------------


class TestImagingSeriesWidgetInit:
    def _make_widget(self, imaging, **kwargs):
        """Create widget without triggering plot backend (ipywidgets-only widget)."""
        with patch.object(ImagingSeriesWidget, "check_backend", return_value="ipywidgets"):
            return ImagingSeriesWidget(imaging, immediate_plot=False, **kwargs)

    def test_single_view(self, small_imaging):
        w = self._make_widget(small_imaging)
        dp = w.data_plot
        assert dp["is_multi_view"] is False
        assert dp["num_frames"] == 50
        assert dp["view_names"] == ["imaging"]
        assert len(dp["times"]) == 50

    def test_multi_view(self, small_imaging):
        imaging_dict = {"view_a": small_imaging, "view_b": small_imaging}
        w = self._make_widget(imaging_dict)
        dp = w.data_plot
        assert dp["is_multi_view"] is True
        assert set(dp["view_names"]) == {"view_a", "view_b"}

    def test_multi_view_mismatched_frames_raises(self):
        img_a = generate_random_imaging(num_frames=50, height=16, width=16, seed=1)
        img_b = generate_random_imaging(num_frames=30, height=16, width=16, seed=2)
        with pytest.raises(ValueError, match="same number of frames"):
            self._make_widget({"a": img_a, "b": img_b})

    def test_frame_index_clamped_high(self, small_imaging):
        w = self._make_widget(small_imaging, frame_index=9999)
        assert w.data_plot["frame_index"] == 49  # clamped to num_frames - 1

    def test_frame_index_clamped_low(self, small_imaging):
        w = self._make_widget(small_imaging, frame_index=-5)
        assert w.data_plot["frame_index"] == 0

    def test_time_range_defaults_to_full(self, small_imaging):
        w = self._make_widget(small_imaging)
        dp = w.data_plot
        np.testing.assert_allclose(dp["time_range"][0], dp["times"][0])
        np.testing.assert_allclose(dp["time_range"][1], dp["times"][-1])

    def test_custom_time_range(self, small_imaging):
        w = self._make_widget(small_imaging, time_range=(0.5, 1.0))
        assert w.data_plot["time_range"] == (0.5, 1.0)

    def test_colormap_and_percentiles(self, small_imaging):
        w = self._make_widget(small_imaging, colormap="viridis", vmin_percentile=5.0, vmax_percentile=95.0)
        dp = w.data_plot
        assert dp["colormap"] == "viridis"
        assert dp["vmin_percentile"] == 5.0
        assert dp["vmax_percentile"] == 95.0

    def test_frame_rate_from_imaging(self, small_imaging):
        w = self._make_widget(small_imaging)
        assert w.data_plot["frame_rate"] == small_imaging.sampling_frequency

    def test_plot_ipywidgets_clamps_contrast_sample_to_num_frames(self):
        # Contrast sampling must request at most num_frames, not a hardcoded 100.
        tiny = generate_random_imaging(num_frames=10, height=16, width=16, seed=0)
        captured = []
        sentinel = RuntimeError("stop after recording")

        def recording_get_series(*args, **kwargs):
            end_frame = args[1] if len(args) >= 2 else kwargs.get("end_frame")
            captured.append(end_frame)
            raise sentinel  # short-circuit before matplotlib/ipywidgets setup

        tiny.get_series = recording_get_series

        with (
            patch.object(ImagingSeriesWidget, "check_backend", return_value="ipywidgets"),
            patch("spikeinterface.widgets.utils_ipywidgets.check_ipywidget_backend"),
        ):
            with pytest.raises(RuntimeError, match="stop after recording"):
                ImagingSeriesWidget(tiny, immediate_plot=True)

        assert captured == [10], f"expected widget to clamp end_frame to num_frames=10; calls were {captured}"


# ---------------------------------------------------------------------------
# ImagingSeriesWidget -- playback loop pacing
# ---------------------------------------------------------------------------


class _FakeButton:
    def __init__(self):
        self.description = ""
        self.button_style = ""


class _FakeSlider:
    def __init__(self, value=0):
        self.value = value


class _PlaybackHarness:
    """Minimal stand-in exposing exactly what _playback_loop/_stop_playback touch, so the
    pacing logic can be exercised without going through __init__/plot_ipywidgets at all."""

    def __init__(self, num_frames, playback_fps, current_frame=0, last_time=0.0):
        self.data_plot = {"num_frames": num_frames}
        self.playback_fps = playback_fps
        self.current_frame = current_frame
        self.is_playing = True
        self.frame_slider = _FakeSlider(current_frame)
        self.play_button = _FakeButton()
        self._playback_last_time = last_time  # normally set by _start_playback/_on_fps_changed
        self._playback_timing_lock = threading.Lock()

    _playback_loop = ImagingSeriesWidget._playback_loop
    _stop_playback = ImagingSeriesWidget._stop_playback
    _on_fps_changed = ImagingSeriesWidget._on_fps_changed


def test_playback_loop_skips_ahead_after_slow_iteration(monkeypatch):
    """A slow redraw (simulated by a big wall-clock jump between poll ticks) should make the
    loop jump straight to the frame matching elapsed time, not silently fall one frame at a
    time -- see _playback_loop's docstring for why a naive `current_frame += 1` on a fixed
    timer drops frames unpredictably instead."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    sleep_calls = []

    def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) == 1:
            fake_time[0] += 1.0  # simulate one slow redraw stalling a full second
        else:
            harness.is_playing = False  # stop right after observing the post-stall frame

    monkeypatch.setattr("time.sleep", fake_sleep)

    harness._playback_loop()

    # at playback_fps=10, a 1s stall should skip straight to frame 10, not crawl to frame 1
    assert harness.current_frame == 10
    assert harness.frame_slider.value == 10
    # poll interval is 1 / (4 * playback_fps)
    assert sleep_calls[0] == pytest.approx(0.025)


def test_playback_loop_stops_at_last_frame(monkeypatch):
    """Reaching the final frame should stop playback (button reset to Play)."""
    harness = _PlaybackHarness(num_frames=5, playback_fps=10, current_frame=3)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    sleep_calls = []

    def fake_sleep(duration):
        sleep_calls.append(duration)
        fake_time[0] += 1.0  # always enough to reach the end in one jump

    monkeypatch.setattr("time.sleep", fake_sleep)

    harness._playback_loop()

    assert harness.current_frame == 4  # num_frames - 1
    assert harness.is_playing is False
    assert harness.play_button.description == "▶ Play"
    assert len(sleep_calls) == 1


def test_playback_loop_does_not_drift_under_irregular_polling(monkeypatch):
    """Advancing the pacing reference to `now` on every frame-advance (instead of by the
    exact duration of the frames just consumed) discards whatever fraction of a frame period
    was left over -- silently running playback slower than the requested fps, with the gap
    growing the longer it plays. A poll interval that doesn't evenly divide the frame period
    (mirroring irregular real-world redraw timing) exposes this: fails against the pre-fix
    code (lands on frame 99, not 101)."""
    fps = 10
    poll_step = 0.017  # deliberately does not evenly divide the frame period (0.1s)
    num_polls = 600
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=fps)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    calls = []

    def fake_sleep(duration):
        calls.append(duration)
        fake_time[0] += poll_step
        if len(calls) >= num_polls:
            harness.is_playing = False

    monkeypatch.setattr("time.sleep", fake_sleep)

    harness._playback_loop()

    assert harness.current_frame == 101


def test_fps_change_does_not_retroactively_apply_to_elapsed_time(monkeypatch):
    """_on_fps_changed must reset the pacing reference; otherwise real time that already
    elapsed under the old fps gets reinterpreted under the new one on the next _playback_loop
    check, producing an incorrect frame jump right at the moment of the change (e.g. 0.1s
    accrued at 10 fps, worth 1 frame, becomes worth 2 frames if read back at 20 fps)."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    fake_time[0] = 0.1  # 0.1s accrued under the old fps=10 (exactly 1 frame's worth)
    harness._on_fps_changed({"new": 20})

    calls = []

    def fake_sleep(duration):
        calls.append(duration)
        harness.is_playing = False  # stop right after the first check

    monkeypatch.setattr("time.sleep", fake_sleep)

    harness._playback_loop()

    # without the reset, this would advance by int(0.1 * 20) = 2 frames instead of 0
    assert harness.current_frame == 0


class _BlockingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_first_enter = True

    def __enter__(self):
        self._lock.acquire()
        if self.block_first_enter:
            self.block_first_enter = False
            self.entered.set()
            assert self.release.wait(timeout=1), "timed out waiting to release playback timing lock"
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


class _TimeAdvancingLock:
    def __init__(self, fake_time, advanced_to):
        self._lock = threading.Lock()
        self.fake_time = fake_time
        self.advanced_to = advanced_to
        self.advanced = False

    def __enter__(self):
        self._lock.acquire()
        if not self.advanced:
            self.fake_time[0] = self.advanced_to
            self.advanced = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


class _SleepFpsLock:
    def __init__(self, harness, updated_fps):
        self._lock = threading.Lock()
        self.harness = harness
        self.updated_fps = updated_fps
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        if self.enter_count == 2:
            self.harness.playback_fps = self.updated_fps
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


def test_playback_loop_samples_time_inside_timing_lock(monkeypatch):
    """The elapsed-time sample should be taken after the timing lock is acquired."""
    fake_time = [0.0]
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, last_time=0.0)
    harness._playback_timing_lock = _TimeAdvancingLock(fake_time, advanced_to=0.1)

    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    monkeypatch.setattr("time.sleep", lambda duration: setattr(harness, "is_playing", False))

    harness._playback_loop()

    assert harness.current_frame == 1


def test_playback_loop_rechecks_fps_before_sleep(monkeypatch):
    """A just-applied FPS change should affect the same iteration's sleep pacing."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, last_time=0.0)
    harness._playback_timing_lock = _SleepFpsLock(harness, updated_fps=20)

    fake_time = [0.1]
    sleep_calls = []

    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    def fake_sleep(duration):
        sleep_calls.append(duration)
        harness.is_playing = False

    monkeypatch.setattr("time.sleep", fake_sleep)

    harness._playback_loop()

    assert harness.current_frame == 1
    assert sleep_calls == [pytest.approx(0.0125)]


def test_fps_change_waits_for_playback_timing_lock(monkeypatch):
    """Playback updates and FPS changes should serialize through the same timing lock."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, last_time=0.0)
    harness._playback_timing_lock = _BlockingLock()

    fake_time = [0.1]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    monkeypatch.setattr("time.sleep", lambda duration: setattr(harness, "is_playing", False))

    playback_thread = threading.Thread(target=harness._playback_loop)
    playback_thread.start()

    assert harness._playback_timing_lock.entered.wait(timeout=1), "playback loop never acquired timing lock"

    fps_change_finished = threading.Event()

    def change_fps():
        harness._on_fps_changed({"new": 20})
        fps_change_finished.set()

    fps_thread = threading.Thread(target=change_fps)
    fps_thread.start()

    assert not fps_change_finished.wait(timeout=0.05), "fps change should block on playback timing lock"

    harness._playback_timing_lock.release.set()
    playback_thread.join(timeout=1)
    fps_thread.join(timeout=1)

    assert not playback_thread.is_alive()
    assert not fps_thread.is_alive()
    assert harness.current_frame == 1
    assert harness.playback_fps == 20
    assert harness._playback_last_time == pytest.approx(0.1)
