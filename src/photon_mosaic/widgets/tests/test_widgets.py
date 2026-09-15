import threading
from types import SimpleNamespace
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
    """Mimics just enough of an ipywidgets trait to make .observe(...) fire on real changes,
    like the real frame_slider does when _playback_loop or a user seek sets .value."""

    def __init__(self, value=0):
        self._value = value
        self._observers = []

    def observe(self, handler, names="value"):
        self._observers.append(handler)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if new_value == self._value:
            return
        old_value = self._value
        self._value = new_value
        for handler in self._observers:
            handler({"new": new_value, "old": old_value, "name": "value"})


class _PlaybackHarness:
    """Minimal stand-in exposing exactly what _playback_loop/_stop_playback touch, so the
    pacing logic can be exercised without going through __init__/plot_ipywidgets at all."""

    def __init__(self, num_frames, playback_fps, current_frame=0, last_time=0.0):
        self.data_plot = {"num_frames": num_frames}
        self.playback_fps = playback_fps
        self.current_frame = current_frame
        self.is_playing = True
        self.frame_slider = _FakeSlider(current_frame)
        self.frame_slider.observe(self._on_frame_changed, names="value")
        self.play_button = _FakeButton()
        self._playback_last_time = last_time  # normally set by _start_playback/_on_fps_changed
        self._playback_timing_lock = threading.RLock()
        self._playback_start_id = 0
        self._frame_generation = 0
        self._slider_write_state = threading.local()
        self._render_lock = threading.Lock()
        self._playback_wake_event = threading.Event()
        self.play_thread = None
        self.display_calls = 0

    def _update_display(self):
        self.display_calls += 1

    _on_play_button_clicked = ImagingSeriesWidget._on_play_button_clicked
    _start_playback = ImagingSeriesWidget._start_playback
    _playback_loop = ImagingSeriesWidget._playback_loop
    _stop_playback = ImagingSeriesWidget._stop_playback
    _on_fps_changed = ImagingSeriesWidget._on_fps_changed
    _on_frame_changed = ImagingSeriesWidget._on_frame_changed
    _publish_current_frame_to_slider = ImagingSeriesWidget._publish_current_frame_to_slider
    seek_to_frame = ImagingSeriesWidget.seek_to_frame


def test_playback_loop_skips_ahead_after_slow_iteration(monkeypatch):
    """A slow redraw (simulated by a big wall-clock jump between poll ticks) should make the
    loop jump straight to the frame matching elapsed time, not silently fall one frame at a
    time -- see _playback_loop's docstring for why a naive `current_frame += 1` on a fixed
    timer drops frames unpredictably instead."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    sleep_calls = []

    def fake_wait(timeout=None):
        sleep_calls.append(timeout)
        if len(sleep_calls) == 1:
            fake_time[0] += 1.0  # simulate one slow redraw stalling a full second
        else:
            harness.is_playing = False  # stop right after observing the post-stall frame

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    # at playback_fps=10, a 1s stall should skip straight to frame 10, not crawl to frame 1
    assert harness.current_frame == 10
    assert harness.frame_slider.value == 10
    # poll interval is 1 / (4 * playback_fps)
    assert sleep_calls[0] == pytest.approx(0.025)


def test_play_button_reuses_existing_alive_playback_thread(monkeypatch):
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10)
    harness.is_playing = False

    class _ExistingThread:
        def __init__(self):
            self.started = False

        def is_alive(self):
            return True

        def start(self):
            self.started = True

    existing_thread = _ExistingThread()
    harness.play_thread = existing_thread

    monkeypatch.setattr("time.monotonic", lambda: 1.23)
    monkeypatch.setattr(
        "threading.Thread",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected new playback thread")),
    )

    harness._on_play_button_clicked(None)

    assert harness.is_playing is True
    assert harness._playback_last_time == pytest.approx(1.23)
    assert harness.play_thread is existing_thread
    assert existing_thread.started is False


def test_start_playback_spawns_and_starts_a_real_thread():
    """Every other test routes around actually spawning a thread (via stand-ins like
    _ExistingThread above) to stay fast and deterministic -- this confirms the real path (no
    existing play_thread) actually works: a genuine threading.Thread gets created and started,
    running _playback_loop for real."""
    # Already at the last frame, so the spawned _playback_loop exits (and stops itself)
    # almost immediately instead of needing to be pieced apart with mocked timing.
    harness = _PlaybackHarness(num_frames=5, playback_fps=10, current_frame=4)

    # Capture play_thread while still holding the same (reentrant) lock _start_playback uses
    # internally: the spawned thread's first action is acquiring that same lock, so it can't
    # reach the end-of-loop cleanup that clears play_thread until this block releases it --
    # an unsynchronized read right after _start_playback() returns would otherwise race the
    # spawned thread's cleanup (current_frame is already the last frame, so it can finish
    # almost immediately).
    with harness._playback_timing_lock:
        harness._start_playback()
        spawned_thread = harness.play_thread

    assert isinstance(spawned_thread, threading.Thread)
    spawned_thread.join(timeout=1)
    assert not spawned_thread.is_alive()
    assert harness.play_thread is None  # cleared by the loop's own cleanup on exit
    assert harness.is_playing is False  # the loop reached the end and stopped itself


def test_start_playback_wakes_a_sleeping_worker_promptly(monkeypatch):
    """A pause-then-play that reuses a still-alive worker (rather than spawning a new one)
    must wake it promptly if it's currently blocked in its poll wait -- otherwise the button
    shows Pause immediately but playback stays visibly unresponsive until that wait happens to
    elapse on its own (up to a full poll interval -- worse at a low fps, where each interval is
    longer)."""
    import time

    harness = _PlaybackHarness(num_frames=10_000, playback_fps=0.01, current_frame=0, last_time=0.0)
    # 1 / (4 * 0.01) = 25s poll interval -- if a resume had to wait that out, this test would
    # time out; the wake mechanism should make it return almost immediately instead.
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # elapsed always 0 -> loop always waits

    # Count the worker's own poll-wait calls directly (not e.g. time.monotonic() calls, which
    # _start_playback also makes itself for an unrelated reason -- resetting the pacing
    # reference -- and would otherwise be indistinguishable from the worker actually reacting).
    wait_calls = []
    original_wait = harness._playback_wake_event.wait

    def counting_wait(timeout=None):
        wait_calls.append(timeout)
        return original_wait(timeout=timeout)

    harness._playback_wake_event.wait = counting_wait

    worker = threading.Thread(target=harness._playback_loop, daemon=True)
    harness.play_thread = worker
    worker.start()

    # Let it settle into its poll wait, then confirm it's genuinely stuck there for a while --
    # not just about to check in again on its own -- well under the 25s it would otherwise be
    # blocked for.
    deadline = time.perf_counter() + 1.0
    while len(wait_calls) < 1 and time.perf_counter() < deadline:
        time.sleep(0.01)
    calls_before = len(wait_calls)
    time.sleep(0.1)
    assert len(wait_calls) == calls_before, "worker should still be in its poll wait, not looping"
    assert worker.is_alive()

    harness._start_playback()

    # The worker should react almost immediately -- well under the 25s poll interval -- not
    # need to wait out the rest of it, i.e. return from its current wait and start a new one.
    deadline = time.perf_counter() + 1.0
    while len(wait_calls) == calls_before and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert len(wait_calls) > calls_before, "worker never woke from its poll wait"

    # Let the worker actually stop instead of looping forever in the background.
    harness.is_playing = False
    harness._playback_wake_event.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_on_frame_changed_wakes_a_sleeping_worker_promptly(monkeypatch):
    """A real seek during active playback (e.g. dragging the frame slider onto the last frame,
    which should stop playback immediately) must wake a worker that's currently sleeping
    between polls -- otherwise it won't notice until that sleep happens to elapse on its own."""
    import time

    harness = _PlaybackHarness(num_frames=10_000, playback_fps=0.01, current_frame=0, last_time=0.0)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # elapsed always 0 -> loop always waits

    wait_calls = []
    original_wait = harness._playback_wake_event.wait

    def counting_wait(timeout=None):
        wait_calls.append(timeout)
        return original_wait(timeout=timeout)

    harness._playback_wake_event.wait = counting_wait

    worker = threading.Thread(target=harness._playback_loop, daemon=True)
    harness.play_thread = worker
    worker.start()

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) < 1 and time.perf_counter() < deadline:
        time.sleep(0.01)
    calls_before = len(wait_calls)
    time.sleep(0.1)
    assert len(wait_calls) == calls_before, "worker should still be in its poll wait, not looping"
    assert worker.is_alive()

    harness._on_frame_changed({"new": 5, "old": 0})

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) == calls_before and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert len(wait_calls) > calls_before, "worker never woke from its poll wait"

    harness.is_playing = False
    harness._playback_wake_event.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_seek_to_frame_wakes_a_sleeping_worker_promptly(monkeypatch):
    """Same missed-wakeup risk as a real slider seek: a programmatic seek during active
    playback must wake a worker that's currently sleeping between polls."""
    import time

    harness = _PlaybackHarness(num_frames=10_000, playback_fps=0.01, current_frame=0, last_time=0.0)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # elapsed always 0 -> loop always waits

    wait_calls = []
    original_wait = harness._playback_wake_event.wait

    def counting_wait(timeout=None):
        wait_calls.append(timeout)
        return original_wait(timeout=timeout)

    harness._playback_wake_event.wait = counting_wait

    worker = threading.Thread(target=harness._playback_loop, daemon=True)
    harness.play_thread = worker
    worker.start()

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) < 1 and time.perf_counter() < deadline:
        time.sleep(0.01)
    calls_before = len(wait_calls)
    time.sleep(0.1)
    assert len(wait_calls) == calls_before, "worker should still be in its poll wait, not looping"
    assert worker.is_alive()

    harness.seek_to_frame(5)

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) == calls_before and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert len(wait_calls) > calls_before, "worker never woke from its poll wait"

    harness.is_playing = False
    harness._playback_wake_event.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_on_fps_changed_wakes_a_sleeping_worker_promptly(monkeypatch):
    """Raising the FPS while the worker is asleep in its poll wait must take effect
    immediately -- otherwise the worker keeps waiting out the *old*, slower interval it's
    already blocked in, and the new rate doesn't apply until that old wait happens to elapse
    on its own (up to a full poll interval, worse at a low fps)."""
    import time

    harness = _PlaybackHarness(num_frames=10_000, playback_fps=0.01, current_frame=0, last_time=0.0)
    # 1 / (4 * 0.01) = 25s poll interval -- if the fps bump had to wait that out, this test
    # would time out; the wake mechanism should make it return almost immediately instead.
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # elapsed always 0 -> loop always waits

    wait_calls = []
    original_wait = harness._playback_wake_event.wait

    def counting_wait(timeout=None):
        wait_calls.append(timeout)
        return original_wait(timeout=timeout)

    harness._playback_wake_event.wait = counting_wait

    worker = threading.Thread(target=harness._playback_loop, daemon=True)
    harness.play_thread = worker
    worker.start()

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) < 1 and time.perf_counter() < deadline:
        time.sleep(0.01)
    calls_before = len(wait_calls)
    time.sleep(0.1)
    assert len(wait_calls) == calls_before, "worker should still be in its poll wait, not looping"
    assert worker.is_alive()

    harness._on_fps_changed({"new": 5.0, "old": 0.01})

    deadline = time.perf_counter() + 1.0
    while len(wait_calls) == calls_before and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert len(wait_calls) > calls_before, "worker never woke from its poll wait"

    harness.is_playing = False
    harness._playback_wake_event.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


class _StartPlaybackDuringPreSleepGapLock:
    """Simulates a concurrent _start_playback() landing in the gap between the post-publish
    recheck and the sleep-decision checkpoint -- acquisition #3 with the harness setup used
    below (fps=10, fake time constant so no frame ever advances and no publish call happens
    each iteration, keeping the per-iteration acquisition count at exactly 3: top-of-loop,
    post-publish recheck, sleep-decision)."""

    def __init__(self, harness):
        self._lock = threading.RLock()
        self.harness = harness
        self.enter_count = 0
        self.injected = False

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enter_count == 3 and not self.injected:
            self.injected = True
            self.harness._playback_start_id += 1
            self.harness._playback_wake_event.set()
        self._lock.release()


def test_playback_loop_skips_the_wait_when_a_restart_lands_just_before_it(monkeypatch):
    """A _start_playback() landing in the gap between the post-publish recheck and the
    sleep-decision checkpoint already set the wake event and moved the start id on -- but the
    worker isn't asleep yet at that point, so there's nothing for that set() to interrupt, and
    the clear() right before waiting would otherwise wipe it for nothing. The sleep-decision
    checkpoint must notice the start id changed and skip the wait entirely instead, looping
    back to reprocess with fresh state rather than sleeping through a restart that just
    happened."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, current_frame=0, last_time=0.0)
    lock = _StartPlaybackDuringPreSleepGapLock(harness)
    harness._playback_timing_lock = lock
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # no advance -> no publish call

    wait_calls = []
    enter_count_at_wait = []

    def fake_wait(timeout=None):
        wait_calls.append(timeout)
        enter_count_at_wait.append(lock.enter_count)
        harness.is_playing = False  # stop right after the first real wait

    harness._playback_wake_event.wait = fake_wait

    harness._playback_loop()

    # Not just "one wait call" -- that alone can't tell a skipped-then-waited pass apart from
    # an immediate one, since either way the loop stops right after the first real wait. The
    # lock's own acquisition count at that moment does: reaching the wait after skipping one
    # full pass (top-of-loop, post-publish recheck, sleep-decision) takes 6 more acquisitions
    # than the initial start-id capture (7 total); reaching it on the very first pass -- the
    # restart's wake-up silently lost -- would take only 3 (4 total).
    assert len(wait_calls) == 1
    assert enter_count_at_wait == [7], "did not skip the pass where the restart landed"


class _SeekDuringPreSleepGapLock:
    """Simulates a concurrent seek (_on_frame_changed / seek_to_frame) landing in the gap
    between the post-publish recheck and the sleep-decision checkpoint -- acquisition #3 with
    the harness setup used below, same as _StartPlaybackDuringPreSleepGapLock. A seek bumps
    _frame_generation, not _playback_start_id -- exercising the branch of the skip-the-wait
    check that a pure restart injection can't."""

    def __init__(self, harness):
        self._lock = threading.RLock()
        self.harness = harness
        self.enter_count = 0
        self.injected = False

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enter_count == 3 and not self.injected:
            self.injected = True
            self.harness._frame_generation += 1
            self.harness._playback_wake_event.set()
        self._lock.release()


def test_playback_loop_skips_the_wait_when_a_seek_lands_just_before_it(monkeypatch):
    """A seek landing in the gap between the post-publish recheck and the sleep-decision
    checkpoint already set the wake event and bumped _frame_generation -- but not
    _playback_start_id, since it isn't a restart. The sleep-decision checkpoint must notice
    _frame_generation changed (not just _playback_start_id) and skip the wait entirely,
    looping back to reprocess with fresh state instead of sleeping through the seek for a full
    poll interval (e.g. a seek onto the last frame would otherwise leave playback looking
    active for that long)."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, current_frame=0, last_time=0.0)
    lock = _SeekDuringPreSleepGapLock(harness)
    harness._playback_timing_lock = lock
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # no advance -> no publish call

    wait_calls = []
    enter_count_at_wait = []

    def fake_wait(timeout=None):
        wait_calls.append(timeout)
        enter_count_at_wait.append(lock.enter_count)
        harness.is_playing = False  # stop right after the first real wait

    harness._playback_wake_event.wait = fake_wait

    harness._playback_loop()

    # Same reasoning as the restart test: reaching the wait after skipping one full pass takes
    # 6 more acquisitions than the initial capture (7 total); reaching it on the very first
    # pass -- the seek's wake-up silently lost -- would take only 3 (4 total).
    assert len(wait_calls) == 1
    assert enter_count_at_wait == [7], "did not skip the pass where the seek landed"


def test_playback_loop_clears_wake_event_before_waiting_not_after(monkeypatch):
    """The wake event must be cleared before the wait begins, not after the wait returns.
    Clearing it afterward would let a _start_playback() landing in the gap between a timed-out
    wait and that trailing clear() have its wake-up silently consumed by the clear instead of
    arming the *next* wait -- delaying the restart for a full poll interval anyway, exactly the
    unresponsiveness this event exists to avoid."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, current_frame=0, last_time=0.0)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)  # elapsed always 0 -> loop always waits

    call_order = []
    original_clear = harness._playback_wake_event.clear

    def recording_clear():
        call_order.append("clear")
        original_clear()

    def recording_wait(timeout=None):
        call_order.append("wait")
        harness.is_playing = False  # stop right after the first wait, don't actually block
        return True

    harness._playback_wake_event.clear = recording_clear
    harness._playback_wake_event.wait = recording_wait

    harness._playback_loop()

    assert call_order == ["clear", "wait"]


def test_playback_loop_initializes_missing_last_time(monkeypatch):
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, last_time=None)

    fake_time = [0.0]
    sleep_calls = []
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    def fake_wait(timeout=None):
        sleep_calls.append(timeout)
        harness.is_playing = False

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    assert harness.current_frame == 0
    assert harness._playback_last_time == pytest.approx(0.0)
    assert sleep_calls == [pytest.approx(0.025)]


def test_playback_loop_stops_at_last_frame(monkeypatch):
    """Reaching the final frame should stop playback (button reset to Play)."""
    harness = _PlaybackHarness(num_frames=5, playback_fps=10, current_frame=3)

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    sleep_calls = []

    def fake_wait(timeout=None):
        sleep_calls.append(timeout)
        fake_time[0] += 1.0  # always enough to reach the end in one jump

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    assert harness.current_frame == 4  # num_frames - 1
    assert harness.is_playing is False
    assert harness.play_button.description == "▶ Play"
    assert len(sleep_calls) == 1


def test_playback_loop_stops_even_when_no_longer_the_registered_play_thread(monkeypatch):
    """A worker that reaches the end must still signal a stop even if it's no longer
    self.play_thread (e.g. superseded by a restart) -- the cleanup block's restart/no-restart
    decision is scoped to "am I still registered", but reaching the end is a fact about this
    worker's own progress, not about that registration."""
    harness = _PlaybackHarness(num_frames=5, playback_fps=10, current_frame=3)
    harness.play_thread = object()  # anything that isn't threading.current_thread()

    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    def fake_wait(timeout=None):
        fake_time[0] += 1.0  # always enough to reach the end in one jump

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    assert harness.current_frame == 4
    assert harness.is_playing is False
    assert harness.play_button.description == "▶ Play"


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

    def fake_wait(timeout=None):
        calls.append(timeout)
        fake_time[0] += poll_step
        if len(calls) >= num_polls:
            harness.is_playing = False

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

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

    def fake_wait(timeout=None):
        calls.append(timeout)
        harness.is_playing = False  # stop right after the first check

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    # without the reset, this would advance by int(0.1 * 20) = 2 frames instead of 0
    assert harness.current_frame == 0


def test_fps_change_clamps_non_positive_values(monkeypatch):
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10)

    monkeypatch.setattr("time.monotonic", lambda: 1.23)

    harness._on_fps_changed({"new": 0})

    assert harness.playback_fps == pytest.approx(0.1)
    assert harness._playback_last_time == pytest.approx(1.23)


class _BlockingLock:
    """Blocks on the loop's *second* acquisition -- the actual top-of-loop checkpoint (reads
    fake_time, playback_fps, advances current_frame) -- not the first, which is now
    _playback_loop's own _playback_start_id capture and does nothing time/fps-related. Blocking
    on #1 instead would release the real top-of-loop computation into an unforced race against
    the concurrent fps-change thread, making the test's current_frame/playback_fps assertions
    scheduler-dependent instead of deterministic."""

    def __init__(self):
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        if self.enter_count == 2:
            self.entered.set()
            assert self.release.wait(timeout=1), "timed out waiting to release playback timing lock"
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


class _TimeAdvanceThenFpsChangeLock:
    """Combines two lock-ordering checks in one: advances the fake clock on the loop's first
    checkpoint acquisition (proving _playback_loop samples time.monotonic() only after
    acquiring the lock, not before), then changes playback_fps on the second (proving the loop
    re-reads fps fresh for the sleep duration rather than reusing the value already captured
    earlier in the same iteration for the elapsed-time computation). Acquisition #1 is
    _playback_loop's own _playback_start_id capture, before the loop's checkpoints begin."""

    def __init__(self, harness, fake_time, advanced_to, updated_fps):
        self._lock = threading.Lock()
        self.harness = harness
        self.fake_time = fake_time
        self.advanced_to = advanced_to
        self.updated_fps = updated_fps
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        if self.enter_count == 2:
            self.fake_time[0] = self.advanced_to
        elif self.enter_count == 3:
            self.harness.playback_fps = self.updated_fps
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


class _CleanupSeekLock:
    """Injects a seek exactly as the cleanup section acquires the lock, for the exact harness
    setup used below (num_frames=10, current_frame=8, fps=10, last_time=0.0, fake_time=0.1).
    That's acquisition #6 with the current _playback_loop structure: the _playback_start_id
    capture, the top-of-loop check, the publish helper's own capture and recheck (it acquires
    the same lock twice), the post-write is_playing/reached_last_frame recheck, then the
    cleanup section itself -- recount (e.g. via a lock that prints enter_count and the calling
    line) if _playback_loop's or _publish_current_frame_to_slider's lock usage changes."""

    def __init__(self, harness):
        self._lock = threading.RLock()
        self.harness = harness
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        if self.enter_count == 6:
            self.harness.current_frame = 2
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()


def test_playback_loop_uses_fresh_state_at_each_checkpoint(monkeypatch):
    """_playback_loop must sample fresh state at each checkpoint under the lock, not a value
    read before acquiring it or captured earlier in the same iteration:

    - time.monotonic() is sampled only after the lock is acquired -- a lock-triggered clock
      advance on the very first acquisition should already be visible to that same read.
    - the fps used for the sleep duration is re-read fresh, not the value already captured
      earlier in the iteration for the elapsed-time computation -- a lock-triggered fps change
      partway through should still affect that same iteration's sleep.
    """
    fake_time = [0.0]
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, last_time=0.0)
    harness._playback_timing_lock = _TimeAdvanceThenFpsChangeLock(harness, fake_time, advanced_to=0.1, updated_fps=20)

    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    sleep_calls = []

    def fake_wait(timeout=None):
        sleep_calls.append(timeout)
        harness.is_playing = False

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    assert harness.current_frame == 1  # 0.1s elapsed at the original fps (10) -> 1 frame
    assert sleep_calls == [pytest.approx(0.0125)]  # 1 / (4 * the new fps, 20)


def test_fps_change_waits_for_playback_timing_lock(monkeypatch):
    """Playback updates and FPS changes should serialize through the same timing lock."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10, last_time=0.0)
    harness._playback_timing_lock = _BlockingLock()

    fake_time = [0.1]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    monkeypatch.setattr(
        harness._playback_wake_event, "wait", lambda timeout=None: setattr(harness, "is_playing", False)
    )

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


class _AlwaysAliveThread:
    """Stand-in for play_thread so _start_playback's alive-check finds an existing thread and
    doesn't spawn a real one -- this test is only about the is_playing/button transition."""

    def is_alive(self):
        return True


class _PausingButton:
    """Pauses the first time .description is set to `pause_on_value`, until released. Used to
    check whether the state lock is still held at exactly that write -- a concurrent lock
    acquisition attempt made while paused here can only succeed if the lock had already been
    released before this write, which is exactly the bug this guards against."""

    def __init__(self, pause_on_value):
        self._pause_on_value = pause_on_value
        self._paused_once = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.description = ""
        self.button_style = ""

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "description" and value == getattr(self, "_pause_on_value", None) and not self._paused_once:
            self._paused_once = True
            self.entered.set()
            assert self.release.wait(timeout=1), "timed out waiting to release the button write"


def test_stop_playback_button_update_is_atomic_with_state():
    """_stop_playback (called by the worker thread when it reaches the last frame) must not
    leave a window where is_playing is already False but the button hasn't been updated yet --
    a concurrent Play click could otherwise see the flag, start new playback, only for this
    call's own (still-pending) button write to overwrite it back to "Play" right after."""
    harness = _PlaybackHarness(num_frames=10_000, playback_fps=10)
    harness.play_thread = _AlwaysAliveThread()
    harness.is_playing = True
    harness.play_button = _PausingButton(pause_on_value="▶ Play")
    harness.play_button.description = "⏸ Pause"
    harness.play_button.button_style = "warning"

    stop_thread = threading.Thread(target=harness._stop_playback)
    stop_thread.start()

    assert harness.play_button.entered.wait(timeout=1), "_stop_playback never reached the button-reset write"

    click_finished = threading.Event()

    def click():
        harness._on_play_button_clicked(None)
        click_finished.set()

    click_thread = threading.Thread(target=click)
    click_thread.start()

    # If the state lock is still held at this exact write (the fix), the click's own lock
    # acquisition must still be blocked. If it isn't (the bug), the click can slip in here and
    # start playback, only to have this call's still-pending button write clobber it right after.
    assert not click_finished.wait(timeout=0.05), "play click completed while the stop transition was still in progress"

    harness.play_button.release.set()
    stop_thread.join(timeout=1)
    click_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert not click_thread.is_alive()
    assert harness.is_playing is True
    assert harness.play_button.description == "⏸ Pause"
    assert harness.play_button.button_style == "warning"


def test_publish_current_frame_write_does_not_clobber_a_concurrent_seek():
    """A publish's own write to frame_slider.value fires _on_frame_changed just like a real
    seek would. If a genuine seek lands before that write executes, the publish's write must
    not undo it -- the thread-local publishing flag (set here exactly as
    _publish_current_frame_to_slider sets it) tells _on_frame_changed this is its own write,
    not an independent new seek."""
    # current_frame=5: what a publish call captured under its lock, about to be (belatedly)
    # written to the slider.
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=5)

    # A real user seek arrives before the publish call gets around to writing frame=5.
    harness.frame_slider.value = 7
    assert harness.current_frame == 7

    # The stale publish write happens next, exactly as _publish_current_frame_to_slider does it.
    harness._slider_write_state.publishing = True
    try:
        harness.frame_slider.value = 5
    finally:
        harness._slider_write_state.publishing = False

    assert harness.current_frame == 7  # not clobbered back to the stale value 5


class _SlidersPublishSeekLock:
    """Injects a real seek exactly as _publish_current_frame_to_slider captures its
    frame/generation snapshot -- the gap right before its own (about to become stale) slider
    write. That's acquisition #3 with the current _playback_loop structure: the
    _playback_start_id capture, the top-of-loop check, then the publish helper's own capture --
    recount (e.g. via a lock that prints enter_count) if that structure changes."""

    def __init__(self, harness, seek_to):
        self._lock = threading.RLock()
        self.harness = harness
        self.seek_to = seek_to
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enter_count == 3:  # right after the publish call's frame/generation snapshot
            self.harness.frame_slider.value = self.seek_to
        self._lock.release()


def test_playback_loop_converges_slider_after_a_seek_races_the_write(monkeypatch):
    """_publish_current_frame_to_slider captures current_frame under the lock, but the actual
    slider write happens after releasing it -- a real seek can still land in that gap, bumping
    _frame_generation. The publish call must detect that its captured generation is now stale
    and republish the latest current_frame, or the widget visibly shows the wrong frame."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=9, last_time=0.0)
    harness._playback_timing_lock = _SlidersPublishSeekLock(harness, seek_to=3)

    monkeypatch.setattr("time.monotonic", lambda: 0.1)  # 1 frame elapsed at fps=10

    def fake_wait(timeout=None):
        harness.is_playing = False

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    harness._playback_loop()

    assert harness.current_frame == 3  # the seek's target, never clobbered
    assert harness.frame_slider.value == 3  # converged -- not left at the stale write (10)


def test_on_frame_changed_resets_pacing_reference_for_a_real_seek(monkeypatch):
    """A user seek during active playback must reset _playback_last_time -- otherwise the next
    poll counts time elapsed since the last *playback* advance (which predates the seek) and
    immediately jumps forward again from the newly seeked position."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, last_time=0.0)

    monkeypatch.setattr("time.monotonic", lambda: 5.0)  # a real seek happens 5s after the last advance
    harness.frame_slider.value = 42

    assert harness.current_frame == 42
    assert harness._playback_last_time == pytest.approx(5.0)
    assert harness.display_calls == 1


def test_seek_to_frame_resets_pacing_reference(monkeypatch):
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, last_time=0.0)

    monkeypatch.setattr("time.monotonic", lambda: 7.5)

    harness.seek_to_frame(12)

    assert harness.current_frame == 12
    assert harness.frame_slider.value == 12
    assert harness._playback_last_time == pytest.approx(7.5)


def test_seek_to_frame_redraws_even_when_already_at_the_target_frame():
    """Seeking to the frame already shown (frame_slider.value already equals frame_number) must
    still redraw. Assigning a trait its current value is a no-op in ipywidgets/traitlets -- no
    observer fires, so _publish_current_frame_to_slider's usual redraw path (_on_frame_changed,
    triggered by the slider write) never runs unless it's handled explicitly."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=5)
    assert harness.frame_slider.value == 5  # already at the target -- the write below is a no-op

    harness.seek_to_frame(5)

    assert harness.display_calls == 1


class _WorkerAdvanceDuringSeekLock:
    """Simulates the playback worker advancing current_frame in the gap between
    seek_to_frame's state-setting lock and its slider publish."""

    def __init__(self, harness, advance_to):
        self._lock = threading.RLock()
        self.harness = harness
        self.advance_to = advance_to
        self.enter_count = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enter_count == 1:  # right after seek_to_frame's own state-setting lock
            self.harness.current_frame = self.advance_to
            self.harness._frame_generation += 1
        self._lock.release()


def test_seek_to_frame_does_not_roll_back_a_concurrent_playback_advance():
    """seek_to_frame's slider write is a publish, not a raw write with the seek's own captured
    frame_number -- if the playback worker advances current_frame in the gap between
    seek_to_frame's locked write and its publish call, the publish must converge to the
    worker's newer state, not roll it back to the (by then stale) seek target."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=5)
    harness._playback_timing_lock = _WorkerAdvanceDuringSeekLock(harness, advance_to=8)

    harness.seek_to_frame(3)

    assert harness.current_frame == 8  # the worker's newer advance, not rolled back to 3
    assert harness.frame_slider.value == 8


class _PausingBeforeObserversSlider(_FakeSlider):
    """Like _FakeSlider, but pauses after storing the new value and before notifying observers
    -- used to hold a write "in flight" on its own thread so something else can be attempted
    while it's paused there. Only its first real value-change pauses; later writes (e.g. a
    seek's own reassertion publish, or a second publish) proceed immediately -- otherwise two
    writes through the same instance would each wait on the same one-shot release."""

    def __init__(self, value=0):
        super().__init__(value)
        self.entered = threading.Event()
        self.release = threading.Event()
        self._paused_once = False

    @_FakeSlider.value.setter
    def value(self, new_value):
        if new_value == self._value:
            return
        old_value = self._value
        self._value = new_value
        if not self._paused_once:
            self._paused_once = True
            self.entered.set()
            assert self.release.wait(timeout=1), "timed out waiting to release the slider write"
        for h in self._observers:
            h({"new": new_value, "old": old_value, "name": "value"})


def test_publishing_flag_is_thread_local_not_a_shared_flag():
    """A publish's own write must only look like "internal" on the thread that's actually
    doing it. If the flag were a plain shared attribute instead of thread-local, a genuine
    seek from another thread landing while the publish thread's write is in flight would be
    misread as that publish's own callback and silently dropped -- current_frame/pacing would
    never be updated for a real user seek."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=5, last_time=0.0)
    harness.frame_slider = _PausingBeforeObserversSlider(0)  # differs from current_frame=5, so
    # the publish's write below is a real change, not a same-value no-op
    harness.frame_slider.observe(harness._on_frame_changed, names="value")

    publish_thread = threading.Thread(target=harness._publish_current_frame_to_slider)
    publish_thread.start()

    assert harness.frame_slider.entered.wait(timeout=1), "publish thread never reached the slider write"

    # A genuine seek arrives on a different thread while the publish thread's write is paused,
    # mid-flight, with its own thread-local publishing flag still set to True.
    seek_finished = threading.Event()

    def seek():
        harness._on_frame_changed({"new": 42, "old": 5})
        seek_finished.set()

    seek_thread = threading.Thread(target=seek)
    seek_thread.start()
    seek_thread.join(timeout=1)

    assert seek_finished.is_set(), "the concurrent seek never completed"
    assert harness.current_frame == 42  # not silently dropped

    harness.frame_slider.release.set()
    publish_thread.join(timeout=1)
    assert not publish_thread.is_alive()


def test_publish_does_not_finalize_stale_state_while_a_seek_is_mid_flight():
    """A genuine seek's raw value-store and its own observer running (the part that bumps
    _frame_generation and updates current_frame) aren't one atomic step from another thread's
    point of view -- ipywidgets/traitlets stores the new value, then notifies observers, as two
    separate steps. If a publish captures the pre-seek state, overwrites the slider back to it,
    and finishes its own generation recheck entirely inside that gap, the recheck finds nothing
    changed (the seek's own bump hasn't run yet) and stops -- leaving the slider showing the
    publish's stale value even after the seek's observer later runs and updates current_frame
    correctly."""
    harness = _PlaybackHarness(num_frames=1000, playback_fps=10, current_frame=5, last_time=0.0)
    harness.frame_slider = _PausingBeforeObserversSlider(5)
    harness.frame_slider.observe(harness._on_frame_changed, names="value")

    seek_thread = threading.Thread(target=lambda: setattr(harness.frame_slider, "value", 9))
    seek_thread.start()
    assert harness.frame_slider.entered.wait(timeout=1), "seek's slider write never paused"

    # A publish for the pre-seek frame (current_frame is still 5 -- the seek's own observer,
    # which would update it, hasn't run yet) completes entirely inside that gap.
    harness._publish_current_frame_to_slider()

    harness.frame_slider.release.set()
    seek_thread.join(timeout=1)

    assert harness.current_frame == 9
    assert harness.frame_slider.value == 9, "slider left showing the publish's stale value"


def test_playback_loop_rechecks_last_frame_after_internal_redraw(monkeypatch):
    harness = _PlaybackHarness(num_frames=10, playback_fps=10, current_frame=8, last_time=0.0)

    fake_time = [0.1]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    sleep_calls = []

    def fake_wait(timeout=None):
        sleep_calls.append(timeout)
        harness.is_playing = False

    monkeypatch.setattr(harness._playback_wake_event, "wait", fake_wait)

    def simulated_internal_redraw():
        harness.display_calls += 1
        with harness._playback_timing_lock:
            harness.current_frame = 2

    harness._update_display = simulated_internal_redraw

    harness._playback_loop()

    assert harness.current_frame == 2
    assert harness.play_button.description == ""
    assert sleep_calls == [pytest.approx(0.025)]


def test_playback_loop_revalidates_last_frame_before_cleanup_stop(monkeypatch):
    harness = _PlaybackHarness(num_frames=10, playback_fps=10, current_frame=8, last_time=0.0)
    harness.play_thread = threading.current_thread()
    harness._playback_timing_lock = _CleanupSeekLock(harness)

    fake_time = [0.1]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])
    monkeypatch.setattr(
        harness._playback_wake_event,
        "wait",
        lambda timeout=None: (_ for _ in ()).throw(AssertionError("unexpected wait")),
    )

    started_threads = []

    class _StubThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True
            started_threads.append(self)

    monkeypatch.setattr("threading.Thread", _StubThread)

    harness._playback_loop()

    assert harness.current_frame == 2
    assert harness.is_playing is True
    assert len(started_threads) == 1


def test_playback_loop_does_not_cancel_a_restart_that_races_its_own_end_decision(monkeypatch):
    """The worker can decide it reached the last frame, release the lock to publish, and only
    then -- before it reaches the cleanup section -- have the user pause and play again. Since
    the worker is still alive, _start_playback reuses it (no new thread) rather than replacing
    it. That must not let the worker's now-stale "I reached the end" decision cancel the fresh
    restart: is_playing must stay True, matching the button the user just clicked, not silently
    flip back to stopped -- and a fresh worker must actually be spawned to re-evaluate with
    current information, not just left dangling (is_playing True, no worker, playback
    genuinely stuck) once this stale worker exits."""
    harness = _PlaybackHarness(num_frames=5, playback_fps=10, current_frame=3, last_time=0.0)
    harness.play_thread = threading.current_thread()  # this call *is* the "spawned" worker

    fake_time = [1.0]  # elapsed=1.0 at fps=10 jumps straight to the last frame in one pass
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    started_threads = []

    class _StubThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon
            self.started = False

        def is_alive(self):
            return False  # never actually run, so a later is_alive() check finds it dead

        def start(self):
            self.started = True
            started_threads.append(self)

    monkeypatch.setattr("threading.Thread", _StubThread)

    restarted = [False]
    original_publish = harness._publish_current_frame_to_slider

    def publish_then_restart():
        original_publish()
        if not restarted[0]:
            restarted[0] = True
            # A pause-then-play lands in the gap between the worker's publish (outside the
            # lock) and its next checkpoint -- reusing this same still-alive worker, exactly as
            # _start_playback's own is_alive() check is meant to.
            harness._stop_playback()
            harness._start_playback()

    harness._publish_current_frame_to_slider = publish_then_restart

    harness._playback_loop()

    assert restarted[0], "the simulated restart never ran"
    assert harness.is_playing is True
    assert harness.play_button.description == "⏸ Pause"
    # Not just the flag/button: a fresh worker must actually be spawned to re-evaluate with
    # current information -- otherwise is_playing stays True with no worker left to advance
    # playback (or, from a later state where there's somewhere left to go, to actually get
    # there), which is just as broken as silently cancelling the restart.
    assert len(started_threads) == 1
    assert harness.play_thread is started_threads[0]


class _FakeImaging:
    def __init__(self, on_get_series=None):
        self.on_get_series = on_get_series
        self.calls = []

    def get_series(self, start, end, epoch_index=0):
        self.calls.append(start)
        if self.on_get_series:
            self.on_get_series(start)
        return np.zeros((1, 2, 2))


class _DisplayHarness:
    """Minimal stand-in exposing exactly what _update_display touches."""

    def __init__(self, num_frames, current_frame, on_get_series=None):
        imaging = _FakeImaging(on_get_series)
        self.data_plot = {
            "view_names": ["v"],
            "imaging_dict": {"v": imaging},
            "is_multi_view": False,
            "times": np.arange(num_frames, dtype=float),
            "epoch_index": 0,
        }
        self.axes = [_FakeAx()]
        self.images = {"v": _FakeImage()}
        self.global_vmin = {"v": 0.0}
        self.global_vmax = {"v": 1.0}
        self.vmin_slider = SimpleNamespace(value=0.0)
        self.vmax_slider = SimpleNamespace(value=100.0)
        self.colormap_dropdown = SimpleNamespace(value="gray")
        self.time_label = SimpleNamespace(value="")
        self.figure = _FakeFigure()
        self.current_frame = current_frame
        self._frame_generation = 0
        self._playback_timing_lock = threading.RLock()
        self._render_lock = threading.Lock()

    _update_display = ImagingSeriesWidget._update_display
    _update_time_label = ImagingSeriesWidget._update_time_label


class _FakeAx:
    def __init__(self):
        self.title = None

    def set_title(self, title):
        self.title = title


class _FakeImage:
    def __init__(self):
        self.clim = None
        self.data = None

    def set_data(self, data):
        self.data = data

    def set_cmap(self, cmap):
        pass

    def set_clim(self, vmin, vmax):
        self.clim = (vmin, vmax)


class _FakeFigure:
    def __init__(self):
        self.draw_count = 0
        self.canvas = SimpleNamespace(draw_idle=self._draw_idle)

    def _draw_idle(self):
        self.draw_count += 1


def test_update_display_does_not_tear_across_a_concurrent_frame_change():
    """current_frame is read at several points in _update_display (image data, title, time
    label) -- without a single captured snapshot, a concurrent change partway through could
    render one frame's pixels under another frame's title. Also verifies the render converges
    (redraws again) if current_frame moved on while the first pass was in flight."""

    def advance_during_first_fetch(frame_fetched):
        if frame_fetched == 5:  # only the very first get_series call, for frame 5
            harness.current_frame = 9
            harness._frame_generation += 1

    harness = _DisplayHarness(num_frames=20, current_frame=5, on_get_series=advance_during_first_fetch)

    harness._update_display()

    # Two passes: frame 5 (the snapshot the first pass captured, torn or not) is fetched, then
    # the convergence check finds the generation changed and redraws with the new frame, 9.
    assert harness.data_plot["imaging_dict"]["v"].calls == [5, 9]
    # The final, settled state is fully consistent with the latest frame, not torn.
    assert harness.axes[0].title == "Frame 9 | Time: 9.000s"
    assert harness.time_label.value == "Time: 9.000s / 19.00s"
    assert harness.figure.draw_count == 2


def test_update_display_does_not_hold_render_lock_during_frame_fetch():
    """imaging.get_series can be slow (disk/network I/O). _render_lock must only guard the
    actual Matplotlib mutations that follow it, not this fetch -- otherwise a stale render
    would block a concurrent one (or any other _render_lock holder) for the full I/O duration
    instead of just the brief mutation-and-draw that actually needs serializing."""
    lock_states_during_fetch = []

    def on_get_series(start):
        lock_states_during_fetch.append(harness._render_lock.locked())

    harness = _DisplayHarness(num_frames=20, current_frame=5, on_get_series=on_get_series)

    harness._update_display()

    assert lock_states_during_fetch == [False], "render lock was held during the frame fetch"


class _LockStateCheckingSlider:
    """Like SimpleNamespace(value=...), but records whether _render_lock was held every time
    .value is read -- used to prove display settings are read while genuinely holding the
    lock, not snapshotted earlier and merely applied under it."""

    def __init__(self, value, render_lock):
        self._value = value
        self._render_lock = render_lock
        self.lock_held_on_read = []

    @property
    def value(self):
        self.lock_held_on_read.append(self._render_lock.locked())
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


def test_update_display_reads_display_settings_while_holding_the_render_lock():
    """vmin/vmax must be read at mutation time, while genuinely holding _render_lock -- not
    snapshotted during the frame-data gather phase alongside frame_data, even though they're
    cheap synchronous reads like it (no I/O). Snapshotting them early would open a gap for a
    concurrent _on_display_changed render to apply newer settings, only for this call's stale
    snapshot to overwrite them right after -- invisible to the generation recheck, since
    display settings don't bump _frame_generation the way current_frame does. Reading them
    while the lock is held rules that out structurally: a concurrent render can't be in its own
    mutation phase at the same time, full stop."""
    harness = _DisplayHarness(num_frames=20, current_frame=5)
    harness.vmin_slider = _LockStateCheckingSlider(0.0, harness._render_lock)
    harness.vmax_slider = _LockStateCheckingSlider(100.0, harness._render_lock)

    harness._update_display()

    assert harness.vmin_slider.lock_held_on_read == [True]
    assert harness.vmax_slider.lock_held_on_read == [True]


def test_update_display_does_not_draw_a_torn_frame_from_a_concurrent_render():
    """The (frame, generation) snapshot at the top of _update_display keeps *one* call's own
    reads consistent, but that alone does nothing to stop a second, genuinely independent call
    -- e.g. a real seek's own _update_display() call (see _on_frame_changed), running
    concurrently with an in-flight render for a frame that's since gone stale -- from
    interleaving Matplotlib mutations on the same Axes/Image objects.

    Starting a second render and immediately releasing the first isn't enough to prove
    anything by itself -- the scheduler could just as easily let the first finish before the
    second gets any CPU time at all, in which case the two never actually overlap and the
    "no torn frame" assertion would pass without having exercised the interleaving it's meant
    to catch. So this proves serialization directly and deterministically instead of inferring
    it from an absence of tearing: while the first render is paused mid-write (confirmed via
    _render_lock.locked(), not assumed), the second is started and proven genuinely unable to
    finish for a bounded wait -- it can only be blocked trying to acquire the still-held lock,
    since nothing else in it can pause. Only after that positive proof of mutual exclusion does
    the test release the first, join both, and check the final render settled on the latest
    frame -- never a mix of one frame's pixels and another's title."""
    harness = _DisplayHarness(num_frames=20, current_frame=5)
    harness.data_plot["imaging_dict"]["v"].get_series = lambda start, end, epoch_index=0: np.full((1, 2, 2), start)

    paused = threading.Event()
    release = threading.Event()
    draws = []  # (image data value, title) captured at each draw_idle, in call order

    original_set_data = harness.images["v"].set_data
    pause_once = [True]  # only the very first call (the render this test forces to stall) pauses

    def pausing_set_data(data):
        original_set_data(data)
        if pause_once[0]:
            pause_once[0] = False
            paused.set()
            assert release.wait(timeout=1), "timed out waiting to resume the paused render"

    harness.images["v"].set_data = pausing_set_data

    original_draw_idle = harness.figure.canvas.draw_idle

    def recording_draw_idle():
        draws.append((harness.images["v"].data.flatten()[0], harness.axes[0].title))
        original_draw_idle()

    harness.figure.canvas.draw_idle = recording_draw_idle

    first_call = threading.Thread(target=harness._update_display)
    first_call.start()
    assert paused.wait(timeout=1), "first render never reached its own image write"
    assert harness._render_lock.locked(), "paused render should still be holding the render lock"

    # A second, independent render for a different frame -- as _on_frame_changed's own real-seek
    # branch would trigger -- starts on its own thread while the first is still paused mid-render.
    harness.current_frame = 9
    harness._frame_generation += 1
    second_finished = threading.Event()

    def run_second():
        harness._update_display()
        second_finished.set()

    second_call = threading.Thread(target=run_second)
    second_call.start()

    # Deterministic proof of serialization, not an inference from a lucky (or unlucky)
    # schedule: the second render has no pause of its own, so if it finishes this quickly it
    # can only be because it ran to completion unblocked -- exactly what the lock must prevent
    # while the first is still holding it.
    assert not second_finished.wait(timeout=0.2), "second render proceeded while the lock was held"

    release.set()
    first_call.join(timeout=1)
    second_call.join(timeout=1)

    assert not first_call.is_alive()
    assert not second_call.is_alive()
    torn = [(data, title) for data, title in draws if f"Frame {data}" not in title]
    assert not torn, f"drew a torn frame (image data vs. title mismatch): {torn}"
