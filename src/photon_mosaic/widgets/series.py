import numpy as np
from spikeinterface.widgets.base import BaseWidget, to_attr

from photon_mosaic.core.baseimaging import BaseImaging

_PLAYBACK_FPS_MIN = 0.1


class ImagingSeriesWidget(BaseWidget):
    """Widget for visualizing an ImagingExtractor series with interactive controls.

    This widget provides an interactive video player for imaging data using ipywidgets,
    following the SpikeInterface BaseWidget design pattern.

    Parameters
    ----------
    imaging : BaseImaging or dict[str, BaseImaging]
        The imaging extractor to visualize. Can be a single BaseImaging object or
        a dictionary of BaseImaging objects for synchronized multi-view display.
    epoch_index : int, optional
        Which segment to display, by default 0
    frame_index : int, optional
        Initial frame to display, by default 0
    time_range : tuple, optional
        Time range to display (start, end) in seconds, by default None (full range)
    colormap : str, optional
        Colormap for image display, by default 'gray'
    vmin_percentile : float, optional
        Minimum percentile for contrast (0-100), by default 2.0
    vmax_percentile : float, optional
        Maximum percentile for contrast (0-100), by default 98.0
    backend : str, optional
        Backend to use, by default None
    **backend_kwargs
        Additional backend-specific arguments
    """

    def __init__(
        self,
        imaging: BaseImaging | dict,
        epoch_index: int = 0,
        frame_index: int = 0,
        time_range: tuple | None = None,
        colormap: str = "gray",
        vmin_percentile: float = 2.0,
        vmax_percentile: float = 98.0,
        backend=None,
        **backend_kwargs,
    ):
        # Check if imaging is a dictionary
        if isinstance(imaging, dict):
            # Get the first imaging object to extract common properties
            first_key = list(imaging.keys())[0]
            first_imaging = imaging[first_key]

            # Validate all imaging objects have the same number of frames
            num_frames = first_imaging.get_num_samples(epoch_index)
            times = first_imaging.get_times(epoch_index)
            frame_rate = first_imaging.sampling_frequency

            for name, img in imaging.items():
                if img.get_num_samples(epoch_index) != num_frames:
                    raise ValueError(
                        f"All imaging objects must have the same number of frames. "
                        f"'{name}' has {img.get_num_samples(epoch_index)} frames, "
                        f"expected {num_frames}"
                    )

            is_multi_view = True
            imaging_dict = imaging
            view_names = list(imaging.keys())
        else:
            is_multi_view = False
            # Single imaging object - wrap in dict for consistent handling
            imaging_dict = {"imaging": imaging}
            view_names = ["imaging"]
            num_frames = imaging.get_num_samples(epoch_index)
            times = imaging.get_times(epoch_index)
            frame_rate = imaging.sampling_frequency

        # Validate parameters
        frame_index = max(0, min(frame_index, num_frames - 1))

        if time_range is None:
            time_range = (times[0], times[-1])

        # Prepare data for plotting
        data_plot = dict(
            imaging_dict=imaging_dict,
            view_names=view_names,
            is_multi_view=is_multi_view,
            epoch_index=epoch_index,
            num_frames=num_frames,
            times=times,
            frame_rate=frame_rate,
            frame_index=frame_index,
            time_range=time_range,
            colormap=colormap,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        """Interactive ipywidgets plot with video controls."""
        import threading

        import matplotlib.pyplot as plt
        from IPython.display import display
        from spikeinterface.widgets.utils_ipywidgets import check_ipywidget_backend

        check_ipywidget_backend()

        dp = to_attr(data_plot)

        # Store data for updates
        self.data_plot = data_plot
        self.current_frame = dp.frame_index
        self.is_playing = False
        self.play_thread = None
        self.playback_fps = min(10.0, dp.frame_rate)  # Default playback speed
        self._playback_last_time: float | None = None  # set by _start_playback/_on_fps_changed
        self._playback_timing_lock = threading.RLock()
        # Bumped under the lock on every _start_playback() call -- i.e. every fresh user
        # request to be playing. A worker captures this at its own start and, when it later
        # decides it reached the end, only honors that decision if no newer start has happened
        # meanwhile (see _playback_loop's cleanup section); otherwise a pause-then-play landing
        # in the gap between the worker's publish and its next checkpoint would have its own
        # stale end-of-playback decision silently cancel the fresh restart.
        self._playback_start_id = 0
        # Bumped under the lock every time current_frame is authoritatively set (a playback
        # advance, a real seek, or seek_to_frame), so _publish_current_frame_to_slider can tell
        # whether its own in-flight publish has gone stale and needs to republish the latest
        # state, instead of leaving the widget showing a value current_frame has moved past.
        self._frame_generation = 0
        # Thread-local: set around a publish's own write to frame_slider.value, so
        # _on_frame_changed can tell that write's own resulting observer call apart from a
        # genuine seek. Thread-local (not a shared flag) because a publish and a genuine seek
        # can only ever happen on two different threads (the playback worker vs. the kernel
        # thread), so one thread's in-flight publish can never be mistaken for the other's seek.
        self._slider_write_state = threading.local()
        # Serializes _update_display's actual Matplotlib mutations (set_data/set_title/
        # draw_idle, ...) across concurrent calls -- e.g. the playback worker's publish-echo
        # call and a genuine seek's own call (_on_frame_changed) can run concurrently on two
        # different threads. The (frame, generation) snapshot each call captures keeps *that*
        # call's own reads consistent, but doesn't stop two calls' writes to the same Axes/
        # Image objects from interleaving. Deliberately separate from _playback_timing_lock so
        # a slow render can't block state-changing callbacks (see _update_display).
        self._render_lock = threading.Lock()
        # _playback_loop waits on this between polls instead of a plain time.sleep(), so
        # _start_playback can wake a reused, still-sleeping worker immediately -- otherwise a
        # pause-then-play would show the Pause button right away but leave playback visibly
        # unresponsive until that poll happens to elapse on its own (up to a full poll
        # interval: 2.5s at the FPS slider's minimum, 0.1).
        self._playback_wake_event = threading.Event()

        # Sample up to 100 frames to compute a global vmin/vmax for the colormap.
        num_samples = min(100, dp.num_frames)

        # Store global vmin/vmax for each view
        self.global_vmin = {}
        self.global_vmax = {}

        for view_name, imaging in dp.imaging_dict.items():
            # TODO: get_random_frames instead
            sampled_data = imaging.get_series(0, num_samples, epoch_index=dp.epoch_index)
            # Calculate global percentiles for fixed colorbar range
            self.global_vmin[view_name] = np.percentile(sampled_data, dp.vmin_percentile)
            self.global_vmax[view_name] = np.percentile(sampled_data, dp.vmax_percentile)

        # Create matplotlib figure with proper size
        cm = 1 / 2.54
        width_cm = backend_kwargs.get("width_cm", 12)

        # Get dimensions from first imaging object
        first_imaging = dp.imaging_dict[dp.view_names[0]]
        ratio = first_imaging.shape[0] / first_imaging.shape[1]
        height_cm = width_cm * ratio

        num_views = len(dp.view_names)

        # Turn off interactive mode to prevent duplicate display
        with plt.ioff():
            # Create figure with multiple subplots if needed
            if num_views > 1:
                self.figure, self.axes = plt.subplots(1, num_views, figsize=(width_cm * num_views * cm, height_cm * cm))
                if num_views == 1:
                    self.axes = [self.axes]  # Make it a list for consistency
            else:
                self.figure, ax = plt.subplots(figsize=(width_cm * cm, height_cm * cm))
                self.axes = [ax]

            # Store image objects and colorbars for each view
            self.images = {}
            self.colorbars = {}

            for idx, view_name in enumerate(dp.view_names):
                imaging = dp.imaging_dict[view_name]
                ax = self.axes[idx]

                # Get initial frame and create image
                frame_data = imaging.get_series(self.current_frame, self.current_frame + 1, epoch_index=dp.epoch_index)
                frame = frame_data[0]

                # Create the image object with fixed colorbar range
                im = ax.imshow(
                    frame,
                    cmap=dp.colormap,
                    vmin=self.global_vmin[view_name],
                    vmax=self.global_vmax[view_name],
                    aspect="auto",
                )

                self.images[view_name] = im

                if dp.is_multi_view:
                    ax.set_title(f"{view_name}\nFrame {self.current_frame} | Time: {dp.times[self.current_frame]:.3f}s")
                else:
                    ax.set_title(f"Frame {self.current_frame} | Time: {dp.times[self.current_frame]:.3f}s")

                ax.axis("off")

                # Add colorbar with fixed range
                self.colorbars[view_name] = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            self.figure.tight_layout()

        # Create control widgets
        self._create_control_widgets(dp)

        # Setup layout
        self._setup_widget_layout()

        # Setup observers
        self._setup_observers()

        # Display if requested
        if backend_kwargs.get("display", True):
            display(self.widget)

    def _create_control_widgets(self, dp):
        """Create all control widgets."""
        import ipywidgets as widgets

        # Play/Pause button
        self.play_button = widgets.Button(
            description="▶ Play", button_style="success", layout=widgets.Layout(width="80px")
        )

        # Frame slider (main time navigation)
        self.frame_slider = widgets.IntSlider(
            value=self.current_frame,
            min=0,
            max=dp.num_frames - 1,
            step=1,
            description="Frame:",
            continuous_update=False,
            layout=widgets.Layout(width="400px"),
        )

        # Time display
        self.time_label = widgets.Label(
            value=f"Time: {dp.times[self.current_frame]:.3f}s / {dp.times[-1]:.2f}s",
            layout=widgets.Layout(width="200px"),
        )

        # Playback speed control
        self.fps_slider = widgets.FloatSlider(
            value=self.playback_fps,
            min=_PLAYBACK_FPS_MIN,
            max=min(30.0, dp.frame_rate),
            step=0.1,
            description="Speed (fps):",
            continuous_update=True,
            layout=widgets.Layout(width="250px"),
        )

        # Colormap selection
        self.colormap_dropdown = widgets.Dropdown(
            options=["gray", "viridis", "plasma", "inferno", "magma", "hot", "cool", "jet"],
            value=dp.colormap,
            description="Colormap:",
            layout=widgets.Layout(width="150px"),
        )

        # Contrast controls - now as percentage of global range
        self.vmin_slider = widgets.FloatSlider(
            value=0.0,  # Start at minimum of range
            min=0,
            max=100,
            step=1.0,
            description="Min %:",
            continuous_update=True,
            layout=widgets.Layout(width="300px"),
        )

        self.vmax_slider = widgets.FloatSlider(
            value=100.0,  # Start at maximum of range
            min=0,
            max=100,
            step=1.0,
            description="Max %:",
            continuous_update=True,
            layout=widgets.Layout(width="300px"),
        )

    def _setup_widget_layout(self):
        """Arrange widgets in layout."""
        import ipywidgets as widgets

        # Top row: play controls and time info
        playback_controls = widgets.HBox([self.play_button, self.frame_slider, self.time_label])

        # Middle row: playback speed
        speed_controls = widgets.HBox([self.fps_slider])

        # Bottom row: display controls
        display_controls = widgets.HBox(
            [
                self.colormap_dropdown,
                self.vmin_slider,
                self.vmax_slider,
            ]
        )

        # Main layout: controls at top, matplotlib canvas below
        self.widget = widgets.VBox(
            [
                playback_controls,
                speed_controls,
                display_controls,
                self.figure.canvas,  # Use the matplotlib canvas directly
            ]
        )

    def _setup_observers(self):
        """Setup widget event observers."""
        self.play_button.on_click(self._on_play_button_clicked)
        self.frame_slider.observe(self._on_frame_changed, names="value")
        self.fps_slider.observe(self._on_fps_changed, names="value")
        self.colormap_dropdown.observe(self._on_display_changed, names="value")
        self.vmin_slider.observe(self._on_display_changed, names="value")
        self.vmax_slider.observe(self._on_display_changed, names="value")

    def _update_display(self):
        """Update the image display.

        Captures current_frame and _frame_generation together under the lock and renders that
        one snapshot throughout, rather than re-reading current_frame at each of the several
        points below (per-view image data, each title, the time label) -- a concurrent seek or
        playback advance partway through could otherwise mix pixels from one frame with
        another's title/time text. Converges the same way _publish_current_frame_to_slider
        does: redraws again if the generation moved on while this render was in flight.

        The actual Matplotlib mutations below are serialized by _render_lock, not
        _playback_timing_lock: two calls capturing different (frame, generation) snapshots --
        e.g. this call's own publish-echo and a genuine seek's own concurrent call -- would
        otherwise be free to interleave their set_data/set_title/draw_idle calls on the same
        Axes/Image objects, drawing a torn frame (one frame's pixels under another's title)
        before either call's generation-recheck could catch up and correct it. Frame data is
        fetched (imaging.get_series, potentially slow disk/network I/O) before the lock is
        taken, not inside it, so a stale render can't block a concurrent one for the fetch's
        duration -- only for the brief mutation-and-draw that actually needs serializing.
        """
        while True:
            with self._playback_timing_lock:
                frame = self.current_frame
                generation = self._frame_generation
            dp = to_attr(self.data_plot)

            # Fetch everything needed per view before taking the render lock -- imaging.get_series
            # can be slow (disk/network I/O), and the lock must only ever guard the actual
            # Matplotlib mutations below, not this fetch, or a stale render would block a
            # concurrent one for the full I/O duration instead of just the brief mutation+draw.
            view_updates = []
            for idx, view_name in enumerate(dp.view_names):
                imaging = dp.imaging_dict[view_name]

                # Get current frame data -- the only part of gathering that's genuinely slow
                # (disk/network I/O), hence the only part done before the render lock.
                frame_data = imaging.get_series(frame, frame + 1, epoch_index=dp.epoch_index)

                if dp.is_multi_view:
                    title = f"{view_name}\nFrame {frame} | Time: {dp.times[frame]:.3f}s"
                else:
                    title = f"Frame {frame} | Time: {dp.times[frame]:.3f}s"

                view_updates.append((idx, frame_data[0], title))

            with self._render_lock:
                for idx, image_data, title in view_updates:
                    view_name = dp.view_names[idx]
                    ax = self.axes[idx]
                    im = self.images[view_name]

                    # vmin/vmax are read here, under the same lock that serializes against
                    # _on_display_changed's own render -- not gathered above with frame_data,
                    # even though they're cheap (no I/O) like it. Snapshotting them early
                    # instead would open a gap for a concurrent contrast/colormap change to
                    # render its newer settings, only for this call's stale ones to overwrite
                    # them right after -- invisible to the generation recheck below, since
                    # display settings don't bump _frame_generation.
                    # Use slider values as scaling factors on the global range
                    # This keeps the colorbar fixed but allows user adjustment
                    range_span = self.global_vmax[view_name] - self.global_vmin[view_name]
                    vmin_val = self.global_vmin[view_name] + (self.vmin_slider.value / 100.0) * range_span
                    vmax_val = self.global_vmin[view_name] + (self.vmax_slider.value / 100.0) * range_span

                    # Update the image data and colormap (much faster than recreating)
                    im.set_data(image_data)  # Remove time dimension
                    im.set_cmap(self.colormap_dropdown.value)
                    im.set_clim(vmin=vmin_val, vmax=vmax_val)

                    # Update title
                    ax.set_title(title)

                # Update time label
                self._update_time_label(frame)

                # Refresh the canvas
                self.figure.canvas.draw_idle()

            with self._playback_timing_lock:
                if self._frame_generation == generation:
                    return

    def _update_time_label(self, frame):
        """Update time display label."""
        dp = to_attr(self.data_plot)
        current_time = dp.times[frame]
        total_time = dp.times[-1]
        self.time_label.value = f"Time: {current_time:.3f}s / {total_time:.2f}s"

    def _on_play_button_clicked(self, button):
        """Handle play/pause button click."""
        with self._playback_timing_lock:
            if self.is_playing:
                self._stop_playback()
            else:
                self._start_playback()

    def _start_playback(self):
        """Start video playback in a separate thread."""
        import threading
        import time

        with self._playback_timing_lock:
            self.is_playing = True
            self._playback_last_time = time.monotonic()
            # Marks this as a fresh request to be playing -- see _playback_start_id in
            # __init__ and _playback_loop's cleanup section.
            self._playback_start_id += 1
            # Wake a reused, still-sleeping worker immediately -- see _playback_wake_event in
            # __init__ and the poll wait in _playback_loop. Harmless if the worker isn't
            # currently waiting on it (a freshly spawned one below never has been yet): the
            # first wait it does reach afterward just clears it away as normal.
            self._playback_wake_event.set()
            # Button update shares the lock with the state flag: _stop_playback can otherwise
            # be called from the worker thread with a gap between setting is_playing and
            # updating the button, letting a concurrent click observe/overwrite a half-applied
            # transition (see _stop_playback).
            self.play_button.description = "⏸ Pause"
            self.play_button.button_style = "warning"
            if self.play_thread is None or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self._playback_loop)
                self.play_thread.daemon = True
                # Start playback thread while still holding the transition lock so a
                # concurrent start cannot replace self.play_thread in between creation and
                # start(), which would otherwise race into double-starting the wrong thread.
                self.play_thread.start()

    def _stop_playback(self):
        """Stop video playback.

        Called both from the play button (via _on_play_button_clicked) and from _playback_loop
        itself when it reaches the last frame -- both already holding the lock (reentrant,
        since _playback_timing_lock is an RLock). The state flag and button update must stay
        atomic either way, or a click landing between them could see is_playing already False
        and start new playback, only for this call's own button update to then overwrite it
        back to "Play" right after.
        """
        with self._playback_timing_lock:
            self.is_playing = False
            self.play_button.description = "▶ Play"
            self.play_button.button_style = "success"
            # Wake a worker currently blocked in its poll wait -- see _playback_wake_event in
            # __init__ -- so it notices is_playing is now False promptly instead of only once
            # that wait happens to elapse on its own (up to a full poll interval: 2.5s at the
            # slider's minimum fps), during which the thread stays alive and play_thread still
            # points at it even though the button already shows "Play". Harmless when called
            # from the worker's own thread (it isn't asleep on this event right now).
            self._playback_wake_event.set()

    def _playback_loop(self):
        """Main playback loop running in separate thread.

        Paced by elapsed wall-clock time rather than a fixed per-iteration increment: each
        redraw involves a full figure re-render plus a Jupyter comm/websocket round trip, which
        can take longer than ``1 / playback_fps``. A naive ``current_frame += 1`` on a fixed
        timer would race ahead of what's actually reached the browser -- since ipywidgets only
        syncs the latest value, whichever intermediate frames never got flushed are silently
        dropped, with no control over which ones. Instead, each iteration computes how many
        frames *should* have elapsed since the last actual advance and jumps straight there --
        deliberately skipping frames (evenly, by real elapsed time) so playback speed stays
        correct under load, rather than an uncontrolled, backpressure-dependent frame drop.
        ``self._playback_last_time`` advances by the exact duration of the frames just
        consumed, not to ``now`` -- carrying over any leftover fraction of a frame period
        instead of discarding it, so playback doesn't drift behind the requested rate over a
        long session. ``_on_fps_changed`` resets it on a rate change (see its own comment).
        """
        import threading
        import time

        dp = to_attr(self.data_plot)

        with self._playback_timing_lock:
            my_start_id = self._playback_start_id

        reached_last_frame = False
        while True:
            with self._playback_timing_lock:
                if not self.is_playing or self.current_frame >= dp.num_frames - 1:
                    reached_last_frame = self.current_frame >= dp.num_frames - 1
                    break
                if self._playback_last_time is None:
                    self._playback_last_time = time.monotonic()
                now = time.monotonic()
                playback_fps = self.playback_fps
                elapsed_seconds = max(0.0, now - self._playback_last_time)
                frames_elapsed = int(elapsed_seconds * playback_fps)
                frame_to_display = None
                if frames_elapsed > 0:
                    self.current_frame = min(self.current_frame + frames_elapsed, dp.num_frames - 1)
                    self._playback_last_time += frames_elapsed / playback_fps
                    self._frame_generation += 1
                    reached_last_frame = self.current_frame >= dp.num_frames - 1
                    frame_to_display = self.current_frame
                else:
                    reached_last_frame = False
            if frame_to_display is not None:
                # Publish, not a raw write -- see _publish_current_frame_to_slider.
                self._publish_current_frame_to_slider()
            with self._playback_timing_lock:
                if not self.is_playing:
                    reached_last_frame = False
                    break
                reached_last_frame = self.current_frame >= dp.num_frames - 1
                pre_sleep_start_id = self._playback_start_id
                pre_sleep_frame_generation = self._frame_generation
            if reached_last_frame:
                break
            with self._playback_timing_lock:
                if (
                    self._playback_start_id != pre_sleep_start_id
                    or self._frame_generation != pre_sleep_frame_generation
                ):
                    # A _start_playback() *or* a seek (_on_frame_changed / seek_to_frame)
                    # landed in the gap between the checkpoint above and here -- either one
                    # already set the wake event, but the clear() below would wipe that out
                    # before we ever start waiting on it (we're not asleep yet, so there's
                    # nothing for that set() to interrupt). A restart bumps
                    # _playback_start_id; a seek bumps _frame_generation instead (it doesn't
                    # touch _playback_start_id), so both must be checked -- otherwise a seek
                    # landing here (e.g. onto the last frame) would have its wake-up silently
                    # consumed and playback would stay visibly active for a full poll
                    # interval before this loop noticed. Skip the wait entirely and loop back
                    # to reprocess with fresh state instead of sleeping through it.
                    continue
                sleep_duration = 1.0 / (4 * self.playback_fps)
                # Cleared here, under the same lock _start_playback sets it under, right
                # before the wait begins -- not after the wait returns. Otherwise a
                # _start_playback() landing in the gap between this wait timing out and a
                # trailing clear() would have its wake-up silently consumed by that clear
                # instead of arming the *next* wait, delaying the restart for a full poll
                # interval anyway (the bug this event exists to avoid).
                self._playback_wake_event.clear()
            # An interruptible wait, not a plain time.sleep(): _start_playback sets this event
            # to wake a reused, still-sleeping worker immediately (see its own comment) instead
            # of leaving it to notice on its own once this poll interval happens to elapse.
            self._playback_wake_event.wait(timeout=sleep_duration)
        # Defaults to the loop's own reached_last_frame: even a worker that's no longer the
        # registered play_thread below (e.g. superseded by a restart) must still signal a stop
        # if it genuinely reached the end from its own perspective.
        should_stop_playback = reached_last_frame
        with self._playback_timing_lock:
            if self.play_thread is threading.current_thread():
                self.play_thread = None
                reached_last_frame = self.current_frame >= dp.num_frames - 1
                # A newer _start_playback() since this worker began means its "reached the end"
                # decision below may be stale (a pause-then-play landing in the gap between
                # this worker's publish and this checkpoint reuses this same still-alive worker
                # rather than replacing it). Rather than just standing down and leaving
                # is_playing True with no worker left to act on it, hand off to a fresh worker
                # that re-evaluates with current information -- same as the "still more to
                # play" case below, and it settles to stopped itself if there genuinely isn't.
                newer_start = self._playback_start_id != my_start_id
                if self.is_playing and (not reached_last_frame or newer_start):
                    self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
                    self.play_thread.start()
                    should_stop_playback = False
                elif self.is_playing:
                    should_stop_playback = True
            # Stop when reaching the end while still holding the transition lock, so a
            # concurrent pause/play cannot start a replacement worker that this exiting worker
            # immediately stops with a stale end-of-playback decision.
            if should_stop_playback:
                self._stop_playback()

    def _publish_current_frame_to_slider(self):
        """Sync frame_slider.value to self.current_frame.

        current_frame may keep changing concurrently -- another seek, or the playback loop's
        own advance -- while this call is in flight, since the actual widget write has to
        happen outside the lock (it synchronously triggers _on_frame_changed's redraw via the
        observer, and holding the lock across a redraw would block other callbacks for its
        duration). So this captures the frame and _frame_generation together under the lock,
        writes it, then checks whether the generation is still the one just captured -- if
        something newer landed while the write was in flight, it loops and republishes the
        latest state instead of leaving the widget showing a value current_frame has already
        moved past. This converges: whichever call is genuinely the last to change
        current_frame is also the one whose post-write check finds nothing newer.

        The write is marked as this thread's own (see self._slider_write_state in __init__) so
        _on_frame_changed treats the resulting observer call as a redraw of state already set
        under lock by the caller, not as an independent new seek.

        If frame_slider.value already equals frame, assigning it is a no-op as far as the
        underlying trait is concerned -- ipywidgets/traitlets only notifies observers on an
        actual change, so _on_frame_changed (and the redraw it triggers) would silently never
        run. Redraw explicitly for that case instead of relying on the observer.
        """
        while True:
            with self._playback_timing_lock:
                frame = self.current_frame
                generation = self._frame_generation
            unchanged = self.frame_slider.value == frame
            self._slider_write_state.publishing = True
            try:
                self.frame_slider.value = frame
                if unchanged:
                    self._update_display()
            finally:
                self._slider_write_state.publishing = False
            with self._playback_timing_lock:
                if self._frame_generation == generation:
                    return

    def _on_frame_changed(self, change):
        """Handle frame slider change."""
        if getattr(self._slider_write_state, "publishing", False):
            # This observer call is _publish_current_frame_to_slider's own write reflecting
            # back -- current_frame was already set under lock by whoever called it. Just
            # redraw.
            self._update_display()
            return
        import time

        with self._playback_timing_lock:
            self.current_frame = change["new"]
            # A user seek during active playback must reset the pacing reference too, or the
            # next poll counts time elapsed since the last *playback* advance (which predates
            # the seek) and immediately jumps forward again from the newly seeked position.
            self._playback_last_time = time.monotonic()
            self._frame_generation += 1
            # Wake a worker that's currently sleeping between polls -- otherwise it won't
            # notice this seek (e.g. one landing on the last frame, which should stop playback
            # immediately) until that sleep happens to elapse on its own (see _start_playback's
            # own comment on this same event).
            self._playback_wake_event.set()
        # Publish, not a raw redraw -- storing the slider's new value and notifying this
        # observer aren't one atomic step against another thread. A concurrent publish can
        # capture the pre-seek state, overwrite the slider back to it, and finish its own
        # generation recheck entirely inside that gap (this seek's own bump above hasn't
        # happened yet from its point of view), leaving the slider showing the publish's stale
        # value. Publishing here reasserts the slider against exactly that, the same way it
        # already reconciles a publish against a concurrent seek.
        self._publish_current_frame_to_slider()

    def _on_fps_changed(self, change):
        """Handle FPS slider change."""
        import time

        new_fps = max(_PLAYBACK_FPS_MIN, float(change["new"]))
        with self._playback_timing_lock:
            self.playback_fps = new_fps
            # Reset the pacing reference so the new rate only applies to time elapsed from here on --
            # otherwise _playback_loop would apply it to time that already elapsed under the old rate,
            # producing an incorrect frame jump right at the moment of the change.
            self._playback_last_time = time.monotonic()
            # If the worker is currently asleep in its poll wait, it's waiting out the *old*
            # fps's (longer) interval -- wake it so the new rate takes effect immediately
            # instead of only once that old interval happens to elapse on its own (up to 2.5s
            # at the slider's minimum fps). Same missed-wakeup risk _start_playback/seeks have.
            self._playback_wake_event.set()

    def _on_display_changed(self, change):
        """Handle display parameter changes (colormap, contrast)."""
        self._update_display()

    def seek_to_frame(self, frame_number: int):
        """Seek to a specific frame.
        Parameters
        ----------
        frame_number : int
            Frame number to seek to
        """
        dp = to_attr(self.data_plot)
        if 0 <= frame_number < dp.num_frames:
            import time

            with self._playback_timing_lock:
                self.current_frame = frame_number
                self._playback_last_time = time.monotonic()
                self._frame_generation += 1
                # Same missed-wakeup risk as a real slider seek (see _on_frame_changed): wake a
                # worker that's currently sleeping between polls, so it notices this seek (e.g.
                # onto the last frame, which should stop playback immediately) right away.
                self._playback_wake_event.set()
            if hasattr(self, "frame_slider"):
                # Publish, not a raw write -- current_frame above is already authoritative, so
                # this is UI sync for a decision already made, not a new seek (a raw write here
                # would be treated as one; see _publish_current_frame_to_slider).
                self._publish_current_frame_to_slider()
            else:
                self._update_display()

    def seek_to_time(self, time_seconds: float):
        """Seek to a specific time.
        Parameters
        ----------
        time_seconds : float
            Time in seconds to seek to
        """
        dp = to_attr(self.data_plot)
        # Find closest frame to the requested time
        frame_idx = np.argmin(np.abs(dp.times - time_seconds))
        self.seek_to_frame(frame_idx)


plot_imaging_series = ImagingSeriesWidget
