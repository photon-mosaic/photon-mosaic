from photon_mosaic.core.baseimaging import BaseImaging

from spikeinterface.widgets.base import BaseWidget, to_attr
import numpy as np


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
        time_range: tuple = None,
        colormap: str = "gray",
        vmin_percentile: float = 2.0,
        vmax_percentile: float = 98.0,
        backend=None,
        **backend_kwargs,
    ):
        # Check if imaging is a dictionary
        is_multi_view = isinstance(imaging, dict)
        
        if is_multi_view:
            # Get the first imaging object to extract common properties
            first_key = list(imaging.keys())[0]
            first_imaging = imaging[first_key]
            
            # Validate all imaging objects have the same number of frames
            num_frames = first_imaging.get_num_samples(epoch_index)
            times = first_imaging.get_times(epoch_index)
            frame_rate = first_imaging.sampling_frequency
            
            for name, img in imaging.items():
                if img.get_num_samples(epoch_index) != num_frames:
                    raise ValueError(f"All imaging objects must have the same number of frames. "
                                   f"'{name}' has {img.get_num_samples(epoch_index)} frames, "
                                   f"expected {num_frames}")
            
            imaging_dict = imaging
            view_names = list(imaging.keys())
        else:
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
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
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

        # Calculate global contrast range from multiple frames for consistent colorbar
        # Sample frames throughout the video to get representative range
        num_samples = 100
        
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
                self.figure, self.axes = plt.subplots(
                    1, num_views, 
                    figsize=(width_cm * num_views * cm, height_cm * cm)
                )
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
                frame_data = imaging.get_series(
                    self.current_frame, self.current_frame + 1, epoch_index=dp.epoch_index
                )
                frame = frame_data[0]

                # Create the image object with fixed colorbar range
                im = ax.imshow(
                    frame, cmap=dp.colormap, 
                    vmin=self.global_vmin[view_name], 
                    vmax=self.global_vmax[view_name], 
                    aspect="auto"
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
            min=0.1,
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
        """Update the image display."""
        dp = to_attr(self.data_plot)

        # Update all views
        for idx, view_name in enumerate(dp.view_names):
            imaging = dp.imaging_dict[view_name]
            ax = self.axes[idx]
            im = self.images[view_name]
            
            # Get current frame data
            frame_data = imaging.get_series(self.current_frame, self.current_frame + 1, epoch_index=dp.epoch_index)
            frame = frame_data[0]  # Remove time dimension

            # Use slider values as scaling factors on the global range
            # This keeps the colorbar fixed but allows user adjustment
            range_span = self.global_vmax[view_name] - self.global_vmin[view_name]
            vmin_val = self.global_vmin[view_name] + (self.vmin_slider.value / 100.0) * range_span
            vmax_val = self.global_vmin[view_name] + (self.vmax_slider.value / 100.0) * range_span

            # Update the image data and colormap (much faster than recreating)
            im.set_data(frame)
            im.set_cmap(self.colormap_dropdown.value)
            im.set_clim(vmin=vmin_val, vmax=vmax_val)

            # Update title
            if dp.is_multi_view:
                ax.set_title(f"{view_name}\nFrame {self.current_frame} | Time: {dp.times[self.current_frame]:.3f}s")
            else:
                ax.set_title(f"Frame {self.current_frame} | Time: {dp.times[self.current_frame]:.3f}s")

        # Update time label
        self._update_time_label()

        # Refresh the canvas
        self.figure.canvas.draw_idle()

    def _update_time_label(self):
        """Update time display label."""
        dp = to_attr(self.data_plot)
        current_time = dp.times[self.current_frame]
        total_time = dp.times[-1]
        self.time_label.value = f"Time: {current_time:.3f}s / {total_time:.2f}s"

    def _on_play_button_clicked(self, button):
        """Handle play/pause button click."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start video playback in a separate thread."""
        import threading

        self.is_playing = True
        self.play_button.description = "⏸ Pause"
        self.play_button.button_style = "warning"

        # Start playback thread
        self.play_thread = threading.Thread(target=self._playback_loop)
        self.play_thread.daemon = True
        self.play_thread.start()

    def _stop_playback(self):
        """Stop video playback."""
        self.is_playing = False
        self.play_button.description = "▶ Play"
        self.play_button.button_style = "success"

    def _playback_loop(self):
        """Main playback loop running in separate thread."""
        import time

        dp = to_attr(self.data_plot)

        while self.is_playing and self.current_frame < dp.num_frames - 1:
            time.sleep(1.0 / self.playback_fps)
            if self.is_playing:  # Check again in case it was stopped
                self.current_frame += 1
                # Update slider and display
                self.frame_slider.value = self.current_frame
        # Stop when reaching the end
        if self.current_frame >= dp.num_frames - 1:
            self._stop_playback()

    def _on_frame_changed(self, change):
        """Handle frame slider change."""
        self.current_frame = change["new"]
        self._update_display()

    def _on_fps_changed(self, change):
        """Handle FPS slider change."""
        self.playback_fps = change["new"]

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
            self.current_frame = frame_number
            if hasattr(self, "frame_slider"):
                self.frame_slider.value = frame_number
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
