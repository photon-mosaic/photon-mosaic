import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from spikeinterface.widgets import get_some_colors
from spikeinterface.widgets.base import BaseWidget, to_attr
from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

from photon_mosaic.core.baseimaging import BaseImaging
from photon_mosaic.core.baserois import BaseRois


class RoisWidget(BaseWidget):
    """Widget for visualizing ROI masks with optional background image.

    This widget displays ROI masks with different colors and optional background
    from an imaging extractor's average image.

    Parameters
    ----------
    rois : BaseRois
        The ROIs object containing masks to visualize
    imaging : BaseImaging, optional
        Optional imaging extractor to use average image as background, by default None
    alpha : float, optional
        Transparency value for ROI masks (0=transparent, 1=opaque), by default 0.5
    colors : list or str, optional
        List of colors for ROIs or colormap name, by default 'tab10'
    show_roi_ids : bool, optional
        Whether to show ROI IDs as text labels, by default True
    backend : str, optional
        Backend to use, by default None
    **backend_kwargs
        Additional backend-specific arguments
    """

    def __init__(
        self,
        rois: BaseRois,
        imaging: BaseImaging | None = None,
        alpha: float = 0.5,
        colors: dict | str | None = None,
        show_roi_ids: bool = False,
        backend=None,
        **backend_kwargs,
    ):
        # Get ROI information
        roi_ids = rois.roi_ids
        num_rois = len(roi_ids)

        # Get background image if imaging provided
        background_imaging = imaging or rois.imaging
        if background_imaging is not None:
            background = background_imaging.get_average_image()

        # Prepare data for plotting
        data_plot = dict(
            rois=rois,
            roi_ids=roi_ids,
            num_rois=num_rois,
            background=background,
            alpha=alpha,
            colors=colors,
            show_roi_ids=show_roi_ids,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """Matplotlib backend for plotting ROI masks."""
        dp = to_attr(data_plot)

        # Setup figure
        cm = 1 / 2.54
        width_cm = backend_kwargs.get("width_cm", 12)
        ratio = dp.background.shape[0] / dp.background.shape[1]
        height_cm = width_cm * ratio
        figsize = (width_cm * cm, height_cm * cm)
        backend_kwargs["figsize"] = figsize

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        # Plot background image if available
        if dp.background is not None:
            self.ax.imshow(dp.background, cmap="gray", aspect="auto")
            extent = [0, dp.background.shape[1], dp.background.shape[0], 0]
        else:
            # Use ROI mask dimensions to set extent
            image_shape = dp.rois.image_shape
            extent = [0, image_shape[1], image_shape[0], 0]
            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])

        # Get colormap for ROIs
        if isinstance(dp.colors, str):
            cmap = plt.get_cmap(dp.colors)
            colors = {roi_id: cmap(i / dp.num_rois) for i, roi_id in enumerate(dp.roi_ids)}
        elif dp.colors is None:
            colors = get_some_colors(dp.roi_ids)
        else:
            colors = dp.colors

        # Get all ROI masks at once: shape (num_rois, height, width)
        masks = dp.rois.get_roi_image_masks(dp.roi_ids)
        # Prepare RGBA overlays for all ROIs
        overlay = np.zeros((*masks.shape[1:], 4), dtype=float)
        for idx, roi_id in enumerate(dp.roi_ids):
            color = to_rgba(colors[roi_id])
            mask = masks[idx]
            overlay[mask > 0, :3] = color[:3]
            overlay[mask > 0, 3] = dp.alpha

        # Plot the combined overlay
        self.ax.imshow(overlay, extent=extent, aspect="auto")

        # Add ROI ID labels if requested
        if dp.show_roi_ids:
            for idx, roi_id in enumerate(dp.roi_ids):
                mask = masks[idx]
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0:
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    self.ax.text(
                        centroid_x,
                        centroid_y,
                        str(roi_id),
                        color="white",
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
                    )

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        self.ax.set_title(f"ROI Masks (n={dp.num_rois})")

        self.figure.tight_layout()

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        """Interactive ipywidgets plot with ROI selection and zoom."""
        import matplotlib.pyplot as plt
        from IPython.display import display
        from spikeinterface.widgets.utils_ipywidgets import check_ipywidget_backend

        check_ipywidget_backend()

        dp = to_attr(data_plot)

        # Store data for updates
        self.data_plot = data_plot
        self.selected_roi_idx = 0

        # Precompute colors for all ROIs
        if isinstance(dp.colors, str):
            cmap = plt.get_cmap(dp.colors)
            self.roi_colors = {roi_id: cmap(i / dp.num_rois) for i, roi_id in enumerate(dp.roi_ids)}
        elif dp.colors is None:
            self.roi_colors = get_some_colors(dp.roi_ids)
        else:
            self.roi_colors = dp.colors

        # Get all masks at once for efficiency
        self.all_masks = dp.rois.get_roi_image_masks(dp.roi_ids)

        # Create matplotlib figures with proper size
        cm = 1 / 2.54
        width_cm = backend_kwargs.get("width_cm", 12)

        if dp.background is not None:
            ratio = dp.background.shape[0] / dp.background.shape[1]
        else:
            image_shape = dp.rois.image_shape
            ratio = image_shape[0] / image_shape[1]

        height_cm = width_cm * ratio

        # Turn off interactive mode to prevent duplicate display
        with plt.ioff():
            # Create figure with two subplots: full FOV and zoomed view
            self.figure, (self.ax_full, self.ax_zoom) = plt.subplots(1, 2, figsize=(width_cm * 2 * cm, height_cm * cm))

            # Plot full FOV
            self._plot_full_fov(dp)

            # Plot zoomed view of selected ROI
            self._plot_zoomed_roi(dp, self.selected_roi_idx)

            self.figure.tight_layout()

        # Create control widgets
        self._create_roi_controls(dp)

        # Setup layout
        self._setup_ipywidget_layout()

        # Setup observers
        self._setup_roi_observers()

        # Display if requested
        if backend_kwargs.get("display", True):
            display(self.widget)

    def _plot_full_fov(self, dp):
        """Plot the full field of view with all ROI masks."""
        self.ax_full.clear()

        # Plot background image if available
        if dp.background is not None:
            self.ax_full.imshow(dp.background, cmap="gray", aspect="auto")
            extent = [0, dp.background.shape[1], dp.background.shape[0], 0]
        else:
            # Use ROI mask dimensions to set extent
            image_shape = dp.rois.image_shape
            extent = [0, image_shape[1], image_shape[0], 0]
            self.ax_full.set_xlim(extent[0], extent[1])
            self.ax_full.set_ylim(extent[2], extent[3])
            self.ax_full.set_facecolor("black")

        # Prepare RGBA overlays for all ROIs
        overlay = np.zeros((*self.all_masks.shape[1:], 4), dtype=float)
        for idx, roi in enumerate(dp.roi_ids):
            color = self.roi_colors[roi]
            mask = self.all_masks[idx]

            # Highlight selected ROI with full opacity, others with reduced alpha
            if idx == self.selected_roi_idx:
                alpha = 0.8
            else:
                alpha = dp.alpha * 0.5  # Reduce alpha for non-selected ROIs

            overlay[mask > 0, :3] = color[:3]
            overlay[mask > 0, 3] = alpha

        # Plot the combined overlay
        self.ax_full.imshow(overlay, extent=extent, aspect="auto")

        # Add ROI ID labels if requested
        if dp.show_roi_ids:
            for idx, roi_id in enumerate(dp.roi_ids):
                mask = self.all_masks[idx]
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0:
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)

                    # Get the color for this ROI
                    roi_color = self.roi_colors[roi_id]

                    # Highlight selected ROI label
                    if idx == self.selected_roi_idx:
                        fontsize = 10
                        fontweight = "bold"
                        bbox_alpha = 0.8
                    else:
                        fontsize = 8
                        fontweight = "normal"
                        bbox_alpha = 0.5

                    self.ax_full.text(
                        centroid_x,
                        centroid_y,
                        str(roi_id),
                        color="white",
                        fontsize=fontsize,
                        fontweight=fontweight,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=roi_color, alpha=bbox_alpha),
                    )

        self.ax_full.set_xlabel("X (pixels)")
        self.ax_full.set_ylabel("Y (pixels)")
        self.ax_full.set_title(f"Full FOV - ROI Masks (n={dp.num_rois})")

    def _plot_zoomed_roi(self, dp, roi_idx):
        """Plot zoomed view of a specific ROI."""
        self.ax_zoom.clear()

        # Get the selected ROI mask and ID
        roi_id = dp.roi_ids[roi_idx]
        mask = self.all_masks[roi_idx]

        # Find bounding box of the ROI with padding
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            self.ax_zoom.text(0.5, 0.5, "Empty ROI", ha="center", va="center", transform=self.ax_zoom.transAxes)
            self.ax_zoom.set_title(f"ROI {roi_id} - Zoomed View")
            return

        pad = 20  # Padding in pixels
        y_min = max(0, y_coords.min() - pad)
        y_max = min(mask.shape[0], y_coords.max() + pad + 1)
        x_min = max(0, x_coords.min() - pad)
        x_max = min(mask.shape[1], x_coords.max() + pad + 1)

        # Crop background if available
        if dp.background is not None:
            bg_crop = dp.background[y_min:y_max, x_min:x_max]
            self.ax_zoom.imshow(bg_crop, cmap="gray", aspect="auto", extent=[x_min, x_max, y_max, y_min])
        else:
            self.ax_zoom.set_xlim(x_min, x_max)
            self.ax_zoom.set_ylim(y_max, y_min)
            self.ax_zoom.set_facecolor("black")

        # Create overlay for the zoomed ROI
        overlay = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=float)
        mask_crop = mask[y_min:y_max, x_min:x_max]

        color = to_rgba(self.roi_colors[roi_id])
        overlay[mask_crop > 0, :3] = color[:3]
        overlay[mask_crop > 0, 3] = 0.8  # Full visibility for selected ROI

        self.ax_zoom.imshow(overlay, extent=[x_min, x_max, y_max, y_min], aspect="auto")

        # Add ROI ID at centroid with matching color
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        self.ax_zoom.text(
            centroid_x,
            centroid_y,
            str(roi_id),
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

        self.ax_zoom.set_xlabel("X (pixels)")
        self.ax_zoom.set_ylabel("Y (pixels)")
        self.ax_zoom.set_title(f"ROI {roi_id} - Zoomed View")

    def _create_roi_controls(self, dp):
        """Create ROI selection widgets."""
        import ipywidgets as widgets

        # Previous/Next buttons for ROI navigation
        self.prev_button = widgets.Button(
            description="◀ Prev", button_style="info", layout=widgets.Layout(width="80px")
        )

        self.next_button = widgets.Button(
            description="Next ▶", button_style="info", layout=widgets.Layout(width="80px")
        )

        # ROI selection dropdown
        roi_options = [(f"ROI {roi_id}", idx) for idx, roi_id in enumerate(dp.roi_ids)]
        self.roi_dropdown = widgets.Dropdown(
            options=roi_options,
            value=self.selected_roi_idx,
            description="Select ROI:",
            layout=widgets.Layout(width="300px"),
        )

        # Alpha slider for transparency control
        self.alpha_slider = widgets.FloatSlider(
            value=dp.alpha,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Alpha:",
            continuous_update=True,
            layout=widgets.Layout(width="300px"),
        )

        # Toggle for ROI IDs
        self.show_ids_toggle = widgets.Checkbox(
            value=dp.show_roi_ids, description="Show ROI IDs", layout=widgets.Layout(width="300px")
        )

    def _setup_ipywidget_layout(self):
        """Arrange widgets in layout."""
        import ipywidgets as widgets

        # ROI navigation row with buttons and dropdown
        roi_nav = widgets.HBox([self.prev_button, self.next_button, self.roi_dropdown])

        # Display controls row
        display_controls = widgets.HBox([self.alpha_slider, self.show_ids_toggle])

        # Main layout: navigation row, display controls, then matplotlib canvas
        self.widget = widgets.VBox([roi_nav, display_controls, self.figure.canvas])

    def _setup_roi_observers(self):
        """Setup observers for widget interactions."""
        self.prev_button.on_click(self._on_prev_clicked)
        self.next_button.on_click(self._on_next_clicked)
        self.roi_dropdown.observe(self._on_roi_changed, names="value")
        self.alpha_slider.observe(self._on_alpha_changed, names="value")
        self.show_ids_toggle.observe(self._on_show_ids_changed, names="value")

    def _on_prev_clicked(self, button):
        """Handle previous button click."""
        if self.selected_roi_idx > 0:
            self.selected_roi_idx -= 1
            # Update dropdown to reflect the change
            self.roi_dropdown.value = self.selected_roi_idx
            self._update_plots()

    def _on_next_clicked(self, button):
        """Handle next button click."""
        dp = to_attr(self.data_plot)
        if self.selected_roi_idx < dp.num_rois - 1:
            self.selected_roi_idx += 1
            # Update dropdown to reflect the change
            self.roi_dropdown.value = self.selected_roi_idx
            self._update_plots()

    def _on_roi_changed(self, change):
        """Handle ROI selection change."""
        self.selected_roi_idx = change["new"]
        self._update_plots()

    def _on_alpha_changed(self, change):
        """Handle alpha slider change."""
        self.data_plot["alpha"] = change["new"]
        self._update_plots()

    def _on_show_ids_changed(self, change):
        """Handle show IDs toggle change."""
        self.data_plot["show_roi_ids"] = change["new"]
        self._update_plots()

    def _update_plots(self):
        """Update both full FOV and zoomed plots."""
        dp = to_attr(self.data_plot)
        self._plot_full_fov(dp)
        self._plot_zoomed_roi(dp, self.selected_roi_idx)
        self.figure.canvas.draw_idle()


plot_rois = RoisWidget
