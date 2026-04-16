from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch
from .baseregistrationsettings import BaseRegistrationSettings
from photon_mosaic.core import BaseImaging, BaseImagingEpoch
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict
import logging
from typing import Optional
import numpy as np
import time
import os

class Suite2pSegmentationSettings(BaseRegistrationSettings):
    """Segmentation image settings"""

    # Basic parameters
    input_dir: Path = Field(
        default=Path("../data/"),
        description="Directory containing the input data files.",
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory for temporary files created during processing.",
    )

    # Cell detection parameters
    diameter: int = Field(
        default=0,
        description=(
            "Expected diameter of cells in pixels. "
            "If set to 0, CellPose will estimate the diameter from the data."
        ),
    )
    init: str = Field(
        default="mean",
        description=(
            "Initialization method for finding masks. Options: "
            "max/mean: Cellpose on max projection image divided by mean image; "
            "mean: Cellpose on mean image; "
            "enhanced_mean: Cellpose on enhanced mean image; "
            "max: Cellpose on maximum projection image; "
            "sourcery: Suite2p's functional mode without 'sparse_mode'; "
            "sparsery: Suite2p's functional mode with 'sparse_mode'; "
        ),
    )    
    bad_frames: list = Field(
        default=[],
        description="List of bad frames to exclude from segmentation"
    ),
    device: str = Field(
        default=None,
        description="Device for computation"
    ),
    stat: np.ndarrray = Field(
        default=None,
        description="Array of ROI statistics dictionaries."
    ),
    save_path: str | Path = Field(
        default=None,
        description="where to save data" # this is likely redundant in this library
    )
    functional_chan: int = Field(
        default=1,
        description="this channel is used to extract functional ROIs (1-based)",
    )
    threshold_scaling: int = Field(
        default=1,
        description="adjust the automatically determined threshold by this scalar multiplier",
    )
    max_overlap: float = Field(
        default=0.75,
        description="cells with more overlap than this get removed during triage, before refinement",
    )
    soma_crop: bool = Field(
        default=False,
        description="crop dendrites for cell classification stats like compactness",
    )
    allow_overlap: bool = Field(
        default=False,
        description="pixels that are overlapping are thrown out (False) or added to both ROIs (True)",
    )
    denoise: bool = Field(
        default=False,
        description=(
            "If True, applies denoising to the binned movie before cell detection."
        ),
    )
    cellprob_threshold: float = Field(
        default=0.0,
        description=(
            "Probability threshold for CellPose cell detection. "
            "Decrease this threshold if CellPose is not returning enough ROIs."
        ),
    )
    flow_threshold: float = Field(
        default=1.5,
        description=(
            "Flow threshold used by CellPose during cell detection. "
            "Increase this threshold if CellPose is not returning enough ROIs."
        ),
    )
    spatial_hp_cp: int = Field(
        default=0,
        description=(
            "Window size for spatial high-pass filtering of the image before CellPose "
            "detection. Set to 0 to disable filtering."
        ),
    )
    pretrained_model: str = Field(
        default="cyto",
        description=(
            "CellPose pretrained model to use. Common options: 'cyto' (standard model), "
            "'cyto2' (improved model), or path to a custom model file."
        ),
    )
    # Neuropil parameters
    neuropil: str = Field(
        default="mutualinfo",
        description=(
            "Method to estimate and subtract neuropil contamination, and whether to perform demixing. cnmf(-e) demix traces of "
            "overlapping ROIs via NMF, suite2p & mutualinfo do not. "
            "Options: 'suite2p' (fixed r=0.7"
        ),
    )


    # CORR_PNR parameters
    min_corr: float = Field(
        default=0.6,
        description=(
            "Minimum local correlation for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )
    min_pnr: float = Field(
        default=4,
        description=(
            "Minimum peak-to-noise ratio for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )

    # Component evaluation parameters
    snr_thr: float = Field(
        default=1.5,
        description=(
            "Signal-to-noise ratio threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    rval_thr: float = Field(
        default=0.6,
        description=(
            "Spatial correlation threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    cnn_thr: float = Field(
        default=0.9,
        description=(
            "CNN classifier threshold for component acceptance in CaImAn evaluation. "
            "Components below this value will be rejected. Set to 0 to disable "
            "CNN-based classification."
        ),
    )

    # Output options
    contour_video: bool = Field(
        default=False,
        description=(
            "If True, creates a video overlaying raw data, ROI activity, and residual "
            "with contours for visualization and quality assessment."
        ),
    )

    verbose: bool = Field(
        default=False, description="Enable verbose logging and debug information."
    )

    # Config for pydantic-settings
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="EXTRACTION_", case_sensitive=False, extra="ignore"
    )

    @field_validator("init", "neuropil")
    @classmethod
    def lowercase_str_fields(cls, v: str) -> str:
        """Convert string fields to lowercase"""
        return v.lower()

    @field_validator("rf")
    @classmethod
    def validate_rf(cls, v: int) -> Optional[int]:
        if v == 0:
            return None
        return v

    def validate_consistency(self) -> Optional[str]:
        """Validate command line arguments for consistency"""
        if self.neuropil == "cnmf" and self.init == "corr_pnr":
            # We'll log a warning but still update the parameters
            self.ssub = 1
            return (
                "'corr_pnr' initialization with neuropil model 'cnmf' does "
                "not support spatial downsampling. Setting ssub to 1"
            )

        if self.neuropil == "cnmf-e" and self.init == "greedy_roi":
            raise ValueError(
                "Can't use neuropil model 'cnmf-e' with 'greedy_roi' initialization"
            )

        if self.init in ("greedy_roi", "corr_pnr") and self.neuropil[:4] != "cnmf":
            raise ValueError(
                "Can't use Suite2p neuropil model with 'greedy_roi' or 'corr_pnr' initialization"
            )

        # For backward compatibility
        if self.init in ("1", "2", "3", "4"):
            self.init = ("max/mean", "mean", "enhanced_mean", "max")[int(self.init) - 1]

        return None  # No warning message

    def model_post_init(self, _) -> None:
        """Run validation after model initialization"""
        warning = self.validate_consistency()
        if warning:
            logging.warning(warning)

def run_suite2p_segmentation(registration_outputs,settings=Suite2pSegmentationSettings,):
    plane_times = dict
    yrange, xrange = registration_outputs["yrange"], registration_outputs["xrange"]
    meanImg_chan2 = registration_outputs.get("meanImg_chan2", None)
    from suite2p.detection import detection_wrapper
    logging.info("----------- ROI DETECTION")
    t11 = time.time()
    if settings.stat is None:
        bad_frames = settings.bad_frames
        if badframes is not None:
            bad_frames[badframes] = True
        logging.info(f"Excluding {bad_frames.sum()} bad frames from detection")
        if not isinstance(settings["diameter"], (list, tuple, np.ndarray)):
            settings["diameter"] = np.array([settings["diameter"], settings["diameter"]])
        elif isinstance(settings["diameter"], (list, tuple)):
            settings["diameter"] = np.array(settings["diameter"])
        if settings["diameter"].size == 1:
            settings["diameter"] = np.array([settings["diameter"], settings["diameter"]])
        detect_outputs, stat, redcell = detection_wrapper(f_reg, 
                                                                eanImg_chan2=meanImg_chan2,
                                                    yrange=yrange, xrange=xrange,
                                                    tau=settings["tau"], fs=settings["fs"],
                                                    diameter=settings["diameter"],
                                                settings=settings["detection"], 
                                                classifier_path=classfile,
                                                badframes=bad_frames,
                                                preclassify=settings["classification"]["preclassify"],
                                                device=device)
        np.save(os.path.join(save_path, "stat.npy"), stat)
        np.save(os.path.join(save_path, "detect_outputs.npy"), detect_outputs)
        if redcell is not None:
            np.save(os.path.join(save_path, "redcell.npy"), redcell)
        
    plane_times["detection"] = time.time() - t11
    logging.info("----------- Total %0.2f sec." % plane_times["detection"])

    if len(stat) == 0:
        logging.info("no ROIs found")
        plane_times["total_plane_runtime"] = time.time() - t1
        return registration_outputs, detect_outputs, stat, None, None, None, None, None, None, None, None, plane_times

class SegmentSuite2PImaging(BasePreprocessor):
    def __init__(
        self,
        imaging: BaseImaging,
        settings: Suite2pSegmentationSettings | dict,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Wrap a `BaseImaging` object with preprocessing metadata.

        Parameters
        ----------
        imaging : BaseImaging
            Parent imaging object providing frames and metadata.
        sampling_frequency : float | None, optional
            Override for the output sampling frequency. Defaults to the parent's value.
        dtype : DTypeLike | None, optional
            Desired dtype for downstream processing. Defaults to parent's dtype.
        """

        BasePreprocessor.__init__(self, imaging)
        
        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            epoch = SegmentSuite2PImagingEpoch(parent_epoch, motion, epoch_idx, **kwargs)
            self.add_epoch(epoch)

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)

class SegmentSuite2PImagingEpoch(BasePreprocessorEpoch):
    def __init__(self, parent_imaging_epoch: BaseImagingEpoch) -> None:
        """Epoch wrapper that delegates metadata to its parent imaging epoch."""
        BaseImagingEpoch.__init__(self, **parent_imaging_epoch.get_times_kwargs())
        self.parent_imaging_epoch = parent_imaging_epoch

    def get_num_samples(self) -> int:
        """Return the number of samples in the parent epoch."""
        return self.parent_imaging_epoch.get_num_samples()

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: int | slice | Sequence[int] | None = None,
    ) -> NDArray[Any]:
        """Return a frame series for the requested interval and planes.

        Subclasses must override this to apply their specific preprocessing before
        returning the requested frames.
        """

        raise NotImplementedError
