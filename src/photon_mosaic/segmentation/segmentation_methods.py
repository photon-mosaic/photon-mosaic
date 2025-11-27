"""Module defining segmentation methods for photon mosaic and functions"""

from .suite2p import suite2p_segmentation

segmentation_methods = {
    "suite2p": suite2p_segmentation,
}
