from pathlib import Path

import pooch

from photon_mosaic.core import select_planes
from photon_mosaic.extractors import read_scan_image_imaging

_CACHE_DIR = Path.home() / ".photon-mosaic" / "sample_data" / "aind"


def download_aind_multiregion_data(
    channels: list[int] = [1],
    regions: list[int] = [0],
    plane_ids=[41],
):
    """Load AIND multi-region sample data.

    Downloads and caches local z-stack TIFF files from the AIND open-data S3
    bucket, reads them as imaging objects, selects requested planes, and
    returns the resulting mapping.

    Data has 2 channels, 81 planes, 4 regions and 20 time frames.
    Each region file is 1.7GB and downloaded individually, so calling this
    function with a new region for the first time will be relatively slow.

    Parameters
    ----------
    channels : list[int], default: [1]
        Channel indices to label in the loaded imaging objects. Channels are in [1,2].
    regions : list[int], default: [0]
        Local z-stack region indices to download and load.
    plane_ids : list[int], default: [41]
        Plane identifiers to keep via :func:`photon_mosaic.core.select_planes`.
        Default is the middle plane.

    Returns
    -------
    dict[tuple[int, int], BaseImaging]
        Mapping from ``(channel, region)`` to plane-selected imaging objects.
    """
    bucket_name = "aind-open-data"
    imagings = {}

    for channel in channels:
        for region in regions:
            file_key = "multiplane-ophys_826616_2026-01-02_11-58-49/pophys/" f"1483779513_local_z_stack{region}.tiff"
            url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

            local_path = pooch.retrieve(
                url=url,
                known_hash=None,
                path=_CACHE_DIR,
                progressbar=True,
            )

            imaging = read_scan_image_imaging(
                file_path=local_path,
                channel_name=f"Channel {channel}",
            )

            imaging = select_planes(
                imaging=imaging,
                plane_ids=plane_ids,
            )

            imagings[(channel, region)] = imaging

    return imagings
