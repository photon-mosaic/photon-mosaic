from pathlib import Path

import pooch

from photon_mosaic.core import select_planes
from photon_mosaic.extractors import read_multi_tiff_multi_page

_CACHE_DIR = Path.home() / ".photon-mosaic" / "sample_data" / "aind"


def download_aind_multiregion_data(
    fields_of_view: list[int] = [0],
    regions: list[int] = [0],
):
    """Load AIND multi-region sample data.

    Downloads and caches local z-stack TIFF files from the AIND open-data S3
    bucket, reads them as imaging objects, selects requested planes, and
    returns the resulting mapping.

    Data has 4 regions, each with 2 single-plane fields of view, and 1620 time frames.
    Each region file is 1.7GB and downloaded individually, so calling this
    function with a new region for the first time will be relatively slow.

    Parameters
    ----------
    fields_of_view : list[int], default: [0]
        Field of view indices in the loaded imaging objects. Indices are in [0,1].
    regions : list[int], default: [0]
        Local z-stack region indices to download and load.

    Returns
    -------
    dict[tuple[int, int], BaseImaging]
        Mapping from ``(field_of_view, region)`` to single-plane imaging objects.
    """
    bucket_name = "aind-open-data"
    imagings = {}

    for field_of_view in fields_of_view:
        for region in regions:
            file_key = "multiplane-ophys_826616_2026-01-02_11-58-49/pophys/" f"1483779513_local_z_stack{region}.tiff"
            url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

            local_path = pooch.retrieve(
                url=url,
                known_hash=None,
                path=_CACHE_DIR,
                progressbar=True,
            )

            imaging = read_multi_tiff_multi_page(
                file_paths=[local_path], sampling_frequency=10.63, dimension_order="ZCT", num_planes=2
            )

            imaging = select_planes(imaging=imaging, plane_ids=[field_of_view])
            imagings[(field_of_view, region)] = imaging

    return imagings
