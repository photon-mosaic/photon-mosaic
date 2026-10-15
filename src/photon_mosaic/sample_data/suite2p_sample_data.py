from pathlib import Path
from urllib.request import Request, urlopen

import pooch

from photon_mosaic.core import BaseImaging
from photon_mosaic.extractors import read_multi_tiff_multi_page, read_tiff_imaging

_FOLDER_ID = "0B649boZqpYG1R3ota25jdUthSzQ"
_RESOURCE_KEY = "0-wSoqFv5rnE6TERPcJHwQtQ"
_CACHE_DIR = Path.home() / ".photon-mosaic" / "sample_data" / "suite2p"


def download_suite2p_google_drive(num_files: int = 5, lazy: bool = True) -> BaseImaging | list[BaseImaging]:
    """Download the \"official\" suite2p sample TIFFs and return them as imaging.

    The files are consecutive pieces of one single-plane recording, sampled at 30 Hz: at most
    50 files of 200 frames each. They are cached in ``~/.photon-mosaic/sample_data/suite2p``.

    Parameters
    ----------
    num_files : int, default=5
        Number of files to download.
    lazy : bool, default=True
        True returns one imaging over all the files, as one continuous movie, that reads only the
        frames asked for. False returns one imaging per file, each read whole into memory (these
        TIFFs cannot be memory-mapped).

    Returns
    -------
    BaseImaging or list of BaseImaging
        One imaging of ``200 * num_files`` frames if ``lazy``, else one imaging per file.
    """
    if num_files < 1:
        raise ValueError("num_files must be >= 1")

    if num_files > 50:
        raise ValueError("num_files must be <=50")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/embeddedfolderview?id={_FOLDER_ID}&resourcekey={_RESOURCE_KEY}#list"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})) as response:
        html_text = response.read().decode("utf-8")

    # Keep this intentionally simple: extract file ids and per-file resource keys from links.
    parts = html_text.split("/file/d/")[1:]
    file_infos = []
    seen_ids = set()
    for part in parts:
        file_id = part.split("/view", 1)[0].strip()
        if not file_id or file_id in seen_ids:
            continue

        resource_key = _RESOURCE_KEY
        if "resourcekey=" in part:
            resource_key = part.split("resourcekey=", 1)[1].split('"', 1)[0].split("&", 1)[0]

        file_infos.append((file_id, resource_key))
        seen_ids.add(file_id)

        if len(file_infos) >= num_files:
            break

    local_paths = []
    for index, (file_id, resource_key) in enumerate(file_infos, start=1):
        file_name = f"file_00002_{index:05d}.tif"
        local_path = pooch.retrieve(
            url=f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t&resourcekey={resource_key}",
            known_hash=None,
            fname=file_name,
            path=_CACHE_DIR,
            progressbar=True,
        )
        local_paths.append(local_path)

    if not lazy:
        return [read_tiff_imaging(file_path=str(p), sampling_frequency=30) for p in local_paths]
    return read_multi_tiff_multi_page(
        file_paths=[str(p) for p in local_paths], sampling_frequency=30, num_channels=1, num_planes=1
    )
