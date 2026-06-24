from pathlib import Path
from urllib.request import Request, urlopen

import pooch

from photon_mosaic.extractors import read_tiff_imaging

_FOLDER_ID = "0B649boZqpYG1R3ota25jdUthSzQ"
_RESOURCE_KEY = "0-wSoqFv5rnE6TERPcJHwQtQ"
_CACHE_DIR = Path.home() / ".photon-mosaic" / "sample_data" / "suite2p"


def download_suite2p_google_drive(num_files: int = 5) -> list:
    """Download and return \"official\" suite2p sample TIFFs as a list of imaging objects.

    Assumes they are sampled at 30 Hz. There are maximum 50 TIFF files available, with 200 frames each.
    Each TIFF file is read into its own imaging object, using a generic tiff imaging reader.
    Note that this reader loads the whole image into memory.

    Parameters
    ----------
    num_files : int, default=5
        Number of files to load.

    Returns
    -------
    list
        A list of imaging objects, one per TIFF file.
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

    return [read_tiff_imaging(file_path=str(p), sampling_frequency=30) for p in local_paths]
