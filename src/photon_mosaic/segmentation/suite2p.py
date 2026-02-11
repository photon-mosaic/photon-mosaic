import importlib.util
import shutil
from pathlib import Path

from spikeinterface.core.job_tools import fix_job_kwargs, split_job_kwargs

from photon_mosaic.core import BaseImaging
from photon_mosaic.core.utils import PathType
from spikeinterface.core import write_binary

suite2p_spec = importlib.util.find_spec("suite2p")
if suite2p_spec is not None:
    HAVE_SUITE2P = True
else:
    HAVE_SUITE2P = False


def suite2p_segmentation(
    imaging: BaseImaging, folder: PathType | None = None, remove_existing_folder: bool = False, **suite2p_params
) -> dict:
    """
    Perform Suite2p segmentation on the provided imaging data.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging data to be segmented.
    **suite2p_params : dict
        Additional parameters to configure Suite2p processing.

    Returns
    -------
    dict
        A dictionary containing the results of the Suite2p extraction.
    """
    if not HAVE_SUITE2P:
        raise ImportError("Suite2p is not installed. Please install Suite2p to use this function.")
    from photon_mosaic.extractors import read_suite2p_rois
    from suite2p import io, pipeline, default_ops

    # TODO: handle multi-segment
    if imaging.get_num_epochs() > 1:
        raise NotImplementedError("Suite2p segmentation for multi-segment imaging is not yet implemented.")

    if folder is None:
        folder = Path("./suite2p_output/")
    folder = Path(folder)
    if folder.is_dir():
        if not remove_existing_folder:
            raise FileExistsError(f"The folder {folder} already exists. To overwrite, set remove_existing_folder=True.")
        else:
            shutil.rmtree(folder)

    folder.mkdir(parents=True)

    Lx, Ly = imaging.image_shape
    n_frames = imaging.get_num_samples()

    if imaging.is_binary_compatible():
        binary_dict = imaging.get_binary_description()
        file_path = binary_dict["file_paths"][0]
        # f_reg gets overritten so we need to copy the file to the output folder
        dest_file = folder / Path(file_path).name
        if not dest_file.exists():
            shutil.copyfile(file_path, dest_file)
        filename = str(dest_file)
    else:
        _, job_kwargs = split_job_kwargs(suite2p_params)
        file_path = Path(folder) / "imaging_binary.dat"
        write_binary(
            imaging,
            file_paths=file_path,
            add_file_extension=False,
            dtype="float32",
            **fix_job_kwargs(job_kwargs),
        )
        filename = str(file_path)

    # Initialize Suite2p with the provided parameters
    ops = default_ops()
    if suite2p_params:
        ops.update(suite2p_params)
    ops["fs"] = imaging.sampling_frequency
    ops["save_path"] = str(folder)
    # We only want to run segmentation
    ops["run_registration"] = False
    ops["roi_detect"] = True
    ops["spikedetect"] = False
    ops["ops_path"] = folder / "ops.npy"
    with io.BinaryFile(Lx=Lx, Ly=Ly, filename=filename, n_frames=n_frames) as f_reg:
        pipeline(f_reg=f_reg, run_registration=False, ops=ops)

    rois = read_suite2p_rois(folder)
    return rois
