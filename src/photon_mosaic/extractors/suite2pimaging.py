"""Suite2p registered-movie imaging extractor.

:class:`Suite2pImaging` reloads the registered binary movie produced by suite2p
(``data.bin``) from a suite2p output folder. It is a thin wrapper over
:class:`~photon_mosaic.core.binaryimaging.BinaryImaging` that reads
``ops.npy`` to recover the frame shape, sampling rate, and dtype.
"""

from pathlib import Path

import numpy as np

from photon_mosaic.core import BinaryImaging


class Suite2pImaging(BinaryImaging):
    """Registered movie loaded from a suite2p output folder."""

    def __init__(
        self,
        folder_path: str | Path,
        *,
        binary_file: str = "data.bin",
        binary_dtype: str = "int16",
    ) -> None:
        """Reload the registered binary movie produced by suite2p.

        Parameters
        ----------
        folder_path : str or Path
            Folder containing ``ops.npy`` and the registered binary movie.
        binary_file : str, default: "data.bin"
            Name of the registered binary movie within ``folder_path``.
        binary_dtype : str, default: "int16"
            Dtype of the registered binary movie (suite2p writes ``int16``).
        """
        folder_path = Path(folder_path)
        ops_file = folder_path / "ops.npy"
        binary_path = folder_path / binary_file
        if not ops_file.is_file():
            raise FileNotFoundError(f"No ops.npy file found in {folder_path}")
        if not binary_path.is_file():
            raise FileNotFoundError(f"No {binary_file} file found in {folder_path}")

        ops = np.load(ops_file, allow_pickle=True).item()
        # stat ypix/xpix and the registered binary are in full-frame coordinates,
        # so use Ly/Lx (not the cropped Lyc/Lxc) as the frame shape.
        height = int(ops.get("Ly", ops.get("Lyc")))
        width = int(ops.get("Lx", ops.get("Lxc")))
        sampling_frequency = float(ops["fs"])

        BinaryImaging.__init__(
            self,
            file_paths=binary_path,
            sampling_frequency=sampling_frequency,
            shape=(height, width, 1),
            dtype=binary_dtype,
        )

        self._kwargs = {
            "folder_path": str(folder_path.absolute()),
            "binary_file": binary_file,
            "binary_dtype": binary_dtype,
        }
