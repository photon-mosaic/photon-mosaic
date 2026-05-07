"""
RoiAnalyzer: core class for pairing BaseRois and BaseImaging for analysis.

Adapted from SpikeInterface's SortingAnalyzer to work with photon-mosaic's
BaseRois and BaseImaging instead of BaseSorting and BaseRecording.
"""

import importlib
import json
import shutil
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import numpy as np
from spikeinterface.core.baseanalyzer import BaseAnalyzer, BaseAnalyzerExtension
from spikeinterface.core.core_tools import (
    check_json,
    clean_zarr_folder_name,
    is_path_remote,
)

import photon_mosaic

from .baseimaging import BaseImaging
from .baserois import BaseRois
from .binaryrois import BinaryFolderRois
from .imaging_tools import do_imaging_attributes_match, get_imaging_attributes
from .numpyimaging import NumpyRois
from .zarrrois import ZarrRois, save_rois_to_zarr

# ---------------------------------------------------------------------------
# High-level factory functions
# ---------------------------------------------------------------------------


def create_roi_analyzer(
    rois: BaseRois,
    imaging: BaseImaging,
    format: str = "memory",
    folder=None,
    overwrite: bool = False,
    backend_options: dict | None = None,
) -> "RoiAnalyzer":
    """Create a RoiAnalyzer by pairing a BaseRois and a BaseImaging.

    The RoiAnalyzer manages a collection of AnalyzerExtension instances for
    post-processing steps such as trace extraction, dF/F, deconvolution, etc.

    Parameters
    ----------
    rois : BaseRois
        The ROIs object.
    imaging : BaseImaging
        The imaging object.
    format : "memory" | "binary_folder" | "zarr", default: "memory"
        The storage backend. If "binary_folder" or "zarr", ``folder`` must be provided.
    folder : str | Path | None, default: None
        The folder where the analyzer is persisted (required for non-memory formats).
    overwrite : bool, default: False
        If True, overwrite the folder if it already exists.
    backend_options : dict | None, default: None
        Backend-specific options. May contain:

            * storage_options: dict | None (fsspec storage options)
            * saving_options: dict | None (e.g. compression for zarr)

    Returns
    -------
    RoiAnalyzer
        The created RoiAnalyzer object.
    """
    if format != "memory" and not is_path_remote(folder):
        folder = clean_zarr_folder_name(folder) if format == "zarr" else folder
        if Path(folder).is_dir():
            if overwrite:
                shutil.rmtree(folder)
            else:
                raise ValueError(f"Folder {folder} already exists! Use overwrite=True to overwrite it.")

    roi_analyzer = RoiAnalyzer.create(
        rois,
        imaging,
        format=format,  # type: ignore[arg-type]
        folder=folder,
        backend_options=backend_options,
    )
    return roi_analyzer


def load_roi_analyzer(
    folder, load_extensions: bool = True, format: str = "auto", backend_options: dict | None = None
) -> "RoiAnalyzer":
    """Load a RoiAnalyzer from disk.

    Parameters
    ----------
    folder : str or Path
        The folder where the analyzer is stored.
    load_extensions : bool, default: True
        Whether to load all saved extensions.
    format : "auto" | "binary_folder" | "zarr", default: "auto"
        The format of the stored analyzer. "auto" will guess from the path.
    backend_options : dict | None, default: None
        Backend-specific options (e.g. storage_options for remote zarr).

    Returns
    -------
    RoiAnalyzer
        The loaded RoiAnalyzer.
    """
    return RoiAnalyzer.load(folder, load_extensions=load_extensions, format=format, backend_options=backend_options)


# ---------------------------------------------------------------------------
# RoiAnalyzer
# ---------------------------------------------------------------------------


class RoiAnalyzer(BaseAnalyzer):
    """Pair BaseRois and BaseImaging for extensible post-processing analysis.

    Maintains a collection of computed AnalyzerExtension instances. Supports
    three storage backends: ``"memory"``, ``"binary_folder"``, and ``"zarr"``.

    Do not instantiate directly — use :func:`create_roi_analyzer` or
    :meth:`RoiAnalyzer.create`.
    """

    _input_name = "imaging"
    _output_name = "rois"

    def __init__(
        self,
        rois: BaseRois,
        imaging: BaseImaging | None = None,
        imaging_attributes: dict | None = None,
        format: str | None = None,
        backend_options: dict | None = None,
    ):
        # Fast init — validation is done in create / load
        self._init_base(
            output_extractor=rois,
            input_extractor=imaging,
            input_attributes=imaging_attributes,
            format=format,
            backend_options=backend_options,
        )

    # ------------------------------------------------------------------
    # Property aliases (map generic base names to PM terminology)
    # ------------------------------------------------------------------

    @property
    def rois(self):
        return self._output_extractor

    @rois.setter
    def rois(self, value):
        self._output_extractor = value

    @property
    def imaging_attributes(self):
        return self._input_attributes

    @imaging_attributes.setter
    def imaging_attributes(self, value):
        self._input_attributes = value

    @property
    def _imaging(self):
        return self._input_extractor

    @_imaging.setter
    def _imaging(self, value):
        self._input_extractor = value

    @property
    def _temporary_imaging(self):
        return self._temporary_input

    @_temporary_imaging.setter
    def _temporary_imaging(self, value):
        self._temporary_input = value

    # ------------------------------------------------------------------
    # Registry hooks (delegate to module-level functions)
    # ------------------------------------------------------------------

    def _get_extension_class(self, extension_name):
        return get_extension_class(extension_name)

    def _get_children_dependencies(self, extension_name):
        return _get_children_dependencies(extension_name)

    def _sort_extensions_by_dependency(self, extensions):
        return _sort_extensions_by_dependency(extensions)

    def _get_available_extensions(self):
        return get_available_analyzer_extensions()

    def _get_default_extension_params(self, extension_name):
        return get_default_analyzer_extension_params(extension_name)

    def _get_extra_pipeline_kwargs(self):
        return {"check_for_peak_source": False}

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        n_rois = self.get_num_rois()
        n_epochs = self.get_num_epochs()
        assert self.imaging_attributes is not None
        shape = tuple(self.imaging_attributes["shape"])
        txt = (
            f"{cls_name}: {n_rois} ROIs - {shape[0]}x{shape[1]} ({shape[2]} planes) - {n_epochs} epochs - {self.format}"
        )
        if self.format != "memory":
            if is_path_remote(str(self.folder)):
                txt += " (remote)"
        if self.has_imaging():
            txt += " - has imaging"
        if self.has_temporary_imaging():
            txt += " - has temporary imaging"
        ext_txt = f"Loaded {len(self.extensions)} extensions"
        if len(self.extensions) > 0:
            ext_txt += f": {', '.join(self.extensions.keys())}"
        txt += "\n" + ext_txt
        return txt

    # ------------------------------------------------------------------
    # Create / load
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        rois: BaseRois,
        imaging: BaseImaging,
        format: Literal["memory", "binary_folder", "zarr"] = "memory",
        folder=None,
        backend_options: dict | None = None,
    ):
        """Create a new RoiAnalyzer.

        Parameters
        ----------
        rois : BaseRois
            The ROIs object.
        imaging : BaseImaging
            The imaging object.
        format : str
            Storage backend.
        folder : str | Path | None
            Required for non-memory formats.
        backend_options : dict | None
            Backend-specific options.

        Returns
        -------
        RoiAnalyzer
        """
        assert imaging is not None, "An imaging object must be provided to create a RoiAnalyzer"

        # Validate compatibility
        if not np.isclose(rois.sampling_frequency, imaging.sampling_frequency, atol=0.1):
            raise ValueError(
                f"Sampling frequency mismatch: rois={rois.sampling_frequency}, " f"imaging={imaging.sampling_frequency}"
            )
        if rois.shape[:2] != imaging.shape[:2]:
            raise ValueError(f"Spatial shape mismatch: rois={rois.shape[:2]}, imaging={imaging.shape[:2]}")
        if rois.num_planes != imaging.num_planes:
            raise ValueError(f"Number of planes mismatch: rois={rois.num_planes}, imaging={imaging.num_planes}")

        if format == "memory":
            return cls.create_memory(rois, imaging, imaging_attributes=None)
        elif format == "binary_folder":
            return cls.create_binary_folder(
                folder, rois, imaging, imaging_attributes=None, backend_options=backend_options
            )
        elif format == "zarr":
            assert folder is not None, "For format='zarr' folder must be provided"
            if not is_path_remote(folder):
                folder = clean_zarr_folder_name(folder)
            return cls.create_zarr(folder, rois, imaging, imaging_attributes=None, backend_options=backend_options)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load(cls, folder, imaging=None, load_extensions=True, format="auto", backend_options=None):
        """Load from folder or zarr.

        The imaging can be provided if its location has changed.
        """
        if format == "auto":
            if str(folder).endswith(".zarr"):
                format = "zarr"
            else:
                format = "binary_folder"

        if format == "binary_folder":
            analyzer = cls.load_from_binary_folder(folder, imaging=imaging, backend_options=backend_options)
        elif format == "zarr":
            analyzer = cls.load_from_zarr(folder, imaging=imaging, backend_options=backend_options)
        else:
            raise ValueError(f"Unknown format: {format}")

        if not is_path_remote(str(folder)):
            if load_extensions:
                analyzer.load_all_saved_extension()

        return analyzer

    # ------------------------------------------------------------------
    # Memory backend
    # ------------------------------------------------------------------

    @classmethod
    def create_memory(cls, rois, imaging, imaging_attributes):
        if imaging_attributes is None:
            assert imaging is not None
            imaging_attributes = get_imaging_attributes(imaging)
        else:
            imaging_attributes = imaging_attributes.copy()

        # Make an in-memory copy of rois for fast access
        rois_copy = NumpyRois(
            roi_image_masks=np.array(rois.get_roi_image_masks()),
            sampling_frequency=rois.sampling_frequency,
            roi_ids=np.array(rois.roi_ids),
        )
        rois.copy_metadata(rois_copy, only_main=False, ids=rois_copy.roi_ids)

        analyzer = RoiAnalyzer(
            rois=rois_copy,
            imaging=imaging,
            imaging_attributes=imaging_attributes,
            format="memory",
        )
        return analyzer

    # ------------------------------------------------------------------
    # Binary folder backend
    # ------------------------------------------------------------------

    @classmethod
    def create_binary_folder(cls, folder, rois, imaging, imaging_attributes, backend_options):
        folder = Path(folder)
        if folder.is_dir():
            raise ValueError(f"Folder already exists: {folder}")
        folder.mkdir(parents=True)

        # photon-mosaic info
        info = dict(
            version=photon_mosaic.__version__,
            object="RoiAnalyzer",
        )
        with open(folder / "photon_mosaic_info.json", "w") as f:
            json.dump(check_json(info), f, indent=4)

        # Save a copy of the rois
        rois.save(format="binary", folder=folder / "rois")

        # Save imaging provenance
        if imaging is not None:
            if imaging.check_if_json_serializable():
                imaging.dump(folder / "imaging.json", relative_to=folder)
            elif imaging.check_serializability("pickle"):
                imaging.dump(folder / "imaging.pickle", relative_to=folder)
            else:
                warnings.warn("The imaging is not serializable. The imaging link will be lost on reload.")

        # Save rois provenance
        if rois.check_if_json_serializable():
            rois.dump(folder / "rois_provenance.json", relative_to=folder)
        elif rois.check_serializability("pickle"):
            rois.dump(folder / "rois_provenance.pickle", relative_to=folder)
        else:
            warnings.warn("The rois provenance is not serializable. The link will be lost on reload.")

        # Imaging attributes
        attr_folder = folder / "imaging_info"
        attr_folder.mkdir()
        if imaging_attributes is None:
            imaging_attributes = get_imaging_attributes(imaging)
        attr_file = attr_folder / "imaging_attributes.json"
        attr_file.write_text(json.dumps(check_json(imaging_attributes), indent=4), encoding="utf8")

        # Create extensions folder
        (folder / "extensions").mkdir()

        return cls.load_from_binary_folder(folder, imaging=imaging, backend_options=backend_options)

    @classmethod
    def load_from_binary_folder(cls, folder, imaging=None, backend_options=None):
        from spikeinterface.core.loading import load as si_load

        folder = Path(folder)
        assert folder.is_dir(), f"Folder does not exist: {folder}"

        # Load rois
        rois = BinaryFolderRois(folder_path=folder / "rois")

        # Try to load the imaging if not provided
        if imaging is None:
            for ext in ("json", "pickle"):
                fname = folder / f"imaging.{ext}"
                if fname.exists():
                    try:
                        imaging = si_load(fname, base_folder=folder)
                        break
                    except Exception:
                        pass

        # Imaging attributes
        attr_file = folder / "imaging_info" / "imaging_attributes.json"
        if not attr_file.exists():
            raise ValueError(f"Not a valid RoiAnalyzer binary_folder: {folder}")
        with open(attr_file, "r") as f:
            imaging_attributes = json.load(f)

        analyzer = RoiAnalyzer(
            rois=rois,
            imaging=imaging,
            imaging_attributes=imaging_attributes,
            format="binary_folder",
            backend_options=backend_options,
        )
        analyzer.folder = folder
        return analyzer

    # ------------------------------------------------------------------
    # Zarr backend
    # ------------------------------------------------------------------

    # _get_zarr_root is inherited from BaseAnalyzer

    @classmethod
    def create_zarr(cls, folder, rois, imaging, imaging_attributes, backend_options):
        import numcodecs
        import zarr

        backend_options = backend_options or {}
        storage_options = backend_options.get("storage_options", {})
        saving_options = backend_options.get("saving_options", {})

        remote = is_path_remote(str(folder))
        if not remote:
            folder = clean_zarr_folder_name(folder)
            if Path(folder).is_dir():
                raise ValueError(f"Folder already exists: {folder}")

        zarr_root = zarr.open(str(folder), mode="w", storage_options=storage_options)

        # Info
        info = dict(
            version=photon_mosaic.__version__,
            object="RoiAnalyzer",
        )
        zarr_root.attrs["photon_mosaic_info"] = check_json(info)

        # Save imaging provenance
        relative_to = folder if not remote else None
        if imaging is not None:
            rec_dict = imaging.to_dict(relative_to=relative_to, recursive=True)
            if imaging.check_if_json_serializable():
                zarr_rec = np.array([check_json(rec_dict)], dtype=object)
                zarr_root.create_dataset("imaging", data=zarr_rec, object_codec=numcodecs.JSON())
            elif imaging.check_serializability("pickle"):
                zarr_rec = np.array([rec_dict], dtype=object)
                zarr_root.create_dataset("imaging", data=zarr_rec, object_codec=numcodecs.Pickle())
            else:
                warnings.warn("The imaging is not serializable. The link will be lost on reload.")

        # Save rois provenance
        rois_dict = rois.to_dict(relative_to=relative_to, recursive=True)
        if rois.check_if_json_serializable():
            zarr_rois = np.array([check_json(rois_dict)], dtype=object)
            zarr_root.create_dataset("rois_provenance", data=zarr_rois, object_codec=numcodecs.JSON())
        elif rois.check_serializability("pickle"):
            zarr_rois = np.array([rois_dict], dtype=object)
            zarr_root.create_dataset("rois_provenance", data=zarr_rois, object_codec=numcodecs.Pickle())
        else:
            warnings.warn("The rois provenance is not serializable. The link will be lost on reload.")

        # Save imaging attributes
        if imaging_attributes is None:
            imaging_attributes = get_imaging_attributes(imaging)
        imaging_info = zarr_root.create_group("imaging_info")
        imaging_info.attrs["imaging_attributes"] = check_json(imaging_attributes)

        # Save rois data
        rois_group = zarr_root.create_group("rois")
        save_rois_to_zarr(rois, rois_group, saving_options=saving_options)

        # Extensions group
        zarr_root.create_group("extensions")

        zarr.consolidate_metadata(zarr_root.store)

        return cls.load_from_zarr(folder, imaging=imaging, backend_options=backend_options)

    @classmethod
    def load_from_zarr(cls, folder, imaging=None, backend_options=None):
        from spikeinterface.core.loading import load as si_load
        from spikeinterface.core.zarrextractors import super_zarr_open

        backend_options = backend_options or {}
        storage_options = backend_options.get("storage_options", {})

        zarr_root = super_zarr_open(str(folder), mode="r", storage_options=storage_options)

        # Load rois
        rois = ZarrRois(folder, zarr_group_name="rois", storage_options=storage_options)

        # Load imaging if possible
        if imaging is None:
            rec_field = zarr_root.get("imaging")
            if rec_field is not None:
                rec_dict = rec_field[0]
                try:
                    imaging = si_load(rec_dict, base_folder=folder)
                except Exception:
                    imaging = None

        # Imaging attributes
        imaging_attributes = zarr_root["imaging_info"].attrs["imaging_attributes"]

        analyzer = RoiAnalyzer(
            rois=rois,
            imaging=imaging,
            imaging_attributes=imaging_attributes,
            format="zarr",
            backend_options=backend_options,
        )
        analyzer.folder = folder
        return analyzer

    # ------------------------------------------------------------------
    # Temporary imaging
    # ------------------------------------------------------------------

    def set_temporary_imaging(self, imaging: BaseImaging, check_dtype: bool = True):
        """Set a temporary imaging (e.g. a cached/loaded copy) without persisting it.

        Parameters
        ----------
        imaging : BaseImaging
            The imaging object.
        check_dtype : bool, default: True
            Whether to validate dtype match.
        """
        assert self.imaging_attributes is not None
        matches, error = do_imaging_attributes_match(imaging, self.imaging_attributes, check_dtype=check_dtype)
        if not matches:
            raise ValueError(error)
        if self._imaging is not None:
            warnings.warn("RoiAnalyzer imaging is already set. Temporarily replacing it.")
        self._temporary_imaging = imaging

    # ------------------------------------------------------------------
    # ROI property helpers
    # ------------------------------------------------------------------

    def set_roi_property(
        self,
        key: str,
        values: list | np.ndarray | tuple,
        ids: list | np.ndarray | tuple | None = None,
        missing_value: Any = None,
        save: bool = True,
    ) -> None:
        """Set a property on the internal ROIs copy, optionally persisting to disk.

        Parameters
        ----------
        key : str
            Property name.
        values : array-like
            Property values.
        ids : array-like | None
            Subset of ROI ids to set. If None, set for all.
        missing_value : Any
            Fill value for ids not in ``ids``.
        save : bool, default: True
            Whether to persist (binary_folder / zarr only).
        """
        self.rois.set_property(key, values, ids=ids, missing_value=missing_value)
        if not self.is_read_only() and save:
            if self.format == "binary_folder":
                assert isinstance(self.folder, Path)
                props_dir = self.folder / "rois" / "properties"
                props_dir.mkdir(exist_ok=True)
                np.save(props_dir / f"{key}.npy", self.rois.get_property(key))
            elif self.format == "zarr":
                import zarr

                zarr_root = self._get_zarr_root(mode="r+")
                prop_values = self.rois.get_property(key)
                if prop_values.dtype.kind == "O":
                    warnings.warn(f"Property '{key}' not saved (object dtype)")
                else:
                    rois_group = zarr_root["rois"]
                    if "properties" not in rois_group:
                        rois_group.create_group("properties")
                    props_group = rois_group["properties"]
                    if key in props_group:
                        props_group[key][:] = prop_values
                    else:
                        props_group.create_dataset(name=key, data=prop_values, compressor=None)
                    zarr.consolidate_metadata(zarr_root.store)

    def get_roi_property(self, key: str, ids=None) -> np.ndarray:
        """Get a property from the internal ROIs copy."""
        return self.rois.get_property(key, ids=ids)

    # ------------------------------------------------------------------
    # Save / select / copy
    # ------------------------------------------------------------------

    def _save_or_select(
        self,
        format="binary_folder",
        folder=None,
        roi_ids=None,
        backend_options=None,
        verbose=False,
    ) -> "RoiAnalyzer":
        """Internal method used by save_as(), copy(), select_rois(), remove_rois().

        Parameters
        ----------
        format : str
            Target backend.
        folder : str | Path | None
            Target folder (for non-memory).
        roi_ids : array-like | None
            Subset of roi ids to keep. None = keep all.
        backend_options : dict | None
            Backend-specific options.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        RoiAnalyzer
        """
        if self.has_imaging():
            imaging = self._imaging
        elif self.has_temporary_imaging():
            imaging = self._temporary_imaging
        else:
            imaging = None

        # Get the original rois provenance (or fall back to internal copy)
        rois_provenance = self.get_rois_provenance()
        if rois_provenance is None:
            rois_provenance = self.rois

        # Copy any in-memory properties added after creation
        for key in self.rois.get_property_keys():
            if key not in rois_provenance.get_property_keys():
                rois_provenance.set_property(key, self.rois.get_property(key))

        if roi_ids is not None:
            rois_provenance = rois_provenance.select_rois(roi_ids)

        backend_options = backend_options or {}

        if format == "memory":
            new_analyzer = RoiAnalyzer.create_memory(rois_provenance, imaging, self.imaging_attributes)
        elif format == "binary_folder":
            assert folder is not None, "folder must be provided for binary_folder"
            new_analyzer = RoiAnalyzer.create_binary_folder(
                Path(folder), rois_provenance, imaging, self.imaging_attributes, backend_options
            )
        elif format == "zarr":
            assert folder is not None, "folder must be provided for zarr"
            folder = clean_zarr_folder_name(folder)
            new_analyzer = RoiAnalyzer.create_zarr(
                folder, rois_provenance, imaging, self.imaging_attributes, backend_options
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Propagate extensions
        extensions_to_compute = _sort_extensions_by_dependency(
            {ext.extension_name: ext.params for ext in self.extensions.values()}
        )
        for extension_name in extensions_to_compute:
            extension = self.extensions[extension_name]
            new_analyzer.extensions[extension_name] = extension.copy(new_analyzer, roi_ids=roi_ids)

        return new_analyzer

    def save_as(self, format="memory", folder=None, backend_options=None) -> "RoiAnalyzer":
        """Save the analyzer to a different backend.

        Parameters
        ----------
        format : "memory" | "binary_folder" | "zarr"
            Target backend.
        folder : str | Path | None
            Required for non-memory formats.
        backend_options : dict | None
            Backend-specific options.

        Returns
        -------
        RoiAnalyzer
        """
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select(format=format, folder=folder, backend_options=backend_options)

    def select_rois(self, roi_ids, format="memory", folder=None) -> "RoiAnalyzer":
        """Create a new RoiAnalyzer with a subset of ROIs.

        Extensions are filtered to the selected ROI ids.

        Parameters
        ----------
        roi_ids : array-like
            ROI ids to keep.
        format : str
            Target backend.
        folder : str | Path | None
            Required for non-memory formats.

        Returns
        -------
        RoiAnalyzer
        """
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select(format=format, folder=folder, roi_ids=roi_ids)

    def remove_rois(self, remove_roi_ids, format="memory", folder=None) -> "RoiAnalyzer":
        """Create a new RoiAnalyzer with some ROIs removed.

        Parameters
        ----------
        remove_roi_ids : array-like
            ROI ids to remove.
        format : str
            Target backend.
        folder : str | Path | None
            Required for non-memory formats.

        Returns
        -------
        RoiAnalyzer
        """
        roi_ids = self.roi_ids[~np.isin(self.roi_ids, remove_roi_ids)]
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select(format=format, folder=folder, roi_ids=roi_ids)

    def copy(self) -> "RoiAnalyzer":
        """Create an in-memory copy."""
        return self._save_or_select(format="memory", folder=None)

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def imaging(self) -> BaseImaging:
        return self.input_extractor

    @property
    def roi_ids(self) -> np.ndarray:
        return self.rois.roi_ids

    @property
    def sampling_frequency(self) -> float:
        assert self.imaging_attributes is not None
        return float(self.imaging_attributes["sampling_frequency"])

    @property
    def shape(self) -> tuple:
        assert self.imaging_attributes is not None
        return tuple(self.imaging_attributes["shape"])

    @property
    def num_planes(self) -> int:
        return self.shape[2]

    def has_imaging(self) -> bool:
        return self.has_input()

    def has_temporary_imaging(self) -> bool:
        return self.has_temporary_input()

    def get_num_rois(self) -> int:
        return self.rois.get_num_rois()

    def get_num_epochs(self) -> int:
        assert self.imaging_attributes is not None
        return self.imaging_attributes["num_epochs"]

    def get_num_samples(self, epoch_index: int | None = None) -> int:
        if epoch_index is None:
            if self.get_num_epochs() == 1:
                epoch_index = 0
            else:
                raise ValueError("epoch_index must be provided for multi-epoch data.")
        assert self.imaging_attributes is not None
        return self.imaging_attributes["num_samples"][epoch_index]

    def get_total_samples(self) -> int:
        assert self.imaging_attributes is not None
        return sum(self.imaging_attributes["num_samples"])

    def get_total_duration(self) -> float:
        if self.has_imaging() or self.has_temporary_imaging():
            return self.imaging.get_total_duration()
        return self.get_total_samples() / self.sampling_frequency

    def get_dtype(self):
        return np.dtype(self.imaging_attributes["dtype"])

    # is_read_only is inherited from BaseAnalyzer

    def get_rois_provenance(self):
        """Get the original rois object if possible, otherwise None."""
        from spikeinterface.core.loading import load as si_load

        if self.format == "memory":
            return None

        elif self.format == "binary_folder":
            assert isinstance(self.folder, Path)
            for ext in ("json", "pickle"):
                fname = self.folder / f"rois_provenance.{ext}"
                if fname.exists():
                    try:
                        return si_load(fname, base_folder=self.folder)
                    except Exception:
                        pass
            return None

        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "rois_provenance" in zarr_root.keys():
                try:
                    rois_dict = zarr_root["rois_provenance"][0]
                    return si_load(rois_dict, base_folder=self.folder)
                except Exception:
                    pass
            return None

    # ------------------------------------------------------------------
    # Extension management
    # ------------------------------------------------------------------
    # compute, compute_one_extension, compute_several_extensions,
    # get_saved_extension_names, get_extension, load_extension,
    # load_all_saved_extension, delete_extension,
    # get_loaded_extension_names, has_extension,
    # get_computable_extensions, get_default_extension_params
    # are all inherited from BaseAnalyzer


# ---------------------------------------------------------------------------
# Extension dependency utilities
# ---------------------------------------------------------------------------

_possible_extensions: list[type["AnalyzerExtension"]] = []

_extension_children: dict[str, list[str]] = {}


def _get_children_dependencies(extension_name):
    """Recursively find all extensions that depend on ``extension_name``."""
    names = []
    children = _extension_children.get(extension_name, [])
    for child in children:
        if child not in names:
            names.append(child)
        names.extend(_get_children_dependencies(child))
    return list(names)


def _sort_extensions_by_dependency(extensions: dict) -> dict:
    """Sort extensions so parents come before children."""
    ext_list = list(extensions.keys())
    params_list = list(extensions.values())

    i = 0
    while i < len(ext_list):
        name = ext_list[i]
        params = params_list[i]

        ext_class = get_extension_class(name)
        required = ext_class.get_required_dependencies(**params)
        optional = ext_class.get_optional_dependencies(**params)
        all_deps = list(chain.from_iterable(d.split("|") for d in required + optional))

        did_nothing = True
        for dep in all_deps:
            if dep in ext_list[i:]:
                dep_idx = ext_list.index(dep)
                params_list.insert(i, params_list.pop(dep_idx))
                ext_list.insert(i, ext_list.pop(dep_idx))
                did_nothing = False

        if did_nothing:
            i += 1

    return dict(zip(ext_list, params_list))


def register_result_extension(extension_class):
    """Register an AnalyzerExtension subclass so it can be discovered by name."""
    assert issubclass(extension_class, AnalyzerExtension)
    assert extension_class.extension_name is not None, "extension_name must not be None"

    global _possible_extensions

    already = any(extension_class is ext for ext in _possible_extensions)
    if not already:
        assert all(
            extension_class.extension_name != ext.extension_name for ext in _possible_extensions
        ), f"Extension name '{extension_class.extension_name}' already registered"

        _possible_extensions.append(extension_class)

        _extension_children[extension_class.extension_name] = []
        for parent_name in extension_class.get_required_dependencies():
            if "|" in parent_name:
                for name in parent_name.split("|"):
                    _extension_children.setdefault(name, []).append(extension_class.extension_name)
            else:
                _extension_children.setdefault(parent_name, []).append(extension_class.extension_name)


def get_extension_class(extension_name: str, auto_import: bool = True):
    """Get extension class by name, auto-importing if needed."""
    global _possible_extensions
    ext_dict = {ext.extension_name: ext for ext in _possible_extensions}

    if extension_name not in ext_dict:
        if extension_name in _builtin_extensions:
            module = _builtin_extensions[extension_name]
            if auto_import:
                importlib.import_module(module)
                ext_dict = {ext.extension_name: ext for ext in _possible_extensions}
            else:
                raise ValueError(f"Extension '{extension_name}' not registered. Import '{module}' first.")
        else:
            warnings.warn(f"Extension '{extension_name}' is unknown.")
            return None

    return ext_dict.get(extension_name)


def get_available_analyzer_extensions() -> list[str]:
    """Get all built-in extension names."""
    return list(_builtin_extensions.keys())


def get_default_analyzer_extension_params(extension_name: str) -> dict:
    """Get the default params for an extension."""
    import inspect

    ext_class = get_extension_class(extension_name)
    sig = inspect.signature(ext_class._set_params)
    return {k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty}


# ---------------------------------------------------------------------------
# AnalyzerExtension
# ---------------------------------------------------------------------------


class AnalyzerExtension(BaseAnalyzerExtension):
    """Extension class for RoiAnalyzer.

    Adds RoiAnalyzer-specific features on top of BaseAnalyzerExtension:
      * ``roi_analyzer`` property
      * ``need_imaging`` attribute
      * ``function_factory()`` for backwards-compatible ``compute_xxx()`` helpers

    Subclasses must set ``extension_name`` and implement:

    * ``_set_params(**params)`` — validate / clean params
    * ``_run(**kwargs)`` — populate ``self.data``
    * ``_select_extension_data(roi_ids)`` — filter data to a subset
    * ``_get_data()`` — return the computed result

    Optionally (for pipeline extensions):

    * ``_get_pipeline_nodes()`` — if ``use_nodepipeline = True``
    """

    need_imaging = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Sync PM-specific need_imaging → generic need_input
        if "need_imaging" in cls.__dict__:
            cls.need_input = cls.__dict__["need_imaging"]

    def __init__(self, roi_analyzer):
        super().__init__(roi_analyzer)

    # ------------------------------------------------------------------
    # PM-specific properties
    # ------------------------------------------------------------------

    @property
    def roi_analyzer(self):
        return self.analyzer

    # ------------------------------------------------------------------
    # CSV index coercion hook
    # ------------------------------------------------------------------

    def _get_entity_ids(self):
        return self.roi_analyzer.roi_ids

    # ------------------------------------------------------------------
    # Copy with PM parameter name
    # ------------------------------------------------------------------

    def copy(self, new_roi_analyzer, roi_ids=None):
        """Copy this extension to a new RoiAnalyzer, optionally filtering to roi_ids."""
        return super().copy(new_roi_analyzer, ids=roi_ids)

    # ------------------------------------------------------------------
    # Override get_default_params to use the module-level function
    # ------------------------------------------------------------------

    @classmethod
    def get_default_params(cls):
        return get_default_analyzer_extension_params(cls.extension_name)

    # ------------------------------------------------------------------
    # Function factory for backwards-compatible compute_xxx() helpers
    # ------------------------------------------------------------------

    @classmethod
    def function_factory(cls):
        class FuncWrapper:
            def __init__(self, extension_name):
                self.extension_name = extension_name

            def __call__(self, roi_analyzer, *args, **kwargs):
                if not isinstance(roi_analyzer, RoiAnalyzer):
                    raise ValueError(f"compute_{self.extension_name}() needs a RoiAnalyzer instance")
                ext = roi_analyzer.compute(cls.extension_name, *args, **kwargs)
                return ext.get_data()

        func = FuncWrapper(cls.extension_name)
        func.__doc__ = cls.__doc__
        return func


# ---------------------------------------------------------------------------
# Built-in extensions registry
# ---------------------------------------------------------------------------

_builtin_extensions: dict[str, str] = {
    "fluorescence": "photon_mosaic.core",
}
