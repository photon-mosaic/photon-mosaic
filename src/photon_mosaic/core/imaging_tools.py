"""Utilities for extracting and validating imaging attributes."""

from .baseimaging import BaseImaging


def get_imaging_attributes(imaging: BaseImaging) -> dict:
    """Extract key attributes from a BaseImaging object for serialization.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging object to extract attributes from.

    Returns
    -------
    dict
        Dictionary with keys: sampling_frequency, shape, num_epochs, num_samples, dtype.
    """
    num_epochs = imaging.get_num_epochs()
    num_samples = [imaging.get_num_frames(epoch_index=i) for i in range(num_epochs)]
    t_starts = imaging._get_t_starts()

    return dict(
        sampling_frequency=float(imaging.sampling_frequency),
        shape=list(imaging.shape),
        num_epochs=num_epochs,
        num_samples=num_samples,
        dtype=str(imaging.get_dtype()),
        t_starts=t_starts,
    )


def do_imaging_attributes_match(imaging: BaseImaging, attributes: dict, check_dtype: bool = True) -> tuple[bool, str]:
    """Validate that an imaging object matches stored attributes.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging object to validate.
    attributes : dict
        The stored attributes to compare against.
    check_dtype : bool, default: True
        Whether to also check that the dtype matches.

    Returns
    -------
    tuple[bool, str]
        A tuple of (matches, error_message). If matches is True, error_message is empty.
    """
    if float(imaging.sampling_frequency) != attributes["sampling_frequency"]:
        return False, (
            f"Sampling frequency mismatch: imaging has {imaging.sampling_frequency}, "
            f"expected {attributes['sampling_frequency']}"
        )

    if list(imaging.shape) != list(attributes["shape"]):
        return False, (f"Shape mismatch: imaging has {imaging.shape}, expected {tuple(attributes['shape'])}")

    if imaging.get_num_epochs() != attributes["num_epochs"]:
        return False, (
            f"Number of epochs mismatch: imaging has {imaging.get_num_epochs()}, "
            f"expected {attributes['num_epochs']}"
        )

    for i in range(imaging.get_num_epochs()):
        if imaging.get_num_frames(epoch_index=i) != attributes["num_samples"][i]:
            return False, (
                f"Number of frames mismatch in epoch {i}: imaging has "
                f"{imaging.get_num_frames(epoch_index=i)}, expected {attributes['num_samples'][i]}"
            )

    if check_dtype:
        if str(imaging.get_dtype()) != attributes["dtype"]:
            return False, (f"Dtype mismatch: imaging has {imaging.get_dtype()}, expected {attributes['dtype']}")

    return True, ""
