from .baseimaging import BaseImaging


def _normalize_epoch_indices(epoch_indices: int | list[int], num_epochs: int) -> list[int]:
	"""Normalize and validate epoch indices requested by the user."""
	if type(epoch_indices) is int:
		normalized_epoch_indices = [epoch_indices]
	elif isinstance(epoch_indices, list):
		normalized_epoch_indices = epoch_indices
	else:
		raise TypeError("epoch_indices must be an int or a list of ints")

	if len(normalized_epoch_indices) == 0:
		raise ValueError("epoch_indices must contain at least one index")

	for epoch_index in normalized_epoch_indices:
		if type(epoch_index) is not int:
			raise TypeError("All epoch indices must be ints")
		if not 0 <= epoch_index < num_epochs:
			raise IndexError(f"Epoch index {epoch_index} out of range for imaging with {num_epochs} epochs")

	return normalized_epoch_indices


class SelectEpochImaging(BaseImaging):
	"""Proxy imaging object exposing only selected epochs from a parent imaging."""

	def __init__(self, imaging: BaseImaging, epoch_indices: int | list[int]):
		normalized_epoch_indices = _normalize_epoch_indices(epoch_indices, imaging.get_num_epochs())

		BaseImaging.__init__(self, sampling_frequency=imaging.sampling_frequency, shape=imaging.shape)
		imaging.copy_metadata(self)

		for epoch_index in normalized_epoch_indices:
			imaging_epoch = imaging.epochs[epoch_index]
			self.add_epoch(imaging_epoch)

		self._parent = imaging
		self._kwargs = {
			"imaging": imaging,
			"epoch_indices": normalized_epoch_indices,
		}


def split_epochs(imaging: BaseImaging, epoch_indices: int | list[int]) -> SelectEpochImaging:
	"""Return a proxy imaging object with only the requested epochs."""
	return SelectEpochImaging(imaging=imaging, epoch_indices=epoch_indices)

