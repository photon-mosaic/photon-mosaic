"""Helpers over a collection of imaging fields (a ``Sequence[BaseImaging]``).

Multi-field (e.g. mesoscope) acquisitions are represented as a plain sequence of independent
:class:`BaseImaging` objects, each carrying its own spatial placement in its ``geometry`` dict (see
:attr:`~photon_mosaic.core.baseimaging.BaseImaging.geometry`). No container object is needed — a list
already provides length, iteration and indexing. This module adds the operations a bare sequence does
not: selecting a field by identity now, and (planned) selecting fields by spatial location.
"""

from __future__ import annotations

from typing import Sequence

from .baseimaging import BaseImaging


def _geometry_value(imaging: BaseImaging, key: str):
    """Return ``imaging.geometry[key]``, or None when the field has no geometry or lacks the key."""
    return (imaging.geometry or {}).get(key)


def get_field(
    imagings: Sequence[BaseImaging],
    *,
    index: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
) -> BaseImaging:
    """Select a single field from ``imagings`` by exactly one of index, name, or uuid.

    ``name`` and ``uuid`` are read from each field's ``geometry`` dict; ``uuid`` is matched
    case-insensitively (the robust selector, since names are not guaranteed unique). ``index`` is the
    field's position in ``imagings``.

    Parameters
    ----------
    imagings : Sequence[BaseImaging]
        The imaging fields to select from.
    index : int | None
        Position of the field.
    name : str | None
        Field name. An ambiguous name raises ``ValueError``.
    uuid : str | None
        Field uuid, matched case-insensitively.

    Returns
    -------
    BaseImaging
        The selected field.

    Raises
    ------
    ValueError
        If not exactly one of index/name/uuid is given, or a name/uuid is ambiguous.
    KeyError
        If a name/uuid matches no field.
    """
    if sum(selector is not None for selector in (index, name, uuid)) != 1:
        raise ValueError("pass exactly one of index, name, or uuid")
    if index is not None:
        return imagings[index]

    if uuid is not None:
        matches = [i for i, im in enumerate(imagings) if (_geometry_value(im, "uuid") or "").lower() == uuid.lower()]
        selector_kind, selector_value = "uuid", uuid
    else:
        # Reaching here means index and uuid are None, so the exactly-one guard leaves name set.
        assert name is not None
        matches = [i for i, im in enumerate(imagings) if _geometry_value(im, "name") == name]
        selector_kind, selector_value = "name", name

    if not matches:
        raise KeyError(f"no field matching {selector_kind}={selector_value!r}")
    if len(matches) > 1:
        raise ValueError(f"{selector_kind}={selector_value!r} is ambiguous: indices {matches}")
    return imagings[matches[0]]


def fields_containing_point(
    imagings: Sequence[BaseImaging],
    point_xy: tuple[float, float],
) -> list[BaseImaging]:
    """Return the fields whose lateral extent contains ``point_xy``.  [PLANNED — not implemented]

    Intended to test ``point_xy`` (in the fields' physical reference units) against each field's
    in-plane bounds derived from its ``geometry`` — ``center_xy`` / ``size_xy`` for an axis-aligned
    extent, or the ``affine`` for a rotated one — and return those that contain it. Not implemented
    yet.

    Parameters
    ----------
    imagings : Sequence[BaseImaging]
        The imaging fields to test.
    point_xy : tuple[float, float]
        The in-plane query point, in the fields' physical reference units.

    Returns
    -------
    list[BaseImaging]
        The fields whose extent contains the point.
    """
    raise NotImplementedError("spatial field selection is not implemented yet")
