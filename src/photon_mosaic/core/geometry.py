"""Helpers for the per-field ``geometry`` metadata carried by :class:`BaseImaging`.

Multi-field (e.g. mesoscope) acquisitions attach a lateral-placement dictionary to each field's
:class:`BaseImaging` via its :attr:`~photon_mosaic.core.baseimaging.BaseImaging.geometry` attribute
(see that property for the conventional keys). This module reads across a collection of such fields:
:func:`geometry_table` tabulates their placement. Geometry *validation* (checking a geometry dict's
keys, types and array shapes) is intended to live here too and will be added alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .baseimaging import BaseImaging

if TYPE_CHECKING:
    import pandas as pd


def _to_microns(
    value_xy: list[float] | None,
    microns_per_unit: float | None,
) -> tuple[float | None, float | None]:
    """Scale an ``(x, y)`` pair to microns, or ``(None, None)`` if either input is missing."""
    if value_xy is None or microns_per_unit is None:
        return (None, None)
    return (value_xy[0] * microns_per_unit, value_xy[1] * microns_per_unit)


def geometry_table(imagings: Sequence[BaseImaging]) -> pd.DataFrame | list[dict]:
    """Tabulate the per-field geometry of a collection of imaging fields, one row per field.

    Pixel shape, plane count, frame count and sampling frequency are read from each field itself;
    identity and lateral placement (``field_name``, ``field_uuid``, ``units``, ``center_xy``,
    ``size_xy``) from its ``geometry`` dict, with matching micron columns derived from
    ``microns_per_unit`` when present. The row ``index`` is the field's position in ``imagings``. The
    3x3 ``affine`` is omitted for readability (reach it via ``imagings[i].geometry["affine"]``). Fields
    with no geometry get null placement columns.

    Parameters
    ----------
    imagings : Sequence[BaseImaging]
        The imaging fields to tabulate.

    Returns
    -------
    pandas.DataFrame | list[dict]
        A ``pandas.DataFrame`` if pandas is available, otherwise a list of row dicts.
    """
    rows = []
    for index, imaging in enumerate(imagings):
        geometry = imaging.geometry or {}
        center_xy = geometry.get("center_xy")
        size_xy = geometry.get("size_xy")
        microns_per_unit = geometry.get("microns_per_unit")
        center_x, center_y = center_xy if center_xy is not None else (None, None)
        size_x, size_y = size_xy if size_xy is not None else (None, None)
        # Micron columns mirror the reference-unit ones, populated only when the scale is known.
        center_x_um, center_y_um = _to_microns(center_xy, microns_per_unit)
        size_x_um, size_y_um = _to_microns(size_xy, microns_per_unit)
        height, width = imaging.shape[:2]
        rows.append(
            dict(
                index=index,
                field_name=geometry.get("field_name"),
                field_uuid=geometry.get("field_uuid"),
                height=height,
                width=width,
                num_planes=imaging.num_planes,
                num_frames=imaging.get_total_frames(),
                sampling_frequency=round(imaging.sampling_frequency, 4),
                units=geometry.get("units"),
                center_x=center_x,
                center_y=center_y,
                size_x=size_x,
                size_y=size_y,
                center_x_um=center_x_um,
                center_y_um=center_y_um,
                size_x_um=size_x_um,
                size_y_um=size_y_um,
            )
        )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        # Fall back to a plain list of row dicts when pandas is unavailable.
        return rows
