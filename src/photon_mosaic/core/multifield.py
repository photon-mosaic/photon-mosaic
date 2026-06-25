"""Container for imaging fields that do not share a common pixel grid.

A multi-plane :class:`BaseImaging` is a *regular* volume: every plane shares one ``(height, width)``
pixel grid and the plane axis is a regularly sampled depth. Some acquisitions (e.g. mesoscope
multi-ROI recordings) instead capture several independent fields of view, each with its own pixel
shape and physical position. These cannot be honestly stacked into a single ``BaseImaging``.

:class:`MultiFieldImaging` keeps each field as its own :class:`BaseImaging` and records its lateral
placement in a parallel :class:`FieldGeometry` table. The two kinds of multiplicity live on two
levels:

* within a field — the ``planes`` axis of one ``BaseImaging`` (a regular z-stack);
* across fields — this container (free pixel shape, free lateral position).

A field that is itself a z-stack is simply a ``BaseImaging`` with ``num_planes > 1``, so multiple
z-stacks compose through the same container. Depth therefore belongs to a field's own plane axis,
not to the container; :class:`FieldGeometry` carries only lateral placement and identity. There is
deliberately no volume/stack view: the fields do not form a regular grid.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .baseimaging import BaseImaging


@dataclass
class FieldGeometry:
    """Identity and lateral placement of one imaging field within the sample.

    Spatial quantities are stored in the source's physical reference units (for ScanImage, scanner
    angle units), not pixels. When the microns-per-unit scale is known, the ``center_xy`` / ``size_xy``
    values are also available in micrometers through the :attr:`center_xy_um` / :attr:`size_xy_um`
    properties, which derive from a single stored scale so the two representations cannot drift. Depth
    (``z``) is intentionally absent: it varies across a field's planes and therefore belongs to the
    field's own plane axis, not to this per-field record.

    Attributes
    ----------
    index : int
        The source-assigned field index.
    name : str | None
        Human-readable field name. Not guaranteed unique.
    uuid : str | None
        Unique field identifier, when the source provides one.
    center_xy : tuple[float, float] | None
        In-plane physical center of the field, in scanner-angle units.
    size_xy : tuple[float, float] | None
        In-plane physical extent of the field, in scanner-angle units.
    affine : np.ndarray | None
        3x3 pixel-to-physical (in-plane) transform, in scanner-angle units.
    microns_per_scanner_unit : float | None
        Micrometers subtended by one scanner-angle unit (for ScanImage, ``SI.objectiveResolution``).
        Set when known; the ``*_um`` properties use it to convert the scanner-unit geometry to microns.
    """

    index: int
    name: str | None = None
    uuid: str | None = None
    center_xy: tuple[float, float] | None = None
    size_xy: tuple[float, float] | None = None
    affine: np.ndarray | None = None
    microns_per_scanner_unit: float | None = None

    @property
    def center_xy_um(self) -> tuple[float, float] | None:
        """In-plane physical center in micrometers, or None if the center or scale is unknown."""
        return self._to_microns(self.center_xy)

    @property
    def size_xy_um(self) -> tuple[float, float] | None:
        """In-plane physical extent in micrometers, or None if the size or scale is unknown."""
        return self._to_microns(self.size_xy)

    def _to_microns(self, value_xy: tuple[float, float] | None) -> tuple[float, float] | None:
        """Scale an (x, y) scanner-unit pair to micrometers, or None if either input is unavailable."""
        if value_xy is None or self.microns_per_scanner_unit is None:
            return None
        return (value_xy[0] * self.microns_per_scanner_unit, value_xy[1] * self.microns_per_scanner_unit)


def _resolve_field_selector(
    geometry: list[FieldGeometry],
    *,
    index: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
) -> int:
    """Resolve exactly one of index/name/uuid to a field position within ``geometry``.

    uuid is matched case-insensitively. Raises ``ValueError`` if not exactly one selector is given or
    a name/uuid is ambiguous, and ``KeyError`` if a name/uuid matches nothing.
    """
    if sum(selector is not None for selector in (index, name, uuid)) != 1:
        raise ValueError("pass exactly one of index, name, or uuid")
    if index is not None:
        return index
    if uuid is not None:
        matches = [i for i, geom in enumerate(geometry) if (geom.uuid or "").lower() == uuid.lower()]
    else:
        matches = [i for i, geom in enumerate(geometry) if geom.name == name]

    selector_kind, selector_value = ("uuid", uuid) if uuid is not None else ("name", name)
    if not matches:
        raise KeyError(f"no field matching {selector_kind}={selector_value!r}")
    if len(matches) > 1:
        raise ValueError(f"{selector_kind}={selector_value!r} is ambiguous: indices {matches}")
    return matches[0]


class MultiFieldImaging:
    """A collection of imaging fields with independent pixel grids and physical positions.

    Each field is a standalone :class:`BaseImaging`, so every per-field capability (ROIs,
    ``RoiAnalyzer``, fluorescence) applies unchanged. The container adds cohesion: shared identity,
    selection by index/name/uuid, and a geometry table describing where each field is located in
    space.
    """

    def __init__(self, fields: list[BaseImaging], geometry: list[FieldGeometry]) -> None:
        """Pair a list of field imagings with their geometry records.

        Parameters
        ----------
        fields : list[BaseImaging]
            One imaging object per field.
        geometry : list[FieldGeometry]
            Placement records, aligned by position with ``fields``.
        """
        if len(fields) != len(geometry):
            raise ValueError(f"fields ({len(fields)}) and geometry ({len(geometry)}) length mismatch")

        # Each field must be a real imaging container; the geometry table only places it in space.
        for field in fields:
            if not isinstance(field, BaseImaging):
                raise TypeError(f"each field must be a BaseImaging, got {type(field).__name__}")

        # Fields are expected to be co-acquired (one shared timebase); flag a mixed set rather than fail.
        sampling_frequencies = {round(field.sampling_frequency, 4) for field in fields}
        if len(sampling_frequencies) > 1:
            warnings.warn(
                f"fields have heterogeneous sampling frequencies: {sorted(sampling_frequencies)}",
                UserWarning,
            )

        self._fields = list(fields)
        self._geometry = list(geometry)

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self._fields)

    def __iter__(self) -> Iterator[BaseImaging]:
        """Iterate over the field imagings."""
        return iter(self._fields)

    def __getitem__(self, index: int | slice) -> BaseImaging | list[BaseImaging]:
        """Index fields by position (int or slice)."""
        return self._fields[index]

    @property
    def fields(self) -> list[BaseImaging]:
        """The per-field imaging objects."""
        return self._fields

    @property
    def geometry(self) -> list[FieldGeometry]:
        """The per-field geometry records, aligned with :attr:`fields`."""
        return self._geometry

    def iter_fields(self) -> Iterator[tuple[FieldGeometry, BaseImaging]]:
        """Iterate over ``(geometry, imaging)`` pairs."""
        return zip(self._geometry, self._fields)

    def get_field(
        self,
        *,
        index: int | None = None,
        name: str | None = None,
        uuid: str | None = None,
    ) -> BaseImaging:
        """Return a single field selected by exactly one of index, name, or uuid.

        Parameters
        ----------
        index : int | None
            Position of the field.
        name : str | None
            Field name. Names are not guaranteed unique; an ambiguous name raises ``ValueError``.
        uuid : str | None
            Field uuid, matched case-insensitively (the robust selector).

        Returns
        -------
        BaseImaging
            The selected field.
        """
        return self._fields[_resolve_field_selector(self._geometry, index=index, name=name, uuid=uuid)]

    def geometry_table(self):
        """Return the geometry as a table.

        Returns a ``pandas.DataFrame`` if pandas is available, otherwise a list of row dicts. Pixel
        shape, frame count, and sampling frequency are read from the fields themselves; the 3x3
        affine is omitted for readability (reach it via ``self.geometry[i].affine``).
        """
        rows = []
        for geom, field in self.iter_fields():
            center_x, center_y = geom.center_xy if geom.center_xy is not None else (None, None)
            size_x, size_y = geom.size_xy if geom.size_xy is not None else (None, None)
            # Micron columns mirror the scanner-unit ones, populated only when the scale is known.
            center_x_um, center_y_um = geom.center_xy_um if geom.center_xy_um is not None else (None, None)
            size_x_um, size_y_um = geom.size_xy_um if geom.size_xy_um is not None else (None, None)
            height, width = field.shape[:2]
            rows.append(
                dict(
                    index=geom.index,
                    name=geom.name,
                    uuid=geom.uuid,
                    height=height,
                    width=width,
                    num_planes=field.num_planes,
                    num_frames=field.get_total_frames(),
                    sampling_frequency=round(field.sampling_frequency, 4),
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

    def __repr__(self) -> str:
        lines = [f"MultiFieldImaging: {len(self)} fields"]
        for geom, field in self.iter_fields():
            height, width = field.shape[:2]
            lines.append(
                f"  [{geom.index}] {geom.name!r} uuid={geom.uuid} "
                f"{height}x{width}px x{field.num_planes} planes "
                f"center_xy={geom.center_xy} size_xy={geom.size_xy}"
            )
        return "\n".join(lines)
