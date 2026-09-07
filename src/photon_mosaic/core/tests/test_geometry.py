"""Tests for the geometry_table helper over a collection of imaging fields."""

import math

from photon_mosaic.core import geometry_table
from photon_mosaic.core.generators import generate_random_imaging


def _is_null(value) -> bool:
    """True for a missing placement value, whether None (list form) or NaN (pandas column)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _make_field(height: int, width: int, sampling_frequency: float = 10.0, seed: int = 0):
    """Create a small random single-plane imaging field."""
    return generate_random_imaging(
        num_frames=5, height=height, width=width, sampling_frequency=sampling_frequency, seed=seed
    )


def _rows(table):
    """Normalise the table (DataFrame or list of dicts) to a list of row dicts."""
    return table.to_dict("records") if hasattr(table, "to_dict") else table


def test_geometry_table_reads_shape_from_fields():
    fields = [_make_field(10, 10, seed=0), _make_field(12, 8, seed=1), _make_field(10, 10, seed=2)]
    fields[0].geometry = {"name": "A", "uuid": "UUID-A", "center_xy": (0.0, 0.0), "size_xy": (1.0, 1.0)}
    fields[1].geometry = {"name": "B", "uuid": "UUID-B", "center_xy": (2.0, 1.0), "size_xy": (1.0, 1.0)}
    # fields[2] intentionally has no geometry.

    rows = _rows(geometry_table(fields))

    # Row index is the field's position; shape/frames/fs read from the field itself.
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert (rows[1]["height"], rows[1]["width"]) == (12, 8)
    assert rows[0]["num_frames"] == 5
    assert rows[0]["num_planes"] == 1
    assert rows[0]["sampling_frequency"] == 10.0
    assert rows[0]["name"] == "A"
    # A field without geometry gets null placement columns rather than an error.
    assert rows[2]["name"] is None and _is_null(rows[2]["center_x"])


def test_geometry_table_expected_columns():
    rows = _rows(geometry_table([_make_field(10, 10)]))
    expected_columns = {
        "index",
        "name",
        "uuid",
        "height",
        "width",
        "num_planes",
        "num_frames",
        "sampling_frequency",
        "center_x",
        "center_y",
        "size_x",
        "size_y",
        "center_x_um",
        "center_y_um",
        "size_x_um",
        "size_y_um",
    }
    assert expected_columns <= set(rows[0])


def test_geometry_table_micron_columns():
    field = _make_field(10, 10)
    field.geometry = {"center_xy": (2.0, 3.0), "size_xy": (1.0, 1.0), "microns_per_scanner_unit": 150.0}

    row = _rows(geometry_table([field]))[0]

    assert (row["center_x_um"], row["center_y_um"]) == (2.0 * 150.0, 3.0 * 150.0)
    assert (row["size_x_um"], row["size_y_um"]) == (150.0, 150.0)


def test_geometry_table_micron_columns_none_without_scale():
    field = _make_field(10, 10)
    field.geometry = {"center_xy": (2.0, 3.0), "size_xy": (1.0, 1.0)}

    row = _rows(geometry_table([field]))[0]

    # Scanner-unit geometry present but no scale -> no micron view (not fabricated).
    assert _is_null(row["center_x_um"]) and _is_null(row["size_x_um"])
    assert (row["center_x"], row["center_y"]) == (2.0, 3.0)
