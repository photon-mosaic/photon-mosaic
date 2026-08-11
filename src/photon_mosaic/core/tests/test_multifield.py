"""Tests for the multifield field-selection helpers (get_field, fields_containing_point)."""

import pytest

from photon_mosaic.core import fields_containing_point, get_field
from photon_mosaic.core.generators import generate_random_imaging


def _make_field(name=None, uuid=None, seed: int = 0):
    """Create a small imaging field, optionally tagged with name/uuid in its geometry."""
    imaging = generate_random_imaging(num_frames=5, height=10, width=10, sampling_frequency=10.0, seed=seed)
    if name is not None or uuid is not None:
        imaging.geometry = {"name": name, "uuid": uuid}
    return imaging


@pytest.fixture
def fields():
    """Three fields with distinct names/uuids."""
    return [
        _make_field(name="A", uuid="UUID-A", seed=0),
        _make_field(name="B", uuid="UUID-B", seed=1),
        _make_field(name="C", uuid="UUID-C", seed=2),
    ]


def test_get_field_by_index(fields):
    assert get_field(fields, index=1) is fields[1]


def test_get_field_by_name(fields):
    assert get_field(fields, name="B") is fields[1]


def test_get_field_by_uuid_is_case_insensitive(fields):
    assert get_field(fields, uuid="uuid-b") is fields[1]


def test_get_field_requires_exactly_one_selector(fields):
    with pytest.raises(ValueError, match="exactly one"):
        get_field(fields)
    with pytest.raises(ValueError, match="exactly one"):
        get_field(fields, index=0, name="A")


def test_get_field_unknown_name_raises(fields):
    with pytest.raises(KeyError, match="no field matching"):
        get_field(fields, name="Z")


def test_get_field_ambiguous_name_raises():
    fields = [_make_field(name="dup", seed=0), _make_field(name="dup", seed=1)]
    with pytest.raises(ValueError, match="ambiguous"):
        get_field(fields, name="dup")


def test_get_field_handles_fields_without_geometry():
    # A field with no geometry simply never matches a name/uuid lookup.
    fields = [_make_field(seed=0), _make_field(name="B", seed=1)]
    assert get_field(fields, name="B") is fields[1]
    with pytest.raises(KeyError):
        get_field(fields, name="A")


def test_fields_containing_point_is_not_implemented(fields):
    with pytest.raises(NotImplementedError, match="spatial field selection"):
        fields_containing_point(fields, (0.0, 0.0))
