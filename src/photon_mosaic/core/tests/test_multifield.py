"""Tests for the MultiFieldImaging container and FieldGeometry."""

import numpy as np
import pytest

from photon_mosaic.core import (
    BaseImaging,
    FieldGeometry,
    MultiFieldImaging,
    generate_random_imaging,
)


def _make_field(height: int, width: int, sampling_frequency: float = 10.0, seed: int = 0) -> BaseImaging:
    """Create a small random single-plane imaging field for testing."""
    return generate_random_imaging(
        num_frames=5,
        height=height,
        width=width,
        sampling_frequency=sampling_frequency,
        seed=seed,
    )


@pytest.fixture
def multifield() -> MultiFieldImaging:
    """A 3-field container with heterogeneous pixel shapes."""
    fields = [_make_field(10, 10, seed=0), _make_field(12, 8, seed=1), _make_field(10, 10, seed=2)]
    geometry = [
        FieldGeometry(index=0, name="A", uuid="UUID-A", center_xy=(0.0, 0.0), size_xy=(1.0, 1.0), affine=np.eye(3)),
        FieldGeometry(index=1, name="B", uuid="UUID-B", center_xy=(2.0, 1.0), size_xy=(1.0, 1.0)),
        FieldGeometry(index=2, name="C", uuid="UUID-C"),
    ]
    return MultiFieldImaging(fields, geometry)


def test_len_and_iteration(multifield):
    assert len(multifield) == 3
    assert all(isinstance(field, BaseImaging) for field in multifield)


def test_getitem_int_and_slice(multifield):
    assert multifield[0] is multifield.fields[0]
    assert multifield[:2] == multifield.fields[:2]


def test_iter_fields_pairs_are_aligned(multifield):
    for geom, field in multifield.iter_fields():
        assert isinstance(geom, FieldGeometry)
        assert isinstance(field, BaseImaging)
    assert [geom.index for geom, _ in multifield.iter_fields()] == [0, 1, 2]


def test_get_field_selects_by_index_name_and_uuid(multifield):
    # Selector-resolution semantics (case-insensitivity, ambiguous/unknown/empty selectors) are
    # covered upstream by roiextractors' _resolve_roi_index tests; here we only confirm the container
    # hands back the field at the resolved position.
    assert multifield.get_field(index=1) is multifield.fields[1]
    assert multifield.get_field(name="B") is multifield.fields[1]
    assert multifield.get_field(uuid="UUID-B") is multifield.fields[1]


def test_geometry_table_reads_shape_from_fields(multifield):
    table = multifield.geometry_table()
    rows = table.to_dict("records") if hasattr(table, "to_dict") else table

    # The middle field has a distinct pixel shape, read from the field itself.
    assert (rows[1]["height"], rows[1]["width"]) == (12, 8)
    assert rows[0]["num_frames"] == 5
    assert rows[0]["num_planes"] == 1
    assert rows[0]["sampling_frequency"] == 10.0
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
    }
    assert expected_columns <= set(rows[0])


def test_geometry_table_includes_micron_columns():
    field = _make_field(10, 10)
    geom = FieldGeometry(index=0, center_xy=(2.0, 3.0), size_xy=(1.0, 1.0), microns_per_scanner_unit=150.0)
    table = MultiFieldImaging([field], [geom]).geometry_table()
    row = (table.to_dict("records") if hasattr(table, "to_dict") else table)[0]

    assert {"center_x_um", "center_y_um", "size_x_um", "size_y_um"} <= set(row)
    assert (row["center_x_um"], row["center_y_um"]) == (2.0 * 150.0, 3.0 * 150.0)
    assert (row["size_x_um"], row["size_y_um"]) == (150.0, 150.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        MultiFieldImaging([_make_field(10, 10)], [FieldGeometry(index=0), FieldGeometry(index=1)])


def test_non_baseimaging_field_raises():
    with pytest.raises(TypeError, match="BaseImaging"):
        MultiFieldImaging(["not imaging"], [FieldGeometry(index=0)])


def test_heterogeneous_sampling_frequency_warns():
    fields = [_make_field(10, 10, sampling_frequency=10.0), _make_field(10, 10, sampling_frequency=20.0)]
    geometry = [FieldGeometry(index=0), FieldGeometry(index=1)]
    with pytest.warns(UserWarning, match="heterogeneous sampling"):
        MultiFieldImaging(fields, geometry)


def test_field_geometry_defaults():
    geom = FieldGeometry(index=3)
    assert geom.name is None and geom.uuid is None
    assert geom.center_xy is None and geom.size_xy is None and geom.affine is None
    # Without a scale, the micron views are unavailable rather than fabricated.
    assert geom.microns_per_scanner_unit is None
    assert geom.center_xy_um is None and geom.size_xy_um is None


def test_field_geometry_microns_conversion():
    # microns_per_scanner_unit mirrors ScanImage's objectiveResolution; the *_um views scale by it.
    geom = FieldGeometry(
        index=0,
        center_xy=(-6.9, -5.5),
        size_xy=(4.6665, 4.6665),
        microns_per_scanner_unit=150.0,
    )
    assert geom.center_xy_um == (-6.9 * 150.0, -5.5 * 150.0)
    assert geom.size_xy_um == (4.6665 * 150.0, 4.6665 * 150.0)


def test_field_geometry_microns_none_when_scale_missing():
    # Scanner-unit geometry present but no scale -> no micron view.
    geom = FieldGeometry(index=0, center_xy=(1.0, 2.0), size_xy=(3.0, 4.0))
    assert geom.center_xy_um is None and geom.size_xy_um is None


def test_field_geometry_microns_none_when_geometry_missing():
    # Scale present but no scanner-unit geometry -> still no micron view.
    geom = FieldGeometry(index=0, microns_per_scanner_unit=150.0)
    assert geom.center_xy_um is None and geom.size_xy_um is None


def _make_field_rois(num_rois: int, sampling_frequency: float = 10.0, seed: int = 0):
    """Generate ROIs that fit a 20x20 field at the given sampling frequency."""
    from photon_mosaic.core import generate_rois

    return generate_rois(
        num_rois=num_rois,
        height=20,
        width=20,
        radius_range=(2, 5),
        sampling_frequency=sampling_frequency,
        seed=seed,
    )


def _make_multifield_roi_analyzer():
    """Build a 2-field MultiFieldRoiAnalyzer with named/uuid'd fields for container tests."""
    from photon_mosaic.core import create_multifield_roi_analyzer

    fields = [_make_field(20, 20, seed=0), _make_field(20, 20, seed=1)]
    geometry = [
        FieldGeometry(index=0, name="A", uuid="UUID-A"),
        FieldGeometry(index=1, name="B", uuid="UUID-B"),
    ]
    multifield = MultiFieldImaging(fields, geometry)
    rois = [_make_field_rois(num_rois=3, seed=0), _make_field_rois(num_rois=2, seed=1)]
    return create_multifield_roi_analyzer(rois, multifield)


def test_create_multifield_roi_analyzer_returns_container():
    from photon_mosaic.core import MultiFieldRoiAnalyzer, RoiAnalyzer

    container = _make_multifield_roi_analyzer()

    assert isinstance(container, MultiFieldRoiAnalyzer)
    assert len(container) == 2
    assert all(isinstance(analyzer, RoiAnalyzer) for analyzer in container)
    # Each analyzer is paired with the matching field's ROIs, in order.
    assert [analyzer.get_num_rois() for analyzer in container] == [3, 2]


def test_container_getitem_and_iter_fields_are_aligned():
    container = _make_multifield_roi_analyzer()

    assert container[0] is container.analyzers[0]
    assert container[:1] == container.analyzers[:1]
    assert [geom.index for geom, _ in container.iter_fields()] == [0, 1]
    for geom, analyzer in container.iter_fields():
        assert isinstance(geom, FieldGeometry)
        assert analyzer.get_num_rois() in (3, 2)


def test_get_analyzer_selects_by_index_name_and_uuid():
    # As with get_field, selector-resolution semantics live in roiextractors; this only checks the
    # container returns the analyzer at the resolved position.
    container = _make_multifield_roi_analyzer()

    assert container.get_analyzer(index=1) is container[1]
    assert container.get_analyzer(name="B") is container[1]
    assert container.get_analyzer(uuid="UUID-B") is container[1]


def test_compute_fans_out_to_every_field():
    container = _make_multifield_roi_analyzer()

    container.compute("fluorescence")

    # Every field computed the extension independently, with its own ROI count on the ROI axis.
    for analyzer, expected_num_rois in zip(container, (3, 2)):
        fluorescence = analyzer.get_extension("fluorescence")
        assert fluorescence is not None
        assert fluorescence.get_data().shape == (5, expected_num_rois)


def test_multifield_roi_analyzer_length_mismatch():
    from photon_mosaic.core import MultiFieldRoiAnalyzer, RoiAnalyzer

    container = _make_multifield_roi_analyzer()
    analyzers: list[RoiAnalyzer] = list(container)

    with pytest.raises(ValueError, match="length mismatch"):
        MultiFieldRoiAnalyzer(analyzers, [FieldGeometry(index=0)])


def test_create_multifield_roi_analyzer_length_mismatch():
    from photon_mosaic.core import create_multifield_roi_analyzer

    multifield = MultiFieldImaging([_make_field(20, 20)], [FieldGeometry(index=0)])
    rois = [_make_field_rois(num_rois=2), _make_field_rois(num_rois=2)]

    with pytest.raises(ValueError, match="must match"):
        create_multifield_roi_analyzer(rois, multifield)
