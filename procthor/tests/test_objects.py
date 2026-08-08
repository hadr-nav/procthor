"""Tests for floor-object placement geometry."""

from shapely.geometry import Polygon

from procthor.generation.objects import OrthogonalPolygon


def test_get_all_rectangles_for_l_shaped_polygon():
    polygon = OrthogonalPolygon(
        Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    )

    assert polygon.get_all_rectangles() == {
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 2.0, 1.0),
        (0.0, 1.0, 1.0, 2.0),
        (0.0, 0.0, 2.0, 1.0),
        (0.0, 0.0, 1.0, 2.0),
    }
