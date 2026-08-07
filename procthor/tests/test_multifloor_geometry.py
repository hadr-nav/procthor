"""Pure tests for schema-2 geometry helpers (no AI2-THOR required)."""

import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "generation" / "multifloor.py"
SPEC = importlib.util.spec_from_file_location(
    "procthor_multifloor_geometry", MODULE_PATH
)
multifloor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multifloor)


def polygon_area(polygon):
    points = [
        (point["x"], point["z"]) if isinstance(point, dict) else point
        for point in polygon
    ]
    return abs(
        sum(
            x0 * z1 - x1 * z0
            for (x0, z0), (x1, z1) in zip(points, points[1:] + points[:1])
        )
        / 2
    )


def polygon_bounds(polygon):
    xs = [point[0] for point in polygon]
    zs = [point[1] for point in polygon]
    return min(xs), min(zs), max(xs), max(zs)


def intersection_area(first, second):
    a = polygon_bounds(first)
    b = polygon_bounds(second)
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


class MultiFloorGeometryTests(unittest.TestCase):
    def test_fixed_vertical_contract(self):
        self.assertEqual(multifloor.MULTI_FLOOR_SCHEMA, "2.0.0")
        self.assertEqual(multifloor.FLOOR_PITCH, 3.0)
        self.assertEqual(multifloor.SLAB_THICKNESS, 0.2)
        self.assertEqual(multifloor.CLEAR_HEIGHT, 2.8)
        self.assertEqual(multifloor.floor_base_y(0), 0.0)
        self.assertEqual(multifloor.floor_base_y(2), 6.0)
        self.assertEqual(multifloor.floor_ceiling_y(1), 5.8)
        with self.assertRaises(multifloor.MultiFloorGeometryError):
            multifloor.StairGeometryContract(clear_height=2.7)

    def test_stair_core_is_centered_and_oriented_along_long_axis(self):
        core = multifloor.place_stair_core((0, 0, 10, 8), margin=0.25)
        self.assertEqual(core.long_axis, "x")
        self.assertEqual(core.yaw, 90.0)
        self.assertTrue(math.isclose(core.bounds.width, 6.5))
        self.assertTrue(math.isclose(core.bounds.depth, 1.2))
        self.assertEqual(core.center, (5.0, 4.0))

        shared = multifloor.place_shared_stair_core(
            [(0, 0, 8, 10), (1, 1, 9, 9)], preferred_axis="z"
        )
        self.assertEqual(shared.long_axis, "z")
        self.assertEqual(shared.center, (4.5, 5.0))

        with self.assertRaises(multifloor.StairCoreDoesNotFit):
            multifloor.place_stair_core((0, 0, 5, 5))

    def test_opening_decomposition_is_exact_and_non_overlapping(self):
        pieces = multifloor.decompose_rectangle_around_opening(
            outer=(0, 0, 10, 8), opening=(3, 2, 7, 6)
        )
        self.assertEqual(len(pieces), 4)
        self.assertTrue(math.isclose(sum(map(polygon_area, pieces)), 80 - 16))
        for index, first in enumerate(pieces):
            for second in pieces[index + 1 :]:
                self.assertEqual(intersection_area(first, second), 0)

        edge_pieces = multifloor.decompose_rectangle_around_opening(
            outer=(0, 0, 10, 8), opening=(0, 2, 2, 4)
        )
        self.assertEqual(len(edge_pieces), 3)
        self.assertTrue(math.isclose(sum(map(polygon_area, edge_pieces)), 80 - 4))

        with self.assertRaises(multifloor.InvalidOpening):
            multifloor.decompose_rectangle_around_opening(
                outer=(0, 0, 4, 4), opening=(3, 3, 5, 5)
            )

    def test_global_room_numbers_and_ids_are_deterministic(self):
        mappings = multifloor.make_global_room_id_maps([[5, 2], [2, 4], [2]])
        self.assertEqual(mappings, [{2: 2, 5: 3}, {2: 4, 4: 5}, {2: 6}])
        ids = {
            multifloor.floor_qualified_id("wall", floor, "exterior", 0)
            for floor in range(3)
        }
        self.assertEqual(len(ids), 3)
        self.assertEqual(
            multifloor.connector_id(1),
            "vertical-connector|floor-1-to-2|connector-1",
        )

    def test_room_surfaces_keep_semantic_footprint_and_use_actual_y(self):
        metadata = multifloor.make_room_surface_metadata(
            floor_index=1,
            room_number=7,
            outer=(0, 0, 10, 8),
            floor_opening=(3, 2, 7, 6),
            ceiling_opening=(3, 2, 7, 6),
            floor_material={"name": "Oak"},
        )
        self.assertEqual(metadata["floorId"], "floor|1")
        self.assertEqual(len(metadata["floorPolygon"]), 4)
        self.assertEqual({point["y"] for point in metadata["floorPolygon"]}, {3.0})
        self.assertEqual(len(metadata["floorPolygons"]), 4)
        self.assertEqual(len(metadata["ceilings"]), 4)
        self.assertEqual(
            {
                point["y"]
                for piece in metadata["floorPolygons"]
                for point in piece["polygon"]
            },
            {3.0},
        )
        self.assertEqual(
            {
                point["y"]
                for piece in metadata["ceilings"]
                for point in piece["polygon"]
            },
            {5.8},
        )
        self.assertTrue(
            math.isclose(
                sum(
                    polygon_area(piece["polygon"])
                    for piece in metadata["floorPolygons"]
                ),
                64,
            )
        )

        floor = multifloor.make_floor_record(
            floor_index=1, room_id_map={2: 7}, room_surfaces={7: metadata}
        )
        self.assertEqual(floor["baseY"], 3.0)
        self.assertEqual(floor["ceilingY"], 5.8)
        self.assertEqual(len(floor["floorSurfaces"]), 4)
        self.assertEqual(len(floor["ceilingSurfaces"]), 4)

    def test_three_floor_connectors_use_parallel_stacked_flights(self):
        core = multifloor.place_stair_core((0, 0, 8, 10), preferred_axis="z")
        connectors = multifloor.build_vertical_connectors(core, [2, 4, 6])
        self.assertEqual(len(connectors), 2)
        self.assertEqual([item["rotation"]["y"] for item in connectors], [0.0, 0.0])
        self.assertEqual([item["position"]["y"] for item in connectors], [0.0, 3.0])
        for connector_index, connector in enumerate(connectors):
            self.assertEqual(
                connector["assetContract"]["landingEgressDepth"],
                multifloor.STAIR_LANDING_EGRESS_DEPTH,
            )
            openings = {
                (opening["floorId"], opening["surfaceType"]): opening
                for opening in connector["openingPolygons"]
            }
            expected_openings = {
                (connector["lowerFloorId"], "ceiling"): multifloor.floor_ceiling_y(
                    connector_index
                ),
                (connector["upperFloorId"], "floor"): multifloor.floor_base_y(
                    connector_index + 1
                ),
            }
            self.assertEqual(len(connector["openingPolygons"]), len(expected_openings))
            self.assertEqual(set(openings), set(expected_openings))
            for opening_key, expected_y in expected_openings.items():
                opening = openings[opening_key]
                serialized_bounds = multifloor.as_rectangle(
                    [(point["x"], point["z"]) for point in opening["polygon"]]
                )
                self.assertEqual(serialized_bounds, core.bounds)
                self.assertEqual(
                    {point["y"] for point in opening["polygon"]},
                    {expected_y},
                )
        all_ids = {
            surface["id"]
            for connector in connectors
            for key in ("landingPolygons", "openingPolygons")
            for surface in connector[key]
        }
        self.assertEqual(len(all_ids), 8)

    def test_floor_openings_use_the_full_reserved_stair_core(self):
        core = multifloor.place_stair_core((0, 0, 8, 10), preferred_axis="z")
        ground = multifloor.stair_floor_opening(core, 0)
        middle = multifloor.stair_floor_opening(core, 1)
        top = multifloor.stair_floor_opening(core, 2)

        self.assertEqual(ground, core.bounds)
        self.assertEqual(middle, core.bounds)
        self.assertEqual(top, core.bounds)

    def test_schema2_structure_composes_floor_maps_and_reservations(self):
        structure = multifloor.build_schema2_structure(
            floor_room_ids=[[2, 5], [2, 3], [2]],
            stair_host_room_ids=[5, 2, 2],
            shared_boundary=(0, 0, 8, 10),
        )
        self.assertEqual(structure["schema"], "2.0.0")
        self.assertEqual(len(structure["floors"]), 3)
        self.assertEqual(
            [floor["roomIdMap"] for floor in structure["floors"]],
            [{"2": 2, "5": 3}, {"2": 4, "3": 5}, {"2": 6}],
        )
        self.assertEqual(len(structure["verticalConnectors"]), 2)
        self.assertEqual(len(structure["stairCoreReservations"]), 3)
        structural_ids = [
            item["id"]
            for item in structure["stairCoreReservations"]
            + structure["verticalConnectors"]
        ]
        self.assertEqual(len(structural_ids), len(set(structural_ids)))


if __name__ == "__main__":
    unittest.main()
