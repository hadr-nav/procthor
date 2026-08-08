"""Tests for constrained, connected multi-floor stair-host partitions."""

import types
import unittest

import numpy as np
from shapely.geometry import Polygon

from procthor.generation.multifloor_generation import (
    _grid_region_is_connected,
    _reserve_shared_stair_host_region,
    _room_polygons,
)
from procthor.generation.multifloor_specs import FloorSpec, HouseSpec
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import LeafRoom


def _room_spec(identifier):
    return RoomSpec(
        room_spec_id=identifier,
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2, ratio=2, room_type="LivingRoom"),
            LeafRoom(room_id=3, ratio=2, room_type="Bedroom"),
        ],
    )


class MultiFloorReservationTests(unittest.TestCase):
    def test_reservation_constructs_one_shared_host_without_splitting_rooms(self):
        local_specs = [_room_spec("floor-{}".format(index)) for index in range(3)]
        house_spec = HouseSpec(
            house_spec_id="three-floor-reservation",
            floors=[
                FloorSpec(room_spec=room_spec, stair_host_room_id=2)
                for room_spec in local_specs
            ],
        )
        room_id_maps = [{2: 2, 3: 3}, {2: 4, 3: 5}, {2: 6, 3: 7}]
        remapped_specs = [
            RoomSpec(
                room_spec_id=room_spec.room_spec_id,
                sampling_weight=1,
                spec=[
                    LeafRoom(
                        room_id=room_id_map[2],
                        ratio=2,
                        room_type="LivingRoom",
                    ),
                    LeafRoom(
                        room_id=room_id_map[3],
                        ratio=2,
                        room_type="Bedroom",
                    ),
                ],
            )
            for room_spec, room_id_map in zip(local_specs, room_id_maps)
        ]
        boundary = np.zeros((6, 6), dtype=int)
        structures = []
        for room_id_map in room_id_maps:
            grid = np.full((6, 6), room_id_map[3], dtype=int)
            grid[0, :] = room_id_map[2]
            structures.append(
                types.SimpleNamespace(
                    interior_boundary=boundary.copy(),
                    floorplan=np.pad(grid, 1, constant_values=1),
                    rowcol_walls={},
                    boundary_groups={},
                    xz_poly_map={},
                    ceiling_height=2.8,
                )
            )

        rebuilt, core, host_ids = _reserve_shared_stair_host_region(
            house_spec=house_spec,
            remapped_room_specs=remapped_specs,
            room_id_maps=room_id_maps,
            structures=structures,
            interior_boundary_scale=1.6,
        )

        self.assertEqual(host_ids, [2, 4, 6])
        self.assertEqual(core.long_axis, "z")
        for structure, room_spec, host_id in zip(rebuilt, remapped_specs, host_ids):
            grid = structure.floorplan[1:-1, 1:-1]
            self.assertTrue(
                all(
                    _grid_region_is_connected(grid, room_number)
                    for room_number in room_spec.room_type_map
                )
            )
            host_polygon = _room_polygons([structure])[0][host_id]
            self.assertTrue(
                host_polygon.covers(
                    Polygon(
                        [
                            (core.bounds.min_x - 0.8, core.bounds.min_z - 0.8),
                            (core.bounds.max_x + 0.8, core.bounds.min_z - 0.8),
                            (core.bounds.max_x + 0.8, core.bounds.max_z + 0.8),
                            (core.bounds.min_x - 0.8, core.bounds.max_z + 0.8),
                        ]
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
