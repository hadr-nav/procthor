"""Focused regression tests for whole-navmesh house validation."""

import types
import unittest
from unittest import mock

from shapely.geometry import Polygon

from procthor.generation.house import House


class _Event:
    def __init__(self, action_return=None):
        self.metadata = {"actionReturn": action_return}


class _Controller:
    def __init__(self, connectivity):
        self.connectivity = connectivity
        self.actions = []

    def reset(self, **_kwargs):
        return _Event()

    def step(self, **kwargs):
        action = kwargs["action"]
        self.actions.append(action)
        if action == "GetReachablePositions":
            return _Event(
                [
                    {"x": 0.5, "y": floor_y + 0.9, "z": 0.5}
                    for floor_y in (0.0, 3.0)
                    for _ in range(5)
                ]
            )
        if action == "GetNavMeshConnectivity":
            return _Event(self.connectivity)
        return _Event()


def _schema2_house():
    room_polygon = Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))
    rooms = {
        room_id: types.SimpleNamespace(
            room_polygon=types.SimpleNamespace(polygon=room_polygon)
        )
        for room_id in (2, 3)
    }
    return House(
        data={
            "metadata": {"agent": {}},
            "floors": [{}, {}],
            "verticalConnectors": [],
        },
        rooms=rooms,
        interior_boundary=None,
        room_spec=types.SimpleNamespace(room_spec_id="unused"),
        add_metadata=False,
        room_floor_map={2: 0.0, 3: 3.0},
    )


def _schema1_house():
    house = _schema2_house()
    house.room_floor_map = None
    house.rooms = {2: house.rooms[2]}
    return house


class HouseValidationTests(unittest.TestCase):
    def test_generated_house_bakes_the_actual_movement_footprint(self):
        house = _schema2_house()

        with mock.patch.object(House, "choose_agent_pose", return_value={}):
            house._add_metadata()

        self.assertEqual(
            house.data["metadata"]["navMeshes"],
            [{"id": 0, "agentRadius": 0.28}],
        )

    def test_schema2_rejects_any_disconnected_navmesh_component(self):
        controller = _Controller(
            {
                "agentTypeId": 0,
                "triangleCount": 17,
                "componentCount": 2,
                "componentSizes": [16, 1],
                "connected": False,
            }
        )

        warnings = _schema2_house().validate(controller)

        self.assertIn("NavMeshNotConnected", warnings)
        self.assertIn("2 components", warnings["NavMeshNotConnected"])
        self.assertIn("[16, 1]", warnings["NavMeshNotConnected"])
        self.assertIn("GetNavMeshConnectivity", controller.actions)

    def test_schema2_accepts_one_complete_navmesh_component(self):
        controller = _Controller(
            {
                "agentTypeId": 0,
                "triangleCount": 17,
                "componentCount": 1,
                "componentSizes": [17],
                "connected": True,
            }
        )

        warnings = _schema2_house().validate(controller)

        self.assertEqual(warnings, {})

    def test_schema1_also_rejects_any_disconnected_navmesh_component(self):
        controller = _Controller(
            {
                "agentTypeId": 0,
                "triangleCount": 9,
                "componentCount": 2,
                "componentSizes": [8, 1],
                "connected": False,
            }
        )

        warnings = _schema1_house().validate(controller)

        self.assertIn("NavMeshNotConnected", warnings)
        self.assertIn("GetNavMeshConnectivity", controller.actions)


if __name__ == "__main__":
    unittest.main()
