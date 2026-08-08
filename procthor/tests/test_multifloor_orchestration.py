"""Focused dependency-isolated tests for schema-2 orchestration helpers."""

import enum
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATION_ROOT = REPO_ROOT / "procthor" / "generation"
UTILS_ROOT = REPO_ROOT / "procthor" / "utils"
_MISSING = object()


def _load_file_module(qualified_name, path):
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _package_shell(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    return package


def _load_orchestration_module():
    """Load the orchestrator without importing optional runtime dependencies."""

    module_names = [
        "numpy",
        "ai2thor",
        "ai2thor.controller",
        "shapely",
        "shapely.geometry",
        "shapely.geometry.polygon",
        "procthor",
        "procthor.constants",
        "procthor.utils",
        "procthor.utils.types",
        "procthor.generation",
        "procthor.generation.generation",
        "procthor.generation.house",
        "procthor.generation.layer",
        "procthor.generation.materials",
        "procthor.generation.multifloor",
        "procthor.generation.multifloor_specs",
        "procthor.generation.room_specs",
        "procthor.generation.multifloor_generation",
    ]
    previous = {name: sys.modules.get(name, _MISSING) for name in module_names}
    try:
        numpy = types.ModuleType("numpy")
        numpy.asarray = lambda value: value
        numpy.array_equal = lambda left, right: left == right
        numpy.ndarray = object
        sys.modules["numpy"] = numpy

        ai2thor = types.ModuleType("ai2thor")
        ai2thor_controller = types.ModuleType("ai2thor.controller")

        class Controller:
            pass

        ai2thor_controller.Controller = Controller
        ai2thor.controller = ai2thor_controller
        sys.modules["ai2thor"] = ai2thor
        sys.modules["ai2thor.controller"] = ai2thor_controller

        shapely = types.ModuleType("shapely")
        shapely_geometry = types.ModuleType("shapely.geometry")

        class GeometryCollection:
            pass

        class MultiPolygon:
            pass

        class Polygon:
            pass

        shapely_geometry.GeometryCollection = GeometryCollection
        shapely_geometry.MultiPolygon = MultiPolygon
        shapely_geometry.Polygon = Polygon
        shapely_geometry.box = lambda *args: None
        shapely_polygon = types.ModuleType("shapely.geometry.polygon")
        shapely_polygon.orient = lambda polygon, sign=1.0: polygon
        shapely.geometry = shapely_geometry
        sys.modules["shapely"] = shapely
        sys.modules["shapely.geometry"] = shapely_geometry
        sys.modules["shapely.geometry.polygon"] = shapely_polygon

        sys.modules["procthor"] = _package_shell("procthor", REPO_ROOT / "procthor")
        sys.modules["procthor.utils"] = _package_shell("procthor.utils", UTILS_ROOT)
        _load_file_module("procthor.utils.types", UTILS_ROOT / "types.py")
        sys.modules["procthor.generation"] = _package_shell(
            "procthor.generation", GENERATION_ROOT
        )

        generation = types.ModuleType("procthor.generation.generation")
        generation.consolidate_walls = lambda walls: walls
        generation.find_walls = lambda floorplan: {}
        generation.get_floor_polygons = lambda xz_poly_map: xz_poly_map
        generation.get_xz_poly_map = lambda boundary_groups, room_ids: {}
        generation.scale_boundary_groups = lambda boundary_groups, scale: (
            boundary_groups
        )
        sys.modules["procthor.generation.generation"] = generation

        house = types.ModuleType("procthor.generation.house")

        class NextSamplingStage(enum.Enum):
            DOORS = 1
            COMPLETE = 9

        class House:
            pass

        class PartialHouse:
            pass

        house.NextSamplingStage = NextSamplingStage
        house.House = House
        house.PartialHouse = PartialHouse
        sys.modules["procthor.generation.house"] = house

        layer = types.ModuleType("procthor.generation.layer")
        layer.assign_layer_to_rooms = lambda partial_house: None
        sys.modules["procthor.generation.layer"] = layer
        materials = types.ModuleType("procthor.generation.materials")
        materials.randomize_wall_and_floor_materials = lambda partial_house, pt_db: None
        sys.modules["procthor.generation.materials"] = materials

        _load_file_module(
            "procthor.generation.multifloor",
            GENERATION_ROOT / "multifloor.py",
        )
        multifloor_specs = types.ModuleType("procthor.generation.multifloor_specs")
        multifloor_specs.HouseSpec = type("HouseSpec", (), {})
        sys.modules["procthor.generation.multifloor_specs"] = multifloor_specs
        room_specs = types.ModuleType("procthor.generation.room_specs")
        room_specs.RoomSpec = type("RoomSpec", (), {})
        sys.modules["procthor.generation.room_specs"] = room_specs

        return _load_file_module(
            "procthor.generation.multifloor_generation",
            GENERATION_ROOT / "multifloor_generation.py",
        )
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


orchestration = _load_orchestration_module()


class _Event:
    def __init__(self, action_return, success=True):
        self.metadata = {
            "lastActionSuccess": success,
            "actionReturn": action_return,
        }


class _Controller:
    def __init__(self, action_return):
        self.action_return = action_return
        self.actions = []

    def step(self, **kwargs):
        self.actions.append(kwargs)
        return _Event(self.action_return)


class MultiFloorOrchestrationTests(unittest.TestCase):
    def test_surface_objects_filter_engine_degenerate_slivers(self):
        class Rectangle:
            def __init__(self, bounds):
                self.bounds = bounds
                min_x, min_z, max_x, max_z = bounds
                self.exterior = types.SimpleNamespace(
                    coords=[
                        (min_x, min_z),
                        (max_x, min_z),
                        (max_x, max_z),
                        (min_x, max_z),
                        (min_x, min_z),
                    ]
                )

        sliver = Rectangle((0.0, 0.0, 2.0, 0.0001))
        engine_epsilon = Rectangle((0.0, 1.0, 2.0, 1.001))
        valid = Rectangle((0.0, 2.0, 2.0, 2.002))

        with mock.patch.object(
            orchestration,
            "_polygon_parts",
            return_value=[sliver, engine_epsilon, valid],
        ) as polygon_parts:
            surfaces = orchestration._surface_objects(
                geometry=object(),
                floor_index=0,
                room_number=3,
                surface_type="floor",
                material=None,
                opening=None,
            )
        polygon_parts.assert_called_once_with(
            mock.ANY,
            preferred_merge_axis="x",
        )

        self.assertEqual(len(surfaces), 1)
        self.assertEqual(
            surfaces[0]["id"],
            "floor-surface|floor-0|room-3|piece-0",
        )
        self.assertEqual(
            {point["z"] for point in surfaces[0]["polygon"]},
            {2.0, 2.002},
        )

    def test_surface_cells_merge_across_complete_edges(self):
        class Rectangle:
            def __init__(self, bounds):
                self.bounds = bounds

        cells = [
            Rectangle((0.0, 0.0, 1.0, 1.0)),
            Rectangle((0.0, 1.0, 1.0, 2.0)),
            Rectangle((1.0, 0.0, 2.0, 1.0)),
            Rectangle((1.0, 1.0, 2.0, 2.0)),
        ]
        with mock.patch.object(
            orchestration,
            "box",
            side_effect=lambda *bounds: Rectangle(bounds),
        ):
            columns = orchestration._merge_surface_rectangles(cells, "z")
            merged = orchestration._merge_surface_rectangles(columns, "x")

        self.assertEqual(
            sorted(rectangle.bounds for rectangle in columns),
            [(0.0, 0.0, 1.0, 2.0), (1.0, 0.0, 2.0, 2.0)],
        )
        self.assertEqual(
            [rectangle.bounds for rectangle in merged],
            [(0.0, 0.0, 2.0, 2.0)],
        )
        with self.assertRaises(ValueError):
            orchestration._merge_surface_rectangles(cells, "y")

    def test_schema2_preflight_success_and_failure(self):
        controller = _Controller({"supportedHouseSchemas": ["1.0.0", "2.0.0"]})
        self.assertEqual(
            orchestration.ensure_schema2_controller(controller),
            ["1.0.0", "2.0.0"],
        )
        self.assertEqual(
            controller.actions,
            [
                {
                    "action": "GetSupportedHouseSchemas",
                    "renderImage": False,
                }
            ],
        )

        unsupported = _Controller(["1.0.0"])
        with self.assertRaises(orchestration.MultiFloorCompatibilityError):
            orchestration.ensure_schema2_controller(unsupported)
        self.assertEqual(unsupported.actions[0]["action"], "GetSupportedHouseSchemas")

    def test_preflight_precedes_seed_and_all_sampling(self):
        order = []

        class RecordingController(_Controller):
            def step(self, **kwargs):
                order.append(kwargs["action"])
                return super().step(**kwargs)

        class Generator:
            controller = RecordingController(["1.0.0", "2.0.0"])
            seed = None

            def set_seed(self, seed):
                order.append("set_seed")
                self.seed = seed

        class FakeSamplingVars:
            @classmethod
            def sample(cls):
                order.append("sampling_vars")
                return object()

        context = types.SimpleNamespace(partial_houses=[], next_sampling_stage=None)

        def select_house_spec(generator):
            order.append("house_spec")
            return object()

        def sample_structure(**kwargs):
            order.append("structure")
            return context

        def assemble(context):
            order.append("assemble")
            return "house"

        with mock.patch.object(
            orchestration.random, "randint", return_value=123
        ), mock.patch.object(
            orchestration, "SamplingVars", FakeSamplingVars
        ), mock.patch.object(
            orchestration, "_select_house_spec", select_house_spec
        ), mock.patch.object(
            orchestration,
            "sample_complete_multifloor_structure",
            sample_structure,
        ), mock.patch.object(
            orchestration, "_assemble_house", assemble
        ):
            house, contexts = orchestration.sample_multifloor_house(Generator())

        self.assertEqual(house, "house")
        self.assertIn(orchestration.NextSamplingStage.COMPLETE, contexts)
        self.assertEqual(
            order,
            [
                "GetSupportedHouseSchemas",
                "set_seed",
                "house_spec",
                "sampling_vars",
                "structure",
                "assemble",
            ],
        )

    def test_doorway_safe_core_validation_retries_without_reseeding(self):
        seed_calls = []

        class Generator:
            controller = _Controller(["1.0.0", "2.0.0"])
            seed = None

            def set_seed(self, seed):
                seed_calls.append(seed)
                self.seed = seed

        house_spec = types.SimpleNamespace(floors=[object(), object()])

        def candidate_context(label):
            partial_houses = [
                types.SimpleNamespace(
                    label="{}-floor-0".format(label),
                    house_structure=object(),
                    room_spec=object(),
                ),
                types.SimpleNamespace(
                    label="{}-floor-1".format(label),
                    house_structure=object(),
                    room_spec=object(),
                ),
            ]
            return types.SimpleNamespace(
                house_spec=house_spec,
                partial_houses=partial_houses,
                room_id_maps=[{2: 2}, {2: 3}],
                stair_core=None,
                stair_host_room_ids=[],
                next_sampling_stage=None,
            )

        first_context = candidate_context("first")
        second_context = candidate_context("second")
        core = orchestration.StairCore(
            bounds=(0, 0, 1.2, 6.5),
            long_axis="z",
            yaw=0,
        )
        sampling_vars = object()

        with mock.patch.object(
            orchestration.random, "randint", return_value=77
        ) as randint, mock.patch.object(
            orchestration, "_select_house_spec", return_value=house_spec
        ) as select_house_spec, mock.patch.object(
            orchestration,
            "sample_complete_multifloor_structure",
            side_effect=[first_context, second_context],
        ) as sample_structure, mock.patch.object(
            orchestration,
            "_run_floor_door_stage",
            side_effect=[
                {2: ["first-lower-clearance"]},
                {3: ["first-upper-clearance"]},
                {2: ["second-lower-clearance"]},
                {3: ["second-upper-clearance"]},
            ],
        ), mock.patch.object(
            orchestration,
            "_validate_reserved_stair_core",
            side_effect=[
                orchestration.StairCoreDoesNotFit(
                    "first doorway-safe validation failed"
                ),
                (core, [2, 3]),
            ],
        ) as validate_core, mock.patch.object(
            orchestration, "_run_floor_generation_stages"
        ) as run_floor_stages, mock.patch.object(
            orchestration, "_assemble_house", return_value="house"
        ):
            house, contexts = orchestration.sample_multifloor_house(
                Generator(),
                sampling_vars=sampling_vars,
            )

        self.assertEqual(house, "house")
        self.assertIs(
            contexts[orchestration.NextSamplingStage.COMPLETE],
            second_context,
        )
        self.assertEqual(seed_calls, [77])
        randint.assert_called_once_with(0, 2**15)
        self.assertEqual(
            Generator.controller.actions,
            [
                {
                    "action": "GetSupportedHouseSchemas",
                    "renderImage": False,
                }
            ],
        )
        self.assertEqual(select_house_spec.call_count, 2)
        self.assertEqual(sample_structure.call_count, 2)
        self.assertEqual(validate_core.call_count, 2)
        self.assertEqual(run_floor_stages.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["partial_house"] in second_context.partial_houses
                for call in run_floor_stages.call_args_list
            )
        )

    def test_lift_world_vectors_changes_world_y_only(self):
        payload = {
            "position": {"x": 1, "y": 0.5, "z": 2},
            "polygon": [
                {"x": 0, "y": 0.0, "z": 0},
                {"x": 1, "y": 2.8, "z": 0},
            ],
            "rotation": {"x": 0, "y": 90, "z": 0},
            "assetOffset": {"x": 0, "y": 0.25, "z": 0},
            "assetPosition": {"x": 0, "y": 1.25, "z": 0},
            "holePolygon": [
                {"x": 0, "y": 0.1, "z": 0},
                {"x": 1, "y": 2.1, "z": 0},
            ],
            "metadata": {"y": 17},
            "boundingBox": {
                "min": {"x": 0, "y": 0.1, "z": 0},
                "max": {"x": 1, "y": 2.1, "z": 1},
            },
            "children": [{"position": {"x": 3, "y": 1.0, "z": 4}}],
        }

        orchestration._lift_world_vectors(payload, base_y=3.0)

        self.assertEqual(payload["position"]["y"], 3.5)
        self.assertEqual(payload["children"][0]["position"]["y"], 4.0)
        self.assertEqual(payload["polygon"][0]["y"], 3.0)
        self.assertEqual(payload["polygon"][1]["y"], 5.8)
        self.assertEqual(payload["boundingBox"]["min"]["y"], 0.1)
        self.assertEqual(payload["boundingBox"]["max"]["y"], 2.1)
        self.assertEqual(payload["rotation"]["y"], 90)
        self.assertEqual(payload["assetOffset"]["y"], 0.25)
        self.assertEqual(payload["assetPosition"]["y"], 1.25)
        self.assertEqual(payload["holePolygon"][0]["y"], 0.1)
        self.assertEqual(payload["holePolygon"][1]["y"], 2.1)
        self.assertEqual(payload["metadata"]["y"], 17)

    def test_lift_world_vectors_lifts_shared_wall_points_once(self):
        shared_polygon = [
            {"x": 0, "y": 0.0, "z": 0},
            {"x": 1, "y": 0.0, "z": 0},
            {"x": 1, "y": 2.8, "z": 0},
            {"x": 0, "y": 2.8, "z": 0},
        ]
        walls = [
            {"polygon": shared_polygon},
            {"polygon": list(reversed(shared_polygon))},
        ]

        orchestration._lift_world_vectors(walls, base_y=3.0)

        self.assertEqual(
            {point["y"] for wall in walls for point in wall["polygon"]},
            {3.0, 5.8},
        )

    def test_assemble_house_keeps_every_upper_wall_on_its_floor(self):
        def make_partial_house(room_number, wall_polygon, room_spec_id):
            room_spec = types.SimpleNamespace(
                room_type_map={room_number: "LivingRoom"},
                room_spec_id=room_spec_id,
            )
            return types.SimpleNamespace(
                room_types=[
                    {
                        "id": "room|{}".format(room_number),
                        "floorPolygon": [
                            {"x": 0, "y": 0.0, "z": 0},
                            {"x": 1, "y": 0.0, "z": 0},
                            {"x": 1, "y": 0.0, "z": 1},
                            {"x": 0, "y": 0.0, "z": 1},
                        ],
                    }
                ],
                walls=[
                    {
                        "id": "wall|{}|interior".format(room_number),
                        "polygon": wall_polygon,
                    },
                    {
                        "id": "wall|exterior|{}".format(room_number),
                        "polygon": list(reversed(wall_polygon)),
                    },
                ],
                doors=[],
                windows=[],
                objects=[],
                procedural_parameters={"lights": []},
                rooms={},
                room_spec=room_spec,
            )

        def wall_polygon(z):
            return [
                {"x": 0, "y": 0.0, "z": z},
                {"x": 1, "y": 0.0, "z": z},
                {"x": 1, "y": 2.8, "z": z},
                {"x": 0, "y": 2.8, "z": z},
            ]

        ground = make_partial_house(2, wall_polygon(0), "ground")
        upper = make_partial_house(3, wall_polygon(1), "upper")
        context = types.SimpleNamespace(
            partial_houses=[ground, upper],
            stair_host_room_ids=[2, 3],
            room_id_maps=[{2: 2}, {3: 3}],
            stair_core=orchestration.StairCore(
                bounds=(0, 0, 1.2, 6.5),
                long_axis="z",
                yaw=0,
            ),
            interior_boundary=object(),
            house_spec=types.SimpleNamespace(
                house_spec_id="wall-height-regression",
                floors=[
                    types.SimpleNamespace(room_spec=ground.room_spec),
                    types.SimpleNamespace(room_spec=upper.room_spec),
                ],
            ),
        )

        def room_surfaces(partial_house, floor_index, **_kwargs):
            room_number = next(iter(partial_house.room_spec.room_type_map))
            return {
                room_number: {
                    "floorId": "floor|{}".format(floor_index),
                    "floorPolygons": [],
                    "ceilings": [],
                }
            }

        class CapturedHouse:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        with mock.patch.object(
            orchestration,
            "_add_schema2_room_surfaces",
            side_effect=room_surfaces,
        ), mock.patch.object(orchestration, "House", CapturedHouse):
            house = orchestration._assemble_house(context)

        expected_heights = {
            "floor|0": {0.0, 2.8},
            "floor|1": {3.0, 5.8},
        }
        for owner_floor_id, expected_y in expected_heights.items():
            floor_walls = [
                wall
                for wall in house.data["walls"]
                if wall["floorId"] == owner_floor_id
            ]
            self.assertEqual(len(floor_walls), 2)
            for wall in floor_walls:
                self.assertEqual(
                    {point["y"] for point in wall["polygon"]},
                    expected_y,
                )

    def test_landing_egress_requires_all_three_edges_on_both_landings(self):
        candidate = types.SimpleNamespace(bounds=(1.0, 2.0, 2.2, 8.5))
        lower = mock.Mock()
        upper = mock.Mock()

        with mock.patch.object(
            orchestration, "box", side_effect=lambda *bounds: bounds
        ):
            lower.covers.return_value = True
            upper.covers.return_value = True
            self.assertTrue(
                orchestration._has_required_landing_egress(
                    [lower], [upper], candidate, "z"
                )
            )
            self.assertEqual(lower.covers.call_count, 3)
            self.assertEqual(upper.covers.call_count, 3)

            lower.reset_mock()
            upper.reset_mock()
            lower.covers.return_value = True
            upper.covers.side_effect = [True, False, True]
            self.assertFalse(
                orchestration._has_required_landing_egress(
                    [lower], [upper], candidate, "z"
                )
            )

        with mock.patch.object(
            orchestration, "box", side_effect=lambda *bounds: bounds
        ):
            lower_aprons, upper_aprons = orchestration._landing_egress_polygons(
                candidate, "z"
            )
        expected_lower = (
            (1.0, 1.4, 2.2, 2.0),
            (0.4, 2.0, 1.0, 3.0),
            (2.2, 2.0, 2.8, 3.0),
        )
        expected_upper = (
            (1.0, 8.5, 2.2, 9.1),
            (0.4, 7.5, 1.0, 8.5),
            (2.2, 7.5, 2.8, 8.5),
        )
        for actual, expected in zip(
            lower_aprons + upper_aprons,
            expected_lower + expected_upper,
        ):
            for actual_coordinate, expected_coordinate in zip(actual, expected):
                self.assertAlmostEqual(actual_coordinate, expected_coordinate)

    def test_shared_stair_core_subtracts_each_host_door_clearance(self):
        lower_host = mock.Mock(name="lower_host")
        upper_host = mock.Mock(name="upper_host")
        lower_clearances = [object(), object()]
        upper_clearance = object()
        lower_after_first = mock.Mock(name="lower_after_first")
        lower_safe = mock.Mock(name="lower_safe")
        upper_safe = mock.Mock(name="upper_safe")
        shared_safe = mock.Mock(name="shared_safe")
        shared_safe.is_empty = False
        lower_host.difference.return_value = lower_after_first
        lower_after_first.difference.return_value = lower_safe
        upper_host.difference.return_value = upper_safe
        lower_safe.intersection.return_value = shared_safe
        core = orchestration.StairCore(
            bounds=(2, 3, 3.2, 9.5),
            long_axis="z",
            yaw=0,
        )

        with mock.patch.object(
            orchestration,
            "_room_polygons",
            return_value=[{12: lower_host}, {27: upper_host}],
        ), mock.patch.object(
            orchestration,
            "_host_candidates",
            return_value=[[(0, 12)], [(0, 27)]],
        ), mock.patch.object(
            orchestration,
            "_fit_core_in_geometry",
            return_value=core,
        ) as fit_core:
            result = orchestration._locate_shared_stair_core(
                house_spec=object(),
                remapped_room_specs=[object(), object()],
                room_id_maps=[{}, {}],
                structures=[object(), object()],
                door_clearance_polygons=[
                    {12: lower_clearances},
                    {27: [upper_clearance]},
                ],
            )

        self.assertEqual(result, (core, [12, 27]))
        lower_host.difference.assert_called_once_with(lower_clearances[0])
        lower_after_first.difference.assert_called_once_with(lower_clearances[1])
        upper_host.difference.assert_called_once_with(upper_clearance)
        lower_safe.intersection.assert_called_once_with(upper_safe)
        fit_core.assert_called_once_with(
            shared_safe,
            lower_egress_geometries=(lower_safe,),
            upper_egress_geometries=(upper_safe,),
        )

    def test_shared_stair_core_rejects_mismatched_clearance_floors(self):
        with self.assertRaisesRegex(
            orchestration.MultiFloorGeometryError,
            "one mapping per floor",
        ):
            orchestration._locate_shared_stair_core(
                house_spec=object(),
                remapped_room_specs=[object(), object()],
                room_id_maps=[{}, {}],
                structures=[object(), object()],
                door_clearance_polygons=[{}],
            )

    def test_floor_object_reservation_buffers_full_stair_core(self):
        core = orchestration.StairCore(
            bounds=(4.0, 5.0, 5.2, 11.5),
            long_axis="z",
            yaw=0,
        )
        box_calls = []
        buffer_calls = []

        class CorePolygon:
            def __init__(self, bounds):
                self.bounds = bounds

            def buffer(self, distance, join_style):
                buffer_calls.append((distance, join_style))
                min_x, min_z, max_x, max_z = self.bounds
                return types.SimpleNamespace(
                    bounds=(
                        min_x - distance,
                        min_z - distance,
                        max_x + distance,
                        max_z + distance,
                    )
                )

        def recording_box(*bounds):
            box_calls.append(bounds)
            return CorePolygon(bounds)

        open_polygon = mock.Mock()
        generation_functions = types.SimpleNamespace(
            add_doors=mock.Mock(return_value={}),
            add_lights=mock.Mock(),
            add_skybox=mock.Mock(),
            add_exterior_walls=mock.Mock(),
            add_rooms=mock.Mock(),
            add_floor_objects=mock.Mock(),
            randomize_object_colors=mock.Mock(),
            randomize_object_states=mock.Mock(),
            add_wall_objects=mock.Mock(),
            add_small_objects=mock.Mock(),
        )
        partial_house = types.SimpleNamespace(
            house_structure=types.SimpleNamespace(
                boundary_groups={},
                xz_poly_map={},
            ),
            room_spec=types.SimpleNamespace(room_type_map={2: "LivingRoom"}),
            rooms={2: types.SimpleNamespace(open_polygon=open_polygon)},
            doors=[],
            windows=[],
            objects=[],
            procedural_parameters={"lights": []},
            advance_sampling_stage=mock.Mock(),
        )
        generator = types.SimpleNamespace(
            generation_functions=generation_functions,
            controller=object(),
            pt_db=object(),
            split="train",
        )
        precomputed_door_polygons = {2: [object()]}

        with mock.patch.object(orchestration, "box", side_effect=recording_box):
            orchestration._run_floor_generation_stages(
                generator=generator,
                partial_house=partial_house,
                floor_index=0,
                stair_host_room_id=2,
                stair_core=core,
                sampling_vars=types.SimpleNamespace(max_floor_objects=1),
                skybox_source=None,
                door_polygons=precomputed_door_polygons,
            )

        generation_functions.add_doors.assert_not_called()
        self.assertIs(
            generation_functions.add_rooms.call_args.kwargs["door_polygons"],
            precomputed_door_polygons,
        )
        self.assertEqual(box_calls, [core.bounds.as_tuple()])
        self.assertEqual(buffer_calls, [(0.8, 2)])
        reservation = open_polygon.subtract.call_args.args[0]
        self.assertEqual(reservation.bounds, (3.2, 4.2, 6.0, 12.3))
        open_polygon.subtract.assert_called_once_with(reservation)

    def test_qualify_floor_ids_updates_references_and_descendants(self):
        room_wall = "wall|8|0.00|0.00|1.00|0.00"
        exterior_wall = "wall|exterior|0.00|0.00|1.00|0.00"
        partial_house = types.SimpleNamespace(
            walls=[{"id": room_wall}, {"id": exterior_wall}],
            doors=[
                {
                    "id": "door|8|9",
                    "room0": "room|8",
                    "room1": "room|9",
                    "wall0": room_wall,
                    "wall1": exterior_wall,
                }
            ],
            windows=[
                {
                    "id": "window|8|0",
                    "wall0": room_wall,
                    "wall1": exterior_wall,
                }
            ],
            objects=[
                {
                    "id": "8|0",
                    "children": [{"id": "8|0|0", "children": []}],
                }
            ],
            procedural_parameters={
                "lights": [
                    {"id": "DirectionalLight", "type": "directional"},
                    {"id": "light_8", "type": "point"},
                ]
            },
        )

        orchestration._qualify_floor_ids(partial_house, floor_index=2)

        wall_ids = {wall["id"] for wall in partial_house.walls}
        self.assertEqual(len(wall_ids), 2)
        self.assertTrue(all("|floor-2|" in wall_id for wall_id in wall_ids))
        self.assertTrue(
            all(wall["floorId"] == "floor|2" for wall in partial_house.walls)
        )
        door = partial_house.doors[0]
        window = partial_house.windows[0]
        self.assertIn(door["wall0"], wall_ids)
        self.assertIn(door["wall1"], wall_ids)
        self.assertIn(window["wall0"], wall_ids)
        self.assertIn(window["wall1"], wall_ids)
        self.assertEqual(door["floorId"], "floor|2")
        self.assertEqual(window["floorId"], "floor|2")
        self.assertEqual(door["room0"], "room|8")
        self.assertEqual(door["room1"], "room|9")
        self.assertEqual(partial_house.objects[0]["floorId"], "floor|2")
        self.assertEqual(partial_house.objects[0]["children"][0]["floorId"], "floor|2")
        directional, point = partial_house.procedural_parameters["lights"]
        self.assertNotIn("floorId", directional)
        self.assertEqual(point["floorId"], "floor|2")
        self.assertIn("|floor-2|", point["id"])

    def test_skybox_policy_reuses_style_but_preserves_light_identity(self):
        source = types.SimpleNamespace(
            procedural_parameters={
                "skyboxId": "Skybox_Blue",
                "skyboxColor": {"r": 0.1, "g": 0.2, "b": 0.3},
                "lights": [
                    {
                        "id": "ground-directional",
                        "type": "directional",
                        "intensity": 0.4,
                    },
                    {
                        "id": "ground-point",
                        "type": "point",
                        "position": {"x": 1, "y": 2, "z": 3},
                        "intensity": 1.7,
                        "rgb": {"r": 0.8, "g": 0.7, "b": 0.6},
                    },
                ],
            }
        )
        target = types.SimpleNamespace(
            procedural_parameters={
                "skyboxId": "old",
                "skyboxColor": {"r": 0, "g": 0, "b": 0},
                "lights": [
                    {
                        "id": "upper-directional",
                        "type": "directional",
                        "intensity": 9.0,
                    },
                    {
                        "id": "upper-point",
                        "type": "point",
                        "position": {"x": 7, "y": 8, "z": 9},
                        "intensity": 0.1,
                    },
                ],
            }
        )

        orchestration._apply_house_skybox_policy(source, target)

        parameters = target.procedural_parameters
        self.assertEqual(parameters["skyboxId"], "Skybox_Blue")
        self.assertEqual(parameters["skyboxColor"], {"r": 0.1, "g": 0.2, "b": 0.3})
        directional, point = parameters["lights"]
        self.assertEqual(directional["intensity"], 9.0)
        self.assertEqual(point["id"], "upper-point")
        self.assertEqual(point["position"], {"x": 7, "y": 8, "z": 9})
        self.assertEqual(point["intensity"], 1.7)
        self.assertEqual(point["rgb"], {"r": 0.8, "g": 0.7, "b": 0.6})
        self.assertIsNot(point["rgb"], source.procedural_parameters["lights"][1]["rgb"])

    def test_shared_dims_validation_and_stability(self):
        self.assertIsNone(
            orchestration._shared_dims_value(types.SimpleNamespace(dims=None))
        )
        self.assertEqual(
            orchestration._shared_dims_value(types.SimpleNamespace(dims=(12, 8))),
            (12, 8),
        )

        calls = []

        def dims():
            calls.append(True)
            return (14, 9)

        house_spec = types.SimpleNamespace(dims=dims)
        self.assertEqual(orchestration._shared_dims_value(house_spec), (14, 9))
        self.assertEqual(orchestration._shared_dims_value(house_spec), (14, 9))
        self.assertEqual(len(calls), 2)
        self.assertIs(house_spec.dims, dims)

        invalid_values = [
            (0, 8),
            (-1, 8),
            (True, 8),
            (12.5, 8),
            (12,),
            (12, 8, 4),
            None,
        ]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(
                orchestration.InvalidMultiFloorPlan
            ):
                orchestration._shared_dims_value(
                    types.SimpleNamespace(dims=lambda value=invalid: value)
                )


if __name__ == "__main__":
    unittest.main()
