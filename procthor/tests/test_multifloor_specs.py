"""Pure tests for multi-floor public specifications and sampling."""

import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATION_ROOT = REPO_ROOT / "procthor" / "generation"
UTILS_ROOT = REPO_ROOT / "procthor" / "utils"


def _install_package_shell(package_name, package_root):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package


def _load_file_module(qualified_name, path):
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_generation_module(module_name, filename):
    """Load pure modules without executing generation/__init__.py."""

    package_name = "procthor.generation"
    _install_package_shell(package_name, GENERATION_ROOT)
    return _load_file_module(
        "{}.{}".format(package_name, module_name), GENERATION_ROOT / filename
    )


# Both package initializers import optional AI2-THOR dependencies. These tests
# intentionally load only the pure types/specification modules.
_install_package_shell("procthor.utils", UTILS_ROOT)
_load_file_module("procthor.utils.types", UTILS_ROOT / "types.py")
room_specs = _load_generation_module("room_specs", "room_specs.py")
multifloor_specs = _load_generation_module("multifloor_specs", "multifloor_specs.py")

from procthor.utils.types import LeafRoom  # noqa: E402


def make_room_spec(spec_id, room_types=("LivingRoom",)):
    return room_specs.RoomSpec(
        room_spec_id=spec_id,
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2 + index, ratio=1, room_type=room_type)
            for index, room_type in enumerate(room_types)
        ],
    )


def make_house_spec(spec_id, num_floors):
    return multifloor_specs.HouseSpec(
        house_spec_id=spec_id,
        floors=[
            multifloor_specs.FloorSpec(
                room_spec=make_room_spec("{}-floor-{}".format(spec_id, index)),
                stair_host_room_id=2,
            )
            for index in range(num_floors)
        ],
        dims=(12, 10),
    )


class MultiFloorSpecTests(unittest.TestCase):
    def setUp(self):
        self._global_random_state = random.getstate()

    def tearDown(self):
        random.setstate(self._global_random_state)

    def test_house_spec_accepts_only_two_or_three_floors(self):
        self.assertEqual(make_house_spec("two", 2).num_floors, 2)
        self.assertEqual(make_house_spec("three", 3).num_floors, 3)

        one_floor = [multifloor_specs.FloorSpec(make_room_spec("one"), 2)]
        with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
            multifloor_specs.HouseSpec("invalid-one", one_floor)

        four_floors = [
            multifloor_specs.FloorSpec(make_room_spec("four-{}".format(index)), 2)
            for index in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
            multifloor_specs.HouseSpec("invalid-four", four_floors)

    def test_explicit_stair_host_must_be_a_leaf_room(self):
        room_spec = make_room_spec("hosts", ("Kitchen", "LivingRoom"))
        floor = multifloor_specs.FloorSpec(room_spec, stair_host_room_id=3)
        self.assertEqual(floor.stair_host_room_id, 3)

        with self.assertRaisesRegex(ValueError, "LivingRoom or Bedroom"):
            multifloor_specs.FloorSpec(room_spec, stair_host_room_id=2)

        with self.assertRaisesRegex(ValueError, "does not exist"):
            multifloor_specs.FloorSpec(room_spec, stair_host_room_id=99)
        with self.assertRaises(TypeError):
            multifloor_specs.FloorSpec(room_spec, stair_host_room_id=True)

    def test_supplied_rng_is_deterministic(self):
        sampler = multifloor_specs.HouseSpecSampler(
            house_specs=[
                make_house_spec("two-a", 2),
                make_house_spec("two-b", 2),
                make_house_spec("three", 3),
            ],
            weights=[1, 3, 2],
        )
        first = sampler.sample(k=20, rng=random.Random(412), num_floors=2)
        second = sampler.sample(k=20, rng=random.Random(412), num_floors=2)
        self.assertEqual(
            [house.house_spec_id for house in first],
            [house.house_spec_id for house in second],
        )
        self.assertEqual({house.num_floors for house in first}, {2})

    def test_samples_and_lookup_are_deep_copy_isolated(self):
        template = make_house_spec("template", 2)
        sampler = multifloor_specs.HouseSpecSampler([template])
        sampled = sampler.sample(rng=random.Random(1))
        looked_up = sampler["template"]

        sampled.floors[0].room_spec.room_spec_id = "mutated"
        sampled.floors[0].room_spec.spec[0].ratio = 99
        looked_up.floors[1].stair_host_room_id = None

        fresh = sampler.sample(rng=random.Random(1))
        self.assertEqual(fresh.floors[0].room_spec.room_spec_id, "template-floor-0")
        self.assertEqual(fresh.floors[0].room_spec.spec[0].ratio, 1)
        self.assertEqual(fresh.floors[1].stair_host_room_id, 2)
        self.assertEqual(template.floors[0].room_spec.spec[0].ratio, 1)

    def test_residential_variants_encode_all_approved_choices(self):
        sampler = multifloor_specs.ResidentialHouseSpecSampler()
        by_floor_count = {
            count: [
                (spec, weight)
                for spec, weight in zip(sampler.house_specs, sampler._weights)
                if spec.num_floors == count
            ]
            for count in (2, 3)
        }
        self.assertEqual(len(by_floor_count[2]), 4)
        self.assertEqual(len(by_floor_count[3]), 8)
        self.assertEqual({weight for _, weight in by_floor_count[2]}, {0.25})
        self.assertEqual({weight for _, weight in by_floor_count[3]}, {0.125})
        self.assertEqual(sum(weight for _, weight in by_floor_count[2]), 1.0)
        self.assertEqual(sum(weight for _, weight in by_floor_count[3]), 1.0)

        encoded_choices = {2: set(), 3: set()}
        for count, variants in by_floor_count.items():
            for house_spec, _ in variants:
                ground = house_spec.floors[0]
                ground_types = list(ground.room_spec.room_type_map.values())
                self.assertEqual(ground_types.count("Kitchen"), 1)
                self.assertEqual(ground_types.count("LivingRoom"), 1)
                self.assertIn(ground_types.count("Bathroom"), {0, 1})
                self.assertEqual(
                    ground.room_spec.room_type_map[ground.stair_host_room_id],
                    "LivingRoom",
                )

                choice = [ground_types.count("Bathroom") == 1]
                for upper in house_spec.floors[1:]:
                    upper_types = list(upper.room_spec.room_type_map.values())
                    self.assertEqual(upper_types.count("LivingRoom"), 1)
                    self.assertEqual(upper_types.count("Bathroom"), 1)
                    self.assertIn(upper_types.count("Bedroom"), {1, 2})
                    self.assertEqual(
                        upper.room_spec.room_type_map[upper.stair_host_room_id],
                        "LivingRoom",
                    )
                    choice.append(upper_types.count("Bedroom"))
                encoded_choices[count].add(tuple(choice))

        self.assertEqual(
            encoded_choices[2],
            {(bathroom, bedrooms) for bathroom in (False, True) for bedrooms in (1, 2)},
        )
        self.assertEqual(
            encoded_choices[3],
            {
                (bathroom, first_bedrooms, second_bedrooms)
                for bathroom in (False, True)
                for first_bedrooms in (1, 2)
                for second_bedrooms in (1, 2)
            },
        )

    def test_supplied_rng_and_sampler_construction_do_not_touch_global_random(self):
        random.seed(90210)
        before_construction = random.getstate()
        sampler = multifloor_specs.ResidentialHouseSpecSampler()
        self.assertEqual(random.getstate(), before_construction)

        private_rng = random.Random(7)
        before_sampling = random.getstate()
        sampler.sample(k=12, rng=private_rng, num_floors=3)
        self.assertEqual(random.getstate(), before_sampling)


if __name__ == "__main__":
    unittest.main()
