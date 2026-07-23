"""Public specifications and samplers for multi-floor houses.

Multi-floor generation deliberately uses a separate specification layer from the
legacy :class:`RoomSpec` sampler.  A ``RoomSpec`` still describes one horizontal
floor; ``FloorSpec`` and ``HouseSpec`` compose those floor-local descriptions into
one house without changing the single-floor API.
"""

import copy
import itertools
import math
import random
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from attr import Attribute, field
from attrs import define

from procthor.utils.types import LeafRoom, MetaRoom

from .room_specs import RoomSpec


HouseDims = Union[Tuple[int, int], Callable[[], Tuple[int, int]]]


@define
class FloorSpec:
    """A floor-local room specification and optional stair host.

    Room ids are local to the floor.  Multi-floor generation is responsible for
    mapping them to globally unique ids when it assembles the final house.
    ``stair_host_room_id`` is likewise a floor-local leaf-room id.
    """

    room_spec: RoomSpec = field()
    stair_host_room_id: Optional[int] = field(default=None)

    @room_spec.validator
    def _valid_room_spec(self, attribute: Attribute, value: RoomSpec) -> None:
        if not isinstance(value, RoomSpec):
            raise TypeError("room_spec must be a RoomSpec instance.")

    @stair_host_room_id.validator
    def _valid_stair_host_room_id(
        self, attribute: Attribute, value: Optional[int]
    ) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("stair_host_room_id must be an integer room id or None.")
        if value not in self.room_spec.room_type_map:
            raise ValueError(
                "stair_host_room_id must identify a leaf room in room_spec; "
                f"room {value} does not exist."
            )
        room_type = self.room_spec.room_type_map[value]
        if room_type not in {"LivingRoom", "Bedroom"}:
            raise ValueError(
                "stair_host_room_id must identify a LivingRoom or Bedroom; "
                f"room {value} is a {room_type}."
            )


@define
class HouseSpec:
    """A complete two- or three-floor house specification.

    ``dims`` is the shared aligned footprint for all floors.  It accepts either
    a concrete ``(x_size, z_size)`` pair or a zero-argument callable, matching the
    existing ``RoomSpec.dims`` convention.
    """

    house_spec_id: str = field()
    floors: List[FloorSpec] = field(converter=list)
    dims: Optional[HouseDims] = field(default=None)

    @house_spec_id.validator
    def _valid_house_spec_id(self, attribute: Attribute, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("house_spec_id must be a non-empty string.")

    @floors.validator
    def _valid_floors(self, attribute: Attribute, value: List[FloorSpec]) -> None:
        if len(value) not in {2, 3}:
            raise ValueError(
                "Multi-floor HouseSpec supports exactly 2 or 3 floors; "
                f"received {len(value)}."
            )
        if any(not isinstance(floor, FloorSpec) for floor in value):
            raise TypeError("floors must contain only FloorSpec instances.")

    @dims.validator
    def _valid_dims(self, attribute: Attribute, value: Optional[HouseDims]) -> None:
        if value is None or callable(value):
            return
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("dims must be a two-integer tuple, a callable, or None.")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or dimension <= 0
            for dimension in value
        ):
            raise ValueError("dims must contain two positive integers.")

    @property
    def num_floors(self) -> int:
        return len(self.floors)


@define
class HouseSpecSampler:
    """Weighted sampler over immutable-by-convention ``HouseSpec`` templates.

    Sampled values are deep copies.  This is important because the existing
    floor-generation pipeline annotates room nodes while it works.
    """

    house_specs: List[HouseSpec] = field(converter=list)
    weights: Optional[List[float]] = field(
        default=None,
        converter=lambda value: None if value is None else list(value),
    )

    house_spec_map: Dict[str, HouseSpec] = field(init=False)
    _weights: List[float] = field(init=False)

    @house_specs.validator
    def _valid_house_specs(self, attribute: Attribute, value: List[HouseSpec]) -> None:
        if not value:
            raise ValueError("house_specs must contain at least one HouseSpec.")
        if any(not isinstance(house_spec, HouseSpec) for house_spec in value):
            raise TypeError("house_specs must contain only HouseSpec instances.")
        house_spec_ids = [house_spec.house_spec_id for house_spec in value]
        if len(set(house_spec_ids)) != len(house_spec_ids):
            duplicate = next(
                house_spec_id
                for house_spec_id in house_spec_ids
                if house_spec_ids.count(house_spec_id) > 1
            )
            raise ValueError(
                "Each HouseSpec must have a unique house_spec_id; "
                f"received duplicate {duplicate!r}."
            )

    @weights.validator
    def _valid_weights(
        self, attribute: Attribute, value: Optional[List[float]]
    ) -> None:
        if value is None:
            return
        if len(value) != len(self.house_specs):
            raise ValueError("weights must have the same length as house_specs.")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
            for weight in value
        ):
            raise ValueError("weights must contain only finite values greater than 0.")

    def __attrs_post_init__(self) -> None:
        self.house_spec_map = {
            house_spec.house_spec_id: house_spec for house_spec in self.house_specs
        }
        self._weights = (
            [1.0] * len(self.house_specs)
            if self.weights is None
            else [float(weight) for weight in self.weights]
        )

    def __getitem__(self, house_spec_id: str) -> HouseSpec:
        return copy.deepcopy(self.house_spec_map[house_spec_id])

    @staticmethod
    def _pick_index(weights: List[float], rng: Any) -> int:
        """Pick one weighted index using only the RNG's ``random`` method."""

        total = sum(weights)
        target = float(rng.random()) * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                return index
        # Some random-like objects may round ``random()`` to exactly 1.0.
        return len(weights) - 1

    def sample(
        self,
        k: int = 1,
        rng: Optional[Any] = None,
        num_floors: Optional[int] = None,
    ) -> Union[HouseSpec, List[HouseSpec]]:
        """Sample one or more specs, optionally filtering by floor count.

        ``rng`` may be the :mod:`random` module, ``random.Random``, or another
        object exposing a ``random()`` method.  Supplying an RNG makes the result
        independent of module-global random state.
        """

        if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if num_floors is not None and num_floors not in {2, 3}:
            raise ValueError("num_floors must be 2, 3, or None.")
        rng = random if rng is None else rng
        if not callable(getattr(rng, "random", None)):
            raise TypeError("rng must expose a callable random() method.")

        candidates = []
        candidate_weights = []
        for house_spec, weight in zip(self.house_specs, self._weights):
            if num_floors is None or house_spec.num_floors == num_floors:
                candidates.append(house_spec)
                candidate_weights.append(weight)
        if not candidates:
            raise KeyError(f"No HouseSpec has num_floors={num_floors}.")

        samples = [
            copy.deepcopy(
                candidates[self._pick_index(weights=candidate_weights, rng=rng)]
            )
            for _ in range(int(k))
        ]
        return samples[0] if k == 1 else samples


def _ground_floor_room_spec(include_bathroom: bool) -> RoomSpec:
    rooms = [
        LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
        LeafRoom(room_id=3, ratio=4, room_type="LivingRoom"),
    ]
    if include_bathroom:
        rooms.append(
            LeafRoom(
                room_id=4,
                ratio=1,
                room_type="Bathroom",
                avoid_doors_from_metarooms=True,
            )
        )
    return RoomSpec(
        room_spec_id=(
            "residential-ground-with-bathroom"
            if include_bathroom
            else "residential-ground"
        ),
        sampling_weight=1,
        spec=[MetaRoom(ratio=sum(room.ratio for room in rooms), children=rooms)],
    )


def _upper_floor_room_spec(num_bedrooms: int, floor_index: int) -> RoomSpec:
    rooms = [
        LeafRoom(room_id=2, ratio=4, room_type="LivingRoom"),
        LeafRoom(
            room_id=3,
            ratio=1,
            room_type="Bathroom",
            avoid_doors_from_metarooms=True,
        ),
    ]
    rooms.extend(
        LeafRoom(room_id=4 + bedroom_index, ratio=2, room_type="Bedroom")
        for bedroom_index in range(num_bedrooms)
    )
    return RoomSpec(
        room_spec_id=(f"residential-upper-{floor_index}-{num_bedrooms}-bedroom"),
        sampling_weight=1,
        spec=[MetaRoom(ratio=sum(room.ratio for room in rooms), children=rooms)],
    )


def _residential_house_specs() -> Tuple[List[HouseSpec], List[float]]:
    """Enumerate variants so each approved binary choice remains independent."""

    house_specs = []
    weights = []
    for num_floors in (2, 3):
        # Ground-floor bathroom and each upper-floor bedroom count are fair,
        # independent choices.  Uniform weighting within a floor count gives
        # exactly those Bernoulli distributions.
        for include_ground_bathroom, *upper_bedroom_bits in itertools.product(
            (False, True), repeat=num_floors
        ):
            bedroom_counts = [1 + int(bit) for bit in upper_bedroom_bits]
            floors = [
                FloorSpec(
                    room_spec=_ground_floor_room_spec(include_ground_bathroom),
                    stair_host_room_id=3,
                )
            ]
            floors.extend(
                FloorSpec(
                    room_spec=_upper_floor_room_spec(
                        num_bedrooms=num_bedrooms, floor_index=floor_index
                    ),
                    stair_host_room_id=2,
                )
                for floor_index, num_bedrooms in enumerate(bedroom_counts, start=1)
            )
            ground_code = "gb1" if include_ground_bathroom else "gb0"
            upper_code = "-".join(f"u{count}" for count in bedroom_counts)
            house_specs.append(
                HouseSpec(
                    house_spec_id=(
                        f"residential-{num_floors}-floor-{ground_code}-{upper_code}"
                    ),
                    floors=floors,
                )
            )
            # Equal total mass for two- and three-floor houses when unfiltered;
            # filtering by num_floors still leaves all within-floor choices fair.
            weights.append(1.0 / (2**num_floors))
    return house_specs, weights


class ResidentialHouseSpecSampler(HouseSpecSampler):
    """Built-in residential sampler for the supported two/three-floor range."""

    def __init__(self) -> None:
        house_specs, weights = _residential_house_specs()
        super().__init__(house_specs=house_specs, weights=weights)


RESIDENTIAL_HOUSE_SPEC_SAMPLER = ResidentialHouseSpecSampler()
"""Default deterministic multi-floor residential specification sampler."""

# Naming parallel to ``PROCTHOR10K_ROOM_SPEC_SAMPLER`` for callers that prefer a
# dataset-oriented constant name.
PROCTHOR_MULTIFLOOR_HOUSE_SPEC_SAMPLER = RESIDENTIAL_HOUSE_SPEC_SAMPLER


__all__ = [
    "FloorSpec",
    "HouseDims",
    "HouseSpec",
    "HouseSpecSampler",
    "PROCTHOR_MULTIFLOOR_HOUSE_SPEC_SAMPLER",
    "RESIDENTIAL_HOUSE_SPEC_SAMPLER",
    "ResidentialHouseSpecSampler",
]
