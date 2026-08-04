"""Pure geometry and schema helpers for schema-2 multi-floor houses.

This module intentionally has no AI2-THOR (or NumPy/Shapely) dependency.  It
contains the deterministic geometry contract shared by the generator and the
engine implementation:

* floors are 3.0 m apart, with a 0.2 m slab and 2.8 m of clear height;
* one 1.2 m by 6.5 m stair envelope is reserved through every floor;
* a reference straight flight is 1.0 m wide and has a 4.5 m run; and
* the second connector in a three-floor house is rotated by 180 degrees.

The helpers return ordinary dictionaries/lists so their output can be inserted
directly into the procedural-house JSON assembled by the generation pipeline.
"""

import copy
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MULTI_FLOOR_SCHEMA = "2.0.0"
"""Schema used only by houses containing two or three floors."""

MIN_MULTI_FLOORS = 2
MAX_MULTI_FLOORS = 3

FLOOR_PITCH = 3.0
SLAB_THICKNESS = 0.2
CLEAR_HEIGHT = 2.8

STAIR_CORE_WIDTH = 1.2
STAIR_CORE_LENGTH = 6.5
STAIR_FLIGHT_WIDTH = 1.0
STAIR_FLIGHT_RUN = 4.5

DEFAULT_STAIR_ASSET_ID = "Staircase_Straight_3m_1m_4_5m"

_EPSILON = 1e-8


class MultiFloorGeometryError(ValueError):
    """Base error for an invalid multi-floor geometry request."""


class InvalidSharedBoundary(MultiFloorGeometryError):
    """Raised when floor footprints have no usable shared rectangle."""


class StairCoreDoesNotFit(MultiFloorGeometryError):
    """Raised when the fixed stair envelope cannot fit in the shared boundary."""


class InvalidOpening(MultiFloorGeometryError):
    """Raised when a slab opening is not a proper subset of its surface."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise MultiFloorGeometryError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MultiFloorGeometryError("{} must be a finite number".format(name))
    if not math.isfinite(result):
        raise MultiFloorGeometryError("{} must be a finite number".format(name))
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise MultiFloorGeometryError("{} must be greater than zero".format(name))
    return result


def _floor_index(value: Any, name: str = "floor_index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultiFloorGeometryError("{} must be a non-negative integer".format(name))
    return value


def _room_number(value: Any, name: str = "room_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise MultiFloorGeometryError(
            "{} must be an integer >= 2 (0 and 1 are reserved)".format(name)
        )
    return value


def _num_floors(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiFloorGeometryError("num_floors must be 2 or 3")
    if not MIN_MULTI_FLOORS <= value <= MAX_MULTI_FLOORS:
        raise MultiFloorGeometryError("num_floors must be 2 or 3")
    return value


class Rectangle:
    """Validated axis-aligned rectangle in the world x/z plane."""

    __slots__ = ("min_x", "min_z", "max_x", "max_z")

    def __init__(self, min_x: Any, min_z: Any, max_x: Any, max_z: Any) -> None:
        self.min_x = _finite_float(min_x, "min_x")
        self.min_z = _finite_float(min_z, "min_z")
        self.max_x = _finite_float(max_x, "max_x")
        self.max_z = _finite_float(max_z, "max_z")
        if self.max_x - self.min_x <= _EPSILON:
            raise MultiFloorGeometryError("rectangle must have positive x extent")
        if self.max_z - self.min_z <= _EPSILON:
            raise MultiFloorGeometryError("rectangle must have positive z extent")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_z - self.min_z

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_z + self.max_z) / 2)

    def contains(self, other: "Rectangle", margin: float = 0.0) -> bool:
        margin = _finite_float(margin, "margin")
        return (
            other.min_x >= self.min_x + margin - _EPSILON
            and other.max_x <= self.max_x - margin + _EPSILON
            and other.min_z >= self.min_z + margin - _EPSILON
            and other.max_z <= self.max_z - margin + _EPSILON
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_x, self.min_z, self.max_x, self.max_z)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Rectangle) and self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return "Rectangle(min_x={!r}, min_z={!r}, max_x={!r}, max_z={!r})".format(
            self.min_x, self.min_z, self.max_x, self.max_z
        )


def as_rectangle(value: Any) -> Rectangle:
    """Coerce common rectangle forms into :class:`Rectangle`.

    Accepted forms are a ``Rectangle``, a four-number sequence ordered as
    ``(min_x, min_z, max_x, max_z)``, four (optionally closed with a fifth)
    corner pairs, or a mapping using snake/camel-case extrema.  A mapping with
    ``min``/``max`` vector dictionaries is also accepted.
    """

    if isinstance(value, Rectangle):
        return value

    if isinstance(value, Mapping):
        key_sets = (
            ("min_x", "min_z", "max_x", "max_z"),
            ("minX", "minZ", "maxX", "maxZ"),
        )
        for keys in key_sets:
            if all(key in value for key in keys):
                return Rectangle(*(value[key] for key in keys))
        if "min" in value and "max" in value:
            minimum = value["min"]
            maximum = value["max"]
            if isinstance(minimum, Mapping) and isinstance(maximum, Mapping):
                if all(key in minimum and key in maximum for key in ("x", "z")):
                    return Rectangle(
                        minimum["x"], minimum["z"], maximum["x"], maximum["z"]
                    )
        raise MultiFloorGeometryError("unrecognized rectangle mapping")

    if isinstance(value, (str, bytes)):
        raise MultiFloorGeometryError("rectangle cannot be text")
    try:
        values = list(value)
    except TypeError:
        raise MultiFloorGeometryError("unrecognized rectangle value")

    if len(values) == 4 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in values
    ):
        return Rectangle(*values)

    if len(values) == 5 and values[0] == values[-1]:
        values = values[:-1]
    if len(values) != 4:
        raise MultiFloorGeometryError("rectangle polygon must have four corners")

    points = []
    for index, point in enumerate(values):
        try:
            x, z = point
        except (TypeError, ValueError):
            raise MultiFloorGeometryError(
                "rectangle corner {} must be an (x, z) pair".format(index)
            )
        points.append((_finite_float(x, "corner x"), _finite_float(z, "corner z")))
    xs = sorted(set(point[0] for point in points))
    zs = sorted(set(point[1] for point in points))
    if (
        len(xs) != 2
        or len(zs) != 2
        or set(points)
        != {
            (xs[0], zs[0]),
            (xs[1], zs[0]),
            (xs[1], zs[1]),
            (xs[0], zs[1]),
        }
    ):
        raise MultiFloorGeometryError("polygon is not an axis-aligned rectangle")
    return Rectangle(xs[0], zs[0], xs[1], zs[1])


class StairGeometryContract:
    """Validated dimensions expected by the curated straight-stair prefab."""

    __slots__ = (
        "floor_pitch",
        "slab_thickness",
        "clear_height",
        "core_width",
        "core_length",
        "flight_width",
        "flight_run",
    )

    def __init__(
        self,
        floor_pitch: float = FLOOR_PITCH,
        slab_thickness: float = SLAB_THICKNESS,
        clear_height: float = CLEAR_HEIGHT,
        core_width: float = STAIR_CORE_WIDTH,
        core_length: float = STAIR_CORE_LENGTH,
        flight_width: float = STAIR_FLIGHT_WIDTH,
        flight_run: float = STAIR_FLIGHT_RUN,
    ) -> None:
        self.floor_pitch = _positive_float(floor_pitch, "floor_pitch")
        self.slab_thickness = _positive_float(slab_thickness, "slab_thickness")
        self.clear_height = _positive_float(clear_height, "clear_height")
        self.core_width = _positive_float(core_width, "core_width")
        self.core_length = _positive_float(core_length, "core_length")
        self.flight_width = _positive_float(flight_width, "flight_width")
        self.flight_run = _positive_float(flight_run, "flight_run")
        if abs(self.clear_height + self.slab_thickness - self.floor_pitch) > _EPSILON:
            raise MultiFloorGeometryError(
                "clear_height + slab_thickness must equal floor_pitch"
            )
        if self.flight_width > self.core_width + _EPSILON:
            raise MultiFloorGeometryError("flight_width cannot exceed core_width")
        if self.flight_run > self.core_length + _EPSILON:
            raise MultiFloorGeometryError("flight_run cannot exceed core_length")

    @property
    def landing_depth(self) -> float:
        return (self.core_length - self.flight_run) / 2


DEFAULT_STAIR_GEOMETRY = StairGeometryContract()


class StairCore:
    """A validated placement of the shared stair envelope."""

    __slots__ = ("bounds", "long_axis", "yaw", "contract")

    def __init__(
        self,
        bounds: Any,
        long_axis: str,
        yaw: Any,
        contract: StairGeometryContract = DEFAULT_STAIR_GEOMETRY,
    ) -> None:
        if long_axis not in {"x", "z"}:
            raise MultiFloorGeometryError("long_axis must be 'x' or 'z'")
        if not isinstance(contract, StairGeometryContract):
            raise MultiFloorGeometryError("contract must be a StairGeometryContract")
        self.bounds = as_rectangle(bounds)
        self.long_axis = long_axis
        self.yaw = _finite_float(yaw, "yaw") % 360
        self.contract = contract
        expected_x = contract.core_length if long_axis == "x" else contract.core_width
        expected_z = contract.core_width if long_axis == "x" else contract.core_length
        if abs(self.bounds.width - expected_x) > _EPSILON:
            raise MultiFloorGeometryError("stair core has the wrong x extent")
        if abs(self.bounds.depth - expected_z) > _EPSILON:
            raise MultiFloorGeometryError("stair core has the wrong z extent")

    @property
    def center(self) -> Tuple[float, float]:
        return self.bounds.center


def _intersection(rectangles: Sequence[Rectangle]) -> Rectangle:
    if not rectangles:
        raise InvalidSharedBoundary("at least one floor boundary is required")
    min_x = max(rect.min_x for rect in rectangles)
    min_z = max(rect.min_z for rect in rectangles)
    max_x = min(rect.max_x for rect in rectangles)
    max_z = min(rect.max_z for rect in rectangles)
    try:
        return Rectangle(min_x, min_z, max_x, max_z)
    except MultiFloorGeometryError:
        raise InvalidSharedBoundary("floor boundaries have no shared rectangle")


def place_stair_core(
    shared_boundary: Any,
    margin: float = 0.0,
    preferred_axis: Optional[str] = None,
    center: Optional[Sequence[float]] = None,
    contract: StairGeometryContract = DEFAULT_STAIR_GEOMETRY,
) -> StairCore:
    """Place the fixed stair core deterministically inside a shared boundary.

    With no explicit center the envelope is centered in the boundary.  Its long
    axis follows the longer boundary dimension when both orientations fit.  A
    preferred ``"x"`` or ``"z"`` orientation can be requested explicitly.
    """

    boundary = as_rectangle(shared_boundary)
    margin = _finite_float(margin, "margin")
    if margin < 0:
        raise MultiFloorGeometryError("margin cannot be negative")
    if preferred_axis not in {None, "x", "z"}:
        raise MultiFloorGeometryError("preferred_axis must be None, 'x', or 'z'")
    if not isinstance(contract, StairGeometryContract):
        raise MultiFloorGeometryError("contract must be a StairGeometryContract")

    fits_x = (
        boundary.width + _EPSILON >= contract.core_length + 2 * margin
        and boundary.depth + _EPSILON >= contract.core_width + 2 * margin
    )
    fits_z = (
        boundary.width + _EPSILON >= contract.core_width + 2 * margin
        and boundary.depth + _EPSILON >= contract.core_length + 2 * margin
    )
    if preferred_axis == "x":
        long_axis = "x" if fits_x else None
    elif preferred_axis == "z":
        long_axis = "z" if fits_z else None
    elif fits_x and fits_z:
        long_axis = "x" if boundary.width > boundary.depth else "z"
    elif fits_x:
        long_axis = "x"
    elif fits_z:
        long_axis = "z"
    else:
        long_axis = None
    if long_axis is None:
        raise StairCoreDoesNotFit(
            "a {:.1f} x {:.1f} m stair core does not fit the shared boundary".format(
                contract.core_width, contract.core_length
            )
        )

    if center is None:
        center_x, center_z = boundary.center
    else:
        try:
            center_x, center_z = center
        except (TypeError, ValueError):
            raise MultiFloorGeometryError("center must be an (x, z) pair")
        center_x = _finite_float(center_x, "center x")
        center_z = _finite_float(center_z, "center z")

    size_x = contract.core_length if long_axis == "x" else contract.core_width
    size_z = contract.core_width if long_axis == "x" else contract.core_length
    bounds = Rectangle(
        center_x - size_x / 2,
        center_z - size_z / 2,
        center_x + size_x / 2,
        center_z + size_z / 2,
    )
    if not boundary.contains(bounds, margin=margin):
        raise StairCoreDoesNotFit(
            "the requested stair core center is outside the boundary"
        )

    # The prefab's local +z axis is its direction of ascent.
    yaw = 90.0 if long_axis == "x" else 0.0
    return StairCore(bounds=bounds, long_axis=long_axis, yaw=yaw, contract=contract)


def place_shared_stair_core(
    floor_boundaries: Iterable[Any],
    margin: float = 0.0,
    preferred_axis: Optional[str] = None,
    center: Optional[Sequence[float]] = None,
    contract: StairGeometryContract = DEFAULT_STAIR_GEOMETRY,
) -> StairCore:
    """Place one core in the intersection shared by all floor boundaries."""

    boundaries = [as_rectangle(boundary) for boundary in floor_boundaries]
    shared_boundary = _intersection(boundaries)
    return place_stair_core(
        shared_boundary=shared_boundary,
        margin=margin,
        preferred_axis=preferred_axis,
        center=center,
        contract=contract,
    )


def floor_base_y(floor_index: int) -> float:
    """World y-coordinate of a floor's walkable surface."""

    return _floor_index(floor_index) * FLOOR_PITCH


def floor_ceiling_y(floor_index: int) -> float:
    """World y-coordinate of a floor's ceiling underside."""

    return floor_base_y(floor_index) + CLEAR_HEIGHT


def floor_id(floor_index: int) -> str:
    return "floor|{}".format(_floor_index(floor_index))


def room_id(room_number: int) -> str:
    return "room|{}".format(_room_number(room_number))


def _id_part(value: Any, name: str) -> str:
    part = str(value).strip()
    if not part or "|" in part:
        raise MultiFloorGeometryError("{} must be a non-empty ID token".format(name))
    return part


def floor_qualified_id(kind: str, floor_index: int, *parts: Any) -> str:
    """Return a deterministic structural ID scoped to a floor."""

    tokens = [_id_part(kind, "kind"), "floor-{}".format(_floor_index(floor_index))]
    tokens.extend(_id_part(part, "ID part") for part in parts)
    return "|".join(tokens)


def connector_id(lower_floor_index: int, connector_index: Optional[int] = None) -> str:
    lower = _floor_index(lower_floor_index, "lower_floor_index")
    if connector_index is None:
        connector_index = lower
    connector_index = _floor_index(connector_index, "connector_index")
    return "vertical-connector|floor-{}-to-{}|connector-{}".format(
        lower, lower + 1, connector_index
    )


def make_global_room_id_maps(
    floor_room_ids: Sequence[Iterable[int]], start_at: int = 2
) -> List[Dict[int, int]]:
    """Allocate deterministic global room numbers for per-floor input IDs.

    Input IDs may repeat on different floors.  IDs are sorted within each floor
    and assigned densely from ``start_at``; 0 and 1 remain reserved.
    """

    _num_floors(len(floor_room_ids))
    start_at = _room_number(start_at, "start_at")
    next_id = start_at
    result = []
    for floor_index, input_ids in enumerate(floor_room_ids):
        ids = list(input_ids)
        if not ids:
            raise MultiFloorGeometryError(
                "floor {} must contain at least one room".format(floor_index)
            )
        for input_id in ids:
            _room_number(input_id, "input room ID")
        if len(set(ids)) != len(ids):
            raise MultiFloorGeometryError(
                "floor {} contains duplicate input room IDs".format(floor_index)
            )
        mapping = {}
        for input_id in sorted(ids):
            mapping[input_id] = next_id
            next_id += 1
        result.append(mapping)
    return result


def rectangle_polygon(rectangle: Any) -> List[Tuple[float, float]]:
    """Return a counter-clockwise four-point x/z polygon."""

    rect = as_rectangle(rectangle)
    return [
        (rect.min_x, rect.min_z),
        (rect.max_x, rect.min_z),
        (rect.max_x, rect.max_z),
        (rect.min_x, rect.max_z),
    ]


def polygon_at_y(rectangle: Any, y: Any) -> List[Dict[str, float]]:
    """Return a rectangle as schema-style world-space Vector3 dictionaries."""

    world_y = _finite_float(y, "y")
    return [{"x": x, "y": world_y, "z": z} for x, z in rectangle_polygon(rectangle)]


def _rectangles_around_opening(outer: Any, opening: Any) -> List[Rectangle]:
    outer_rect = as_rectangle(outer)
    opening_rect = as_rectangle(opening)
    if not outer_rect.contains(opening_rect):
        raise InvalidOpening("opening must be contained by the outer rectangle")

    # Clamp values that differ only by floating point noise at a boundary.
    opening_rect = Rectangle(
        max(outer_rect.min_x, opening_rect.min_x),
        max(outer_rect.min_z, opening_rect.min_z),
        min(outer_rect.max_x, opening_rect.max_x),
        min(outer_rect.max_z, opening_rect.max_z),
    )
    if opening_rect.area >= outer_rect.area - _EPSILON:
        raise InvalidOpening("opening must leave some surface area")

    pieces = []

    def add(min_x: float, min_z: float, max_x: float, max_z: float) -> None:
        if max_x - min_x > _EPSILON and max_z - min_z > _EPSILON:
            pieces.append(Rectangle(min_x, min_z, max_x, max_z))

    # Full-height side strips plus strips above/below the opening form a
    # deterministic, non-overlapping partition of outer - opening.
    add(outer_rect.min_x, outer_rect.min_z, opening_rect.min_x, outer_rect.max_z)
    add(opening_rect.max_x, outer_rect.min_z, outer_rect.max_x, outer_rect.max_z)
    add(
        opening_rect.min_x,
        outer_rect.min_z,
        opening_rect.max_x,
        opening_rect.min_z,
    )
    add(
        opening_rect.min_x,
        opening_rect.max_z,
        opening_rect.max_x,
        outer_rect.max_z,
    )
    return pieces


def decompose_rectangle_around_opening(
    outer: Any, opening: Any
) -> List[List[Tuple[float, float]]]:
    """Partition an orthogonal rectangle around a rectangular opening.

    Returned polygons are counter-clockwise, pairwise interior-disjoint, and
    cover exactly ``outer - opening``.  Between one and four pieces are emitted
    depending on whether the opening touches an outer edge.
    """

    return [
        rectangle_polygon(piece)
        for piece in _rectangles_around_opening(outer=outer, opening=opening)
    ]


def make_surface_pieces(
    floor_index: int,
    room_number: int,
    outer: Any,
    surface_kind: str,
    opening: Optional[Any] = None,
    material: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create explicit actual-y floor or ceiling surface pieces for one room."""

    floor_index = _floor_index(floor_index)
    room_number = _room_number(room_number)
    if surface_kind not in {"floor", "ceiling"}:
        raise MultiFloorGeometryError("surface_kind must be 'floor' or 'ceiling'")
    y = (
        floor_base_y(floor_index)
        if surface_kind == "floor"
        else floor_ceiling_y(floor_index)
    )
    rectangles = (
        [as_rectangle(outer)]
        if opening is None
        else _rectangles_around_opening(outer=outer, opening=opening)
    )
    surfaces = []
    for piece_index, rectangle in enumerate(rectangles):
        surface = {
            "id": floor_qualified_id(
                "{}-surface".format(surface_kind),
                floor_index,
                "room-{}".format(room_number),
                "piece-{}".format(piece_index),
            ),
            "floorId": floor_id(floor_index),
            "roomId": room_id(room_number),
            "surfaceType": surface_kind,
            "polygon": polygon_at_y(rectangle, y),
        }
        if surface_kind == "floor":
            surface["slabThickness"] = SLAB_THICKNESS
        if material is not None:
            surface["material"] = copy.deepcopy(dict(material))
        surfaces.append(surface)
    return surfaces


def make_room_surface_metadata(
    floor_index: int,
    room_number: int,
    outer: Any,
    floor_opening: Optional[Any] = None,
    ceiling_opening: Optional[Any] = None,
    floor_material: Optional[Mapping[str, Any]] = None,
    ceiling_material: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build schema-2 room surface fields while retaining ``floorPolygon``.

    ``floorPolygon`` remains the semantic, uncut room footprint used by room
    ownership code.  ``floorPolygons`` and ``ceilings`` are the physical mesh
    pieces and may contain a stair opening.
    """

    floor_index = _floor_index(floor_index)
    room_number = _room_number(room_number)
    outer_rect = as_rectangle(outer)
    return {
        "floorId": floor_id(floor_index),
        "floorPolygon": polygon_at_y(outer_rect, floor_base_y(floor_index)),
        "floorPolygons": make_surface_pieces(
            floor_index=floor_index,
            room_number=room_number,
            outer=outer_rect,
            surface_kind="floor",
            opening=floor_opening,
            material=floor_material,
        ),
        "ceilings": make_surface_pieces(
            floor_index=floor_index,
            room_number=room_number,
            outer=outer_rect,
            surface_kind="ceiling",
            opening=ceiling_opening,
            material=ceiling_material,
        ),
    }


def make_floor_record(
    floor_index: int,
    room_id_map: Mapping[int, int],
    room_surfaces: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a top-level schema-2 floor record.

    ``room_id_map`` maps that floor's input room IDs to allocated global room
    numbers.  Optional ``room_surfaces`` is keyed by global room number and is
    flattened into the record's explicit physical surface arrays.
    """

    floor_index = _floor_index(floor_index)
    if not room_id_map:
        raise MultiFloorGeometryError("room_id_map cannot be empty")
    normalized_map = {}
    for input_id, global_id in room_id_map.items():
        input_id = _room_number(input_id, "input room ID")
        global_id = _room_number(global_id, "global room ID")
        normalized_map[input_id] = global_id
    if len(set(normalized_map.values())) != len(normalized_map):
        raise MultiFloorGeometryError("global room IDs must be unique on a floor")

    floor_surfaces = []
    ceiling_surfaces = []
    if room_surfaces is not None:
        expected = set(normalized_map.values())
        supplied = set(room_surfaces.keys())
        if supplied != expected:
            raise MultiFloorGeometryError(
                "room_surfaces keys must match the floor's global room IDs"
            )
        for global_id in sorted(expected):
            metadata = room_surfaces[global_id]
            if metadata.get("floorId") != floor_id(floor_index):
                raise MultiFloorGeometryError("room surface has the wrong floorId")
            floor_surfaces.extend(copy.deepcopy(metadata.get("floorPolygons", [])))
            ceiling_surfaces.extend(copy.deepcopy(metadata.get("ceilings", [])))

    return {
        "id": floor_id(floor_index),
        "index": floor_index,
        "baseY": floor_base_y(floor_index),
        "ceilingY": floor_ceiling_y(floor_index),
        "slabThickness": SLAB_THICKNESS,
        "roomIds": [normalized_map[key] for key in sorted(normalized_map)],
        "roomIdMap": {str(key): normalized_map[key] for key in sorted(normalized_map)},
        "floorSurfaces": floor_surfaces,
        "ceilingSurfaces": ceiling_surfaces,
    }


def stair_core_reservations(core: StairCore, num_floors: int) -> List[Dict[str, Any]]:
    """Create a floor-qualified reservation for the core on every level."""

    if not isinstance(core, StairCore):
        raise MultiFloorGeometryError("core must be a StairCore")
    num_floors = _num_floors(num_floors)
    reservations = []
    for floor_index in range(num_floors):
        reservations.append(
            {
                "id": floor_qualified_id("stair-core", floor_index, "reservation"),
                "floorId": floor_id(floor_index),
                "polygon": polygon_at_y(core.bounds, floor_base_y(floor_index)),
                "width": core.contract.core_width,
                "length": core.contract.core_length,
            }
        )
    return reservations


def _landing_rectangles(core: StairCore, reverse: bool) -> Tuple[Rectangle, Rectangle]:
    depth = core.contract.landing_depth
    bounds = core.bounds
    if core.long_axis == "z":
        negative = Rectangle(
            bounds.min_x, bounds.min_z, bounds.max_x, bounds.min_z + depth
        )
        positive = Rectangle(
            bounds.min_x, bounds.max_z - depth, bounds.max_x, bounds.max_z
        )
    else:
        negative = Rectangle(
            bounds.min_x, bounds.min_z, bounds.min_x + depth, bounds.max_z
        )
        positive = Rectangle(
            bounds.max_x - depth, bounds.min_z, bounds.max_x, bounds.max_z
        )
    return (positive, negative) if reverse else (negative, positive)


def stair_floor_opening(core: StairCore, floor_index: int) -> Rectangle:
    """Return the full reserved stair-core slab opening for one floor."""

    if not isinstance(core, StairCore):
        raise MultiFloorGeometryError("core must be a StairCore")
    _floor_index(floor_index)
    return core.bounds


def make_vertical_connector(
    lower_floor_index: int,
    lower_room_number: int,
    upper_room_number: int,
    core: StairCore,
    asset_id: str = DEFAULT_STAIR_ASSET_ID,
    connector_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Create one adjacent-floor straight-stair connector record."""

    lower_floor_index = _floor_index(lower_floor_index, "lower_floor_index")
    lower_room_number = _room_number(lower_room_number, "lower_room_number")
    upper_room_number = _room_number(upper_room_number, "upper_room_number")
    if not isinstance(core, StairCore):
        raise MultiFloorGeometryError("core must be a StairCore")
    asset_id = _id_part(asset_id, "asset_id")
    if connector_index is None:
        connector_index = lower_floor_index
    connector_index = _floor_index(connector_index, "connector_index")
    lower_landing, upper_landing = _landing_rectangles(core, reverse=False)
    upper_floor_index = lower_floor_index + 1
    upper_floor_opening = stair_floor_opening(core, upper_floor_index)
    center_x, center_z = core.center
    yaw = core.yaw
    identifier = connector_id(lower_floor_index, connector_index)

    return {
        "id": identifier,
        "connectorType": "stairs",
        "assetId": asset_id,
        "lowerFloorId": floor_id(lower_floor_index),
        "upperFloorId": floor_id(upper_floor_index),
        "lowerRoomId": room_id(lower_room_number),
        "upperRoomId": room_id(upper_room_number),
        "position": {
            "x": center_x,
            "y": floor_base_y(lower_floor_index),
            "z": center_z,
        },
        "rotation": {"x": 0.0, "y": yaw, "z": 0.0},
        "assetContract": {
            "rise": core.contract.floor_pitch,
            "flightWidth": core.contract.flight_width,
            "flightRun": core.contract.flight_run,
            "reservedWidth": core.contract.core_width,
            "reservedLength": core.contract.core_length,
            "walkableRampCollider": True,
        },
        "landingPolygons": [
            {
                "id": floor_qualified_id(
                    "stair-landing",
                    lower_floor_index,
                    "connector-{}".format(connector_index),
                    "lower",
                ),
                "floorId": floor_id(lower_floor_index),
                "roomId": room_id(lower_room_number),
                "polygon": polygon_at_y(lower_landing, floor_base_y(lower_floor_index)),
            },
            {
                "id": floor_qualified_id(
                    "stair-landing",
                    upper_floor_index,
                    "connector-{}".format(connector_index),
                    "upper",
                ),
                "floorId": floor_id(upper_floor_index),
                "roomId": room_id(upper_room_number),
                "polygon": polygon_at_y(upper_landing, floor_base_y(upper_floor_index)),
            },
        ],
        "openingPolygons": [
            {
                "id": floor_qualified_id(
                    "stair-opening",
                    lower_floor_index,
                    "connector-{}".format(connector_index),
                    "ceiling",
                ),
                "floorId": floor_id(lower_floor_index),
                "surfaceType": "ceiling",
                "polygon": polygon_at_y(
                    core.bounds, floor_ceiling_y(lower_floor_index)
                ),
            },
            {
                "id": floor_qualified_id(
                    "stair-opening",
                    upper_floor_index,
                    "connector-{}".format(connector_index),
                    "floor",
                ),
                "floorId": floor_id(upper_floor_index),
                "surfaceType": "floor",
                "polygon": polygon_at_y(
                    upper_floor_opening, floor_base_y(upper_floor_index)
                ),
            },
        ],
    }


def build_vertical_connectors(
    core: StairCore,
    stair_host_room_numbers: Sequence[int],
    asset_id: str = DEFAULT_STAIR_ASSET_ID,
) -> List[Dict[str, Any]]:
    """Build the one or two connectors for a 2/3-floor house."""

    num_floors = _num_floors(len(stair_host_room_numbers))
    hosts = [
        _room_number(room_number, "stair host room number")
        for room_number in stair_host_room_numbers
    ]
    return [
        make_vertical_connector(
            lower_floor_index=connector_index,
            lower_room_number=hosts[connector_index],
            upper_room_number=hosts[connector_index + 1],
            core=core,
            asset_id=asset_id,
            connector_index=connector_index,
        )
        for connector_index in range(num_floors - 1)
    ]


def build_schema2_structure(
    floor_room_ids: Sequence[Iterable[int]],
    stair_host_room_ids: Sequence[int],
    shared_boundary: Any,
    stair_asset_id: str = DEFAULT_STAIR_ASSET_ID,
    stair_margin: float = 0.0,
    preferred_axis: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose deterministic schema-2 structural metadata for an orchestrator.

    Room meshes are deliberately left empty here: the generation pipeline can
    create each room's physical pieces with :func:`make_room_surface_metadata`
    and pass them to :func:`make_floor_record` after floorplan generation.
    """

    num_floors = _num_floors(len(floor_room_ids))
    if len(stair_host_room_ids) != num_floors:
        raise MultiFloorGeometryError(
            "one stair_host_room_id is required for each floor"
        )
    room_maps = make_global_room_id_maps(floor_room_ids)
    global_hosts = []
    for floor_index, input_host in enumerate(stair_host_room_ids):
        if input_host not in room_maps[floor_index]:
            raise MultiFloorGeometryError(
                "stair host {} is not on floor {}".format(input_host, floor_index)
            )
        global_hosts.append(room_maps[floor_index][input_host])

    core = place_stair_core(
        shared_boundary=shared_boundary,
        margin=stair_margin,
        preferred_axis=preferred_axis,
    )
    return {
        "schema": MULTI_FLOOR_SCHEMA,
        "floors": [
            make_floor_record(floor_index=index, room_id_map=room_maps[index])
            for index in range(num_floors)
        ],
        "verticalConnectors": build_vertical_connectors(
            core=core,
            stair_host_room_numbers=global_hosts,
            asset_id=stair_asset_id,
        ),
        "stairCoreReservations": stair_core_reservations(
            core=core, num_floors=num_floors
        ),
    }


__all__ = [
    "CLEAR_HEIGHT",
    "DEFAULT_STAIR_ASSET_ID",
    "DEFAULT_STAIR_GEOMETRY",
    "FLOOR_PITCH",
    "InvalidOpening",
    "InvalidSharedBoundary",
    "MAX_MULTI_FLOORS",
    "MIN_MULTI_FLOORS",
    "MULTI_FLOOR_SCHEMA",
    "MultiFloorGeometryError",
    "Rectangle",
    "SLAB_THICKNESS",
    "STAIR_CORE_LENGTH",
    "STAIR_CORE_WIDTH",
    "STAIR_FLIGHT_RUN",
    "STAIR_FLIGHT_WIDTH",
    "StairCore",
    "StairCoreDoesNotFit",
    "StairGeometryContract",
    "as_rectangle",
    "build_schema2_structure",
    "build_vertical_connectors",
    "connector_id",
    "decompose_rectangle_around_opening",
    "floor_base_y",
    "floor_ceiling_y",
    "floor_id",
    "floor_qualified_id",
    "make_floor_record",
    "make_global_room_id_maps",
    "make_room_surface_metadata",
    "make_surface_pieces",
    "make_vertical_connector",
    "place_shared_stair_core",
    "place_stair_core",
    "polygon_at_y",
    "rectangle_polygon",
    "room_id",
    "stair_core_reservations",
    "stair_floor_opening",
]
