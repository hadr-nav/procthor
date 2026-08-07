"""Schema-2 orchestration for multi-floor ProcTHOR houses.

The legacy :class:`HouseGenerator` path remains in ``generation.__init__``.
This module is imported lazily only for a two- or three-floor request and
coordinates existing generation functions one horizontal floor at a time.
"""

import copy
import itertools
import logging
import math
import random
from collections import defaultdict
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from ai2thor.controller import Controller
from attrs import define
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from procthor.constants import (
    MULTI_FLOOR_CLEAR_HEIGHT,
    MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS,
    MULTI_FLOOR_SCHEMA,
    MULTI_FLOOR_SLAB_THICKNESS,
    MULTI_FLOOR_STAIR_ASSET_ID,
    OUTDOOR_ROOM_ID,
    PROCTHOR_INITIALIZATION,
    STAIR_LANDING_EGRESS_DEPTH,
)
from procthor.utils.types import (
    InvalidFloorplan,
    InvalidMultiFloorPlan,
    LeafRoom,
    MetaRoom,
    MultiFloorCompatibilityError,
    SamplingVars,
)

from .generation import (
    consolidate_walls,
    find_walls,
    get_floor_polygons,
    get_xz_poly_map,
    scale_boundary_groups,
)
from .house import House, NextSamplingStage, PartialHouse
from .layer import assign_layer_to_rooms
from .materials import randomize_wall_and_floor_materials
from .multifloor import (
    DEFAULT_STAIR_GEOMETRY,
    MultiFloorGeometryError,
    Rectangle,
    StairCore,
    StairCoreDoesNotFit,
    build_vertical_connectors,
    floor_base_y,
    floor_ceiling_y,
    floor_id,
    floor_qualified_id,
    make_floor_record,
    make_global_room_id_maps,
    room_id,
    stair_core_reservations,
    stair_floor_opening,
)
from .multifloor_specs import HouseSpec
from .room_specs import RoomSpec


_CORE_CLEARANCE = 1e-4
_STAIR_OBJECT_CLEARANCE = 0.8
_GEOMETRY_EPSILON = 1e-8
# AI2-THOR's schema-2 rectangle validator treats spans up to 1 mm as zero.
_MIN_SURFACE_SPAN = 1e-3


@define
class MultiFloorPartialHouse:
    """Generation context spanning all floor-local ``PartialHouse`` objects."""

    house_spec: HouseSpec
    partial_houses: List[PartialHouse]
    room_id_maps: List[Dict[int, int]]
    stair_core: StairCore
    stair_host_room_ids: List[int]
    interior_boundary: Any
    next_sampling_stage: NextSamplingStage = NextSamplingStage.DOORS


def _action_return(event: Any) -> Any:
    metadata = getattr(event, "metadata", None)
    if metadata is None and isinstance(event, Mapping):
        metadata = event.get("metadata", event)
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("lastActionSuccess") is False:
        return None
    return metadata.get("actionReturn")


def _schema_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in (
            "schemas",
            "supportedSchemas",
            "supportedHouseSchemas",
            "houseSchemas",
        ):
            if key in value:
                return _schema_values(value[key])
        return []
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return []


def ensure_schema2_controller(controller: Any) -> List[str]:
    """Fail fast unless ``controller`` advertises procedural-house schema 2.

    This action must be the first generation-related controller action.  The
    caller intentionally invokes it before choosing an implicit seed, sampling
    a ``HouseSpec``, or sampling ``SamplingVars``.
    """

    if controller is None or not callable(getattr(controller, "step", None)):
        raise MultiFloorCompatibilityError(
            "Multi-floor generation requires an AI2-THOR controller."
        )
    try:
        event = controller.step(action="GetSupportedHouseSchemas", renderImage=False)
    except Exception as error:
        raise MultiFloorCompatibilityError(
            "The active AI2-THOR build does not expose "
            "GetSupportedHouseSchemas; schema 2.0.0 is required."
        ) from error

    supported = _schema_values(_action_return(event))
    if MULTI_FLOOR_SCHEMA not in supported:
        raise MultiFloorCompatibilityError(
            "The active AI2-THOR build does not advertise procedural-house "
            "schema {} (reported: {}).".format(MULTI_FLOOR_SCHEMA, supported)
        )
    return supported


def _clone_room_node(node: Any, room_id_map: Mapping[int, int]) -> Any:
    if isinstance(node, LeafRoom):
        return LeafRoom(
            room_id=room_id_map[node.room_id],
            ratio=node.ratio,
            room_type=node.room_type,
            avoid_doors_from_metarooms=node.avoid_doors_from_metarooms,
        )
    if isinstance(node, MetaRoom):
        return MetaRoom(
            ratio=node.ratio,
            children=[_clone_room_node(child, room_id_map) for child in node.children],
            room_type=node.room_type,
        )
    raise TypeError("RoomSpec nodes must be LeafRoom or MetaRoom instances.")


def _remap_room_spec(room_spec: RoomSpec, room_id_map: Mapping[int, int]) -> RoomSpec:
    """Clone one RoomSpec with globally allocated leaf ids."""

    return RoomSpec(
        room_spec_id=room_spec.room_spec_id,
        sampling_weight=room_spec.sampling_weight,
        spec=[_clone_room_node(node, room_id_map) for node in room_spec.spec],
        dims=room_spec.dims,
    )


def _room_polygons(structures: Sequence[Any]) -> List[Dict[int, Polygon]]:
    result = []
    for structure in structures:
        by_schema_id = get_floor_polygons(xz_poly_map=structure.xz_poly_map)
        floor_polygons = {}
        for schema_room_id, polygon in by_schema_id.items():
            room_number = int(schema_room_id.rsplit("|", 1)[1])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or not isinstance(polygon, Polygon):
                raise MultiFloorGeometryError(
                    "room {} has an invalid floor polygon".format(room_number)
                )
            floor_polygons[room_number] = polygon
        result.append(floor_polygons)
    return result


def _host_candidates(
    house_spec: HouseSpec,
    remapped_room_specs: Sequence[RoomSpec],
    room_id_maps: Sequence[Mapping[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Return ``(preference, global id)`` candidates for every floor."""

    result = []
    for floor_index, (floor_spec, remapped_spec, id_map) in enumerate(
        zip(house_spec.floors, remapped_room_specs, room_id_maps)
    ):
        if floor_spec.stair_host_room_id is not None:
            result.append([(0, id_map[floor_spec.stair_host_room_id])])
            continue

        candidates = []
        for preference, room_type in enumerate(("LivingRoom", "Bedroom")):
            candidates.extend(
                (preference, room_number)
                for room_number, candidate_type in sorted(
                    remapped_spec.room_type_map.items()
                )
                if candidate_type == room_type
            )
        if not candidates:
            raise StairCoreDoesNotFit(
                "floor {} has no LivingRoom or Bedroom stair host".format(floor_index)
            )
        result.append(candidates)
    return result


def _geometry_coordinates(geometry: Any) -> Tuple[List[float], List[float]]:
    xs = []
    zs = []

    def add_polygon(polygon: Polygon) -> None:
        rings = [polygon.exterior]
        rings.extend(polygon.interiors)
        for ring in rings:
            for x, z in ring.coords:
                xs.append(float(x))
                zs.append(float(z))

    if isinstance(geometry, Polygon):
        add_polygon(geometry)
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                add_polygon(part)
    return xs, zs


def _axis_order(geometry: Any) -> List[str]:
    min_x, min_z, max_x, max_z = geometry.bounds
    if max_x - min_x > max_z - min_z:
        return ["x", "z"]
    return ["z", "x"]


def _landing_egress_polygons(candidate: Any, long_axis: str) -> Tuple[Any, Any]:
    """Return the three required host-room aprons at each stair landing."""

    min_x, min_z, max_x, max_z = candidate.bounds
    landing_depth = DEFAULT_STAIR_GEOMETRY.landing_depth
    egress = STAIR_LANDING_EGRESS_DEPTH
    if long_axis == "z":
        return (
            (
                box(min_x, min_z - egress, max_x, min_z),
                box(min_x - egress, min_z, min_x, min_z + landing_depth),
                box(max_x, min_z, max_x + egress, min_z + landing_depth),
            ),
            (
                box(min_x, max_z, max_x, max_z + egress),
                box(min_x - egress, max_z - landing_depth, min_x, max_z),
                box(max_x, max_z - landing_depth, max_x + egress, max_z),
            ),
        )

    return (
        (
            box(min_x - egress, min_z, min_x, max_z),
            box(min_x, min_z - egress, min_x + landing_depth, min_z),
            box(min_x, max_z, min_x + landing_depth, max_z + egress),
        ),
        (
            box(max_x, min_z, max_x + egress, max_z),
            box(max_x - landing_depth, min_z - egress, max_x, min_z),
            box(max_x - landing_depth, max_z, max_x, max_z + egress),
        ),
    )


def _has_required_landing_egress(
    lower_geometries: Sequence[Any],
    upper_geometries: Sequence[Any],
    candidate: Any,
    long_axis: str,
) -> bool:
    lower_egress, upper_egress = _landing_egress_polygons(candidate, long_axis)
    return all(
        geometry.covers(apron)
        for geometry in lower_geometries
        for apron in lower_egress
    ) and all(
        geometry.covers(apron)
        for geometry in upper_geometries
        for apron in upper_egress
    )


def _fit_core_in_geometry(
    geometry: Any,
    lower_egress_geometries: Sequence[Any] = (),
    upper_egress_geometries: Sequence[Any] = (),
) -> Optional[StairCore]:
    """Find a deterministic axis-aligned core contained by ``geometry``."""

    if geometry.is_empty:
        return None
    safe_geometry = geometry.buffer(-_CORE_CLEARANCE, join_style=2)
    if safe_geometry.is_empty:
        return None
    source_xs, source_zs = _geometry_coordinates(safe_geometry)
    if not source_xs or not source_zs:
        return None

    centroid = safe_geometry.centroid
    min_x, min_z, max_x, max_z = safe_geometry.bounds
    for long_axis in _axis_order(safe_geometry):
        size_x = (
            DEFAULT_STAIR_GEOMETRY.core_length
            if long_axis == "x"
            else DEFAULT_STAIR_GEOMETRY.core_width
        )
        size_z = (
            DEFAULT_STAIR_GEOMETRY.core_width
            if long_axis == "x"
            else DEFAULT_STAIR_GEOMETRY.core_length
        )
        if max_x - min_x < size_x or max_z - min_z < size_z:
            continue

        lower_xs = {
            min_x,
            max_x - size_x,
            centroid.x - size_x / 2,
        }
        lower_zs = {
            min_z,
            max_z - size_z,
            centroid.y - size_z / 2,
        }
        for x in source_xs:
            lower_xs.update((x, x - size_x))
        for z in source_zs:
            lower_zs.update((z, z - size_z))

        candidates = []
        for lower_x, lower_z in itertools.product(lower_xs, lower_zs):
            candidate = box(lower_x, lower_z, lower_x + size_x, lower_z + size_z)
            if not safe_geometry.covers(candidate):
                continue
            lower_hosts = lower_egress_geometries or (safe_geometry,)
            upper_hosts = upper_egress_geometries or (safe_geometry,)
            if not _has_required_landing_egress(
                lower_hosts,
                upper_hosts,
                candidate,
                long_axis,
            ):
                continue
            object_clearance = candidate.buffer(
                _STAIR_OBJECT_CLEARANCE,
                join_style=2,
            )
            if not all(
                host_geometry.covers(object_clearance)
                for host_geometry in tuple(lower_hosts) + tuple(upper_hosts)
            ):
                continue
            center = candidate.centroid
            candidates.append(
                (
                    (center.x - centroid.x) ** 2 + (center.y - centroid.y) ** 2,
                    lower_x,
                    lower_z,
                    candidate,
                )
            )
        if candidates:
            _, _, _, candidate = min(candidates, key=lambda item: item[:3])
            bounds = candidate.bounds
            return StairCore(
                bounds=Rectangle(bounds[0], bounds[1], bounds[2], bounds[3]),
                long_axis=long_axis,
                yaw=90.0 if long_axis == "x" else 0.0,
            )
    return None


def _locate_shared_stair_core(
    house_spec: HouseSpec,
    remapped_room_specs: Sequence[RoomSpec],
    room_id_maps: Sequence[Mapping[int, int]],
    structures: Sequence[Any],
    door_clearance_polygons: Optional[Sequence[Mapping[int, Sequence[Polygon]]]] = None,
) -> Tuple[StairCore, List[int]]:
    if door_clearance_polygons is not None and (
        len(door_clearance_polygons) != len(structures)
        or any(
            not isinstance(floor_clearances, Mapping)
            for floor_clearances in door_clearance_polygons
        )
    ):
        raise MultiFloorGeometryError(
            "door_clearance_polygons must contain one mapping per floor"
        )

    floor_polygons = _room_polygons(structures)
    candidates_per_floor = _host_candidates(
        house_spec=house_spec,
        remapped_room_specs=remapped_room_specs,
        room_id_maps=room_id_maps,
    )
    combinations = list(itertools.product(*candidates_per_floor))
    combinations.sort(
        key=lambda combination: (
            sum(preference for preference, _ in combination),
            tuple(preference for preference, _ in combination),
            tuple(room_number for _, room_number in combination),
        )
    )

    for combination in combinations:
        host_ids = [room_number for _, room_number in combination]
        host_geometries = []
        for floor_index, host_id in enumerate(host_ids):
            floor_geometry = floor_polygons[floor_index][host_id]
            if door_clearance_polygons is not None:
                for clearance in door_clearance_polygons[floor_index].get(host_id, ()):
                    floor_geometry = floor_geometry.difference(clearance)
            host_geometries.append(floor_geometry)

        shared = host_geometries[0]
        for floor_geometry in host_geometries[1:]:
            shared = shared.intersection(floor_geometry)
            if shared.is_empty:
                break
        if shared.is_empty:
            continue

        core = _fit_core_in_geometry(
            shared,
            lower_egress_geometries=tuple(host_geometries[:-1]),
            upper_egress_geometries=tuple(host_geometries[1:]),
        )
        if core is not None:
            return core, host_ids
    if door_clearance_polygons is not None:
        raise StairCoreDoesNotFit(
            "no shared doorway-safe 1.2 x 6.5 m stair core fits the selected "
            "host rooms"
        )
    raise StairCoreDoesNotFit(
        "no shared 1.2 x 6.5 m stair core fits the selected host rooms"
    )


def _grid_region_is_connected(grid: np.ndarray, room_number: int) -> bool:
    coordinates = np.argwhere(grid == room_number)
    if not len(coordinates):
        return False
    start = tuple(int(value) for value in coordinates[0])
    visited = {start}
    frontier = [start]
    height, width = grid.shape
    while frontier:
        row, column = frontier.pop()
        for candidate in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            candidate_row, candidate_column = candidate
            if (
                0 <= candidate_row < height
                and 0 <= candidate_column < width
                and candidate not in visited
                and grid[candidate] == room_number
            ):
                visited.add(candidate)
                frontier.append(candidate)
    return len(visited) == len(coordinates)


def _validate_reserved_stair_core(
    core: StairCore,
    host_ids: Sequence[int],
    structures: Sequence[Any],
    door_clearance_polygons: Optional[Sequence[Mapping[int, Sequence[Polygon]]]] = None,
) -> Tuple[StairCore, List[int]]:
    """Prove that the fixed structural core remains clear after door sampling."""

    if len(host_ids) != len(structures):
        raise MultiFloorGeometryError(
            "stair host ids must contain one room id per floor"
        )
    if door_clearance_polygons is not None and len(door_clearance_polygons) != len(
        structures
    ):
        raise MultiFloorGeometryError(
            "door_clearance_polygons must contain one mapping per floor"
        )

    floor_polygons = _room_polygons(structures)
    host_geometries = []
    for floor_index, host_id in enumerate(host_ids):
        geometry = floor_polygons[floor_index][host_id]
        if door_clearance_polygons is not None:
            for clearance in door_clearance_polygons[floor_index].get(host_id, ()):
                geometry = geometry.difference(clearance)
        host_geometries.append(geometry)

    candidate = box(*core.bounds.as_tuple())
    lower_hosts = tuple(host_geometries[:-1])
    upper_hosts = tuple(host_geometries[1:])
    object_clearance = candidate.buffer(_STAIR_OBJECT_CLEARANCE, join_style=2)
    if (
        not all(geometry.covers(candidate) for geometry in host_geometries)
        or not _has_required_landing_egress(
            lower_hosts,
            upper_hosts,
            candidate,
            core.long_axis,
        )
        or not all(
            geometry.covers(object_clearance) for geometry in host_geometries
        )
    ):
        raise StairCoreDoesNotFit(
            "the reserved stair core or one of its three-sided landing "
            "clearances intersects a room or doorway boundary"
        )
    return core, list(host_ids)


def _rebuild_structure_from_grid(
    structure: Any,
    grid: np.ndarray,
    room_ids: Set[int],
    interior_boundary_scale: float,
) -> Any:
    """Rebuild exact wall and polygon records after a constrained partition."""

    rebuilt = copy.deepcopy(structure)
    floorplan = np.pad(
        grid,
        pad_width=1,
        mode="constant",
        constant_values=OUTDOOR_ROOM_ID,
    )
    rowcol_walls = find_walls(floorplan=floorplan)
    boundary_groups = consolidate_walls(walls=rowcol_walls)
    boundary_groups = scale_boundary_groups(
        boundary_groups=boundary_groups,
        scale=interior_boundary_scale,
    )
    rebuilt.floorplan = floorplan
    rebuilt.rowcol_walls = rowcol_walls
    rebuilt.boundary_groups = boundary_groups
    rebuilt.xz_poly_map = get_xz_poly_map(
        boundary_groups=boundary_groups,
        room_ids=room_ids,
    )
    return rebuilt


def _reserve_shared_stair_host_region(
    house_spec: HouseSpec,
    remapped_room_specs: Sequence[RoomSpec],
    room_id_maps: Sequence[Mapping[int, int]],
    structures: Sequence[Any],
    interior_boundary_scale: float,
) -> Tuple[List[Any], StairCore, List[int]]:
    """Constrain every floor partition around one shared stair-host region.

    The ordinary floorplan sampler partitions each floor independently.  This
    helper makes the stair volume a first-class structural constraint: it
    enumerates grid-aligned reservations large enough for the core and its
    furnishing clearance, assigns the same reservation to each selected host,
    and accepts only assignments that leave every room four-connected.
    """

    if not structures:
        raise StairCoreDoesNotFit("multi-floor generation returned no structures")
    boundary = np.asarray(structures[0].interior_boundary)
    if boundary.ndim != 2:
        raise MultiFloorGeometryError("interior boundary must be a 2D grid")
    if not math.isfinite(interior_boundary_scale) or interior_boundary_scale <= 0:
        raise MultiFloorGeometryError(
            "interior_boundary_scale must be a positive finite number"
        )

    floor_grids = []
    for structure in structures:
        grid = np.asarray(structure.floorplan)[1:-1, 1:-1]
        if grid.shape != boundary.shape:
            raise MultiFloorGeometryError(
                "all floorplans must match the shared interior boundary"
            )
        floor_grids.append(grid)

    candidates_per_floor = _host_candidates(
        house_spec=house_spec,
        remapped_room_specs=remapped_room_specs,
        room_id_maps=room_id_maps,
    )
    combinations = list(itertools.product(*candidates_per_floor))
    combinations.sort(
        key=lambda combination: (
            sum(preference for preference, _ in combination),
            tuple(preference for preference, _ in combination),
            tuple(room_number for _, room_number in combination),
        )
    )

    reserved_short = DEFAULT_STAIR_GEOMETRY.core_width + 2 * _STAIR_OBJECT_CLEARANCE
    reserved_long = DEFAULT_STAIR_GEOMETRY.core_length + 2 * _STAIR_OBJECT_CLEARANCE
    short_cells = int(math.ceil(reserved_short / interior_boundary_scale))
    long_cells = int(math.ceil(reserved_long / interior_boundary_scale))
    height, width = boundary.shape
    boundary_center = ((height - 1) / 2.0, (width - 1) / 2.0)
    valid = []

    for long_axis, reservation_width, reservation_height in (
        # ProcTHOR's floorplan row coordinate becomes world x, while its
        # column coordinate becomes world z during wall reconstruction.
        ("z", long_cells, short_cells),
        ("x", short_cells, long_cells),
    ):
        if reservation_width > width or reservation_height > height:
            continue
        for row in range(height - reservation_height + 1):
            for column in range(width - reservation_width + 1):
                row_slice = slice(row, row + reservation_height)
                column_slice = slice(column, column + reservation_width)
                if np.any(boundary[row_slice, column_slice] == OUTDOOR_ROOM_ID):
                    continue
                for combination in combinations:
                    host_ids = [room_number for _, room_number in combination]
                    rebuilt_structures = []
                    reassigned_cells = 0
                    for grid, host_id, room_spec, structure in zip(
                        floor_grids,
                        host_ids,
                        remapped_room_specs,
                        structures,
                    ):
                        constrained = grid.copy()
                        reassigned_cells += int(
                            np.count_nonzero(
                                constrained[row_slice, column_slice] != host_id
                            )
                        )
                        constrained[row_slice, column_slice] = host_id
                        room_ids = set(room_spec.room_type_map)
                        if not all(
                            _grid_region_is_connected(constrained, room_number)
                            for room_number in room_ids
                        ):
                            break
                        try:
                            rebuilt = _rebuild_structure_from_grid(
                                structure=structure,
                                grid=constrained,
                                room_ids=room_ids,
                                interior_boundary_scale=interior_boundary_scale,
                            )
                        except Exception as error:
                            if not str(error).startswith("No connecting wall for"):
                                raise
                            break
                        try:
                            _room_polygons([rebuilt])
                        except MultiFloorGeometryError:
                            break
                        rebuilt_structures.append(rebuilt)
                    if len(rebuilt_structures) != len(structures):
                        continue
                    reservation_center = (
                        row + (reservation_height - 1) / 2.0,
                        column + (reservation_width - 1) / 2.0,
                    )
                    center_x = (
                        row + reservation_height / 2.0
                    ) * interior_boundary_scale
                    center_z = (
                        column + reservation_width / 2.0
                    ) * interior_boundary_scale
                    half_x = (
                        DEFAULT_STAIR_GEOMETRY.core_length / 2.0
                        if long_axis == "x"
                        else DEFAULT_STAIR_GEOMETRY.core_width / 2.0
                    )
                    half_z = (
                        DEFAULT_STAIR_GEOMETRY.core_width / 2.0
                        if long_axis == "x"
                        else DEFAULT_STAIR_GEOMETRY.core_length / 2.0
                    )
                    core = StairCore(
                        bounds=Rectangle(
                            center_x - half_x,
                            center_z - half_z,
                            center_x + half_x,
                            center_z + half_z,
                        ),
                        long_axis=long_axis,
                        yaw=90.0 if long_axis == "x" else 0.0,
                    )
                    try:
                        _validate_reserved_stair_core(
                            core=core,
                            host_ids=host_ids,
                            structures=rebuilt_structures,
                        )
                    except StairCoreDoesNotFit:
                        continue
                    center_distance = (
                        (reservation_center[0] - boundary_center[0]) ** 2
                        + (reservation_center[1] - boundary_center[1]) ** 2
                    )
                    valid.append(
                        (
                            sum(preference for preference, _ in combination),
                            reassigned_cells,
                            center_distance,
                            0 if long_axis == "z" else 1,
                            row,
                            column,
                            rebuilt_structures,
                            core,
                            host_ids,
                        )
                    )

    if not valid:
        raise StairCoreDoesNotFit(
            "no connected shared stair-host reservation fits every floor partition"
        )
    selected = min(valid, key=lambda candidate: candidate[:6])
    return selected[6], selected[7], selected[8]


def _shared_dims_value(house_spec: HouseSpec) -> Optional[Tuple[int, int]]:
    dims = house_spec.dims
    if dims is None:
        return None
    value = dims() if callable(dims) else dims
    try:
        x_size, z_size = value
    except (TypeError, ValueError):
        raise InvalidMultiFloorPlan(
            "HouseSpec.dims must resolve to an (x_size, z_size) pair."
        )
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or dimension <= 0
        for dimension in (x_size, z_size)
    ):
        raise InvalidMultiFloorPlan(
            "HouseSpec.dims must resolve to two positive integers."
        )
    return (int(x_size), int(z_size))


def sample_complete_multifloor_structure(
    generator: Any,
    house_spec: HouseSpec,
    sampling_vars: SamplingVars,
    max_attempts: int = MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS,
) -> MultiFloorPartialHouse:
    """Sample every floor as one bounded structural rejection attempt."""

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, Integral):
        raise ValueError("max_attempts must be a positive integer.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")

    local_room_ids = [
        sorted(floor.room_spec.room_type_map) for floor in house_spec.floors
    ]
    room_id_maps = make_global_room_id_maps(local_room_ids)
    remapped_room_specs = [
        _remap_room_spec(floor.room_spec, room_id_map)
        for floor, room_id_map in zip(house_spec.floors, room_id_maps)
    ]
    # When no boundary is supplied, let the most populated floor establish the
    # shared footprint so later floors are not forced into a smaller layout.
    structure_order = sorted(
        range(len(remapped_room_specs)),
        key=lambda index: (-len(remapped_room_specs[index].room_type_map), index),
    )
    last_error = None

    for _ in range(int(max_attempts)):
        try:
            shared_dims = _shared_dims_value(house_spec)
            if shared_dims is not None:
                for room_spec in remapped_room_specs:
                    room_spec.dims = lambda value=shared_dims: value

            shared_boundary = copy.deepcopy(generator.interior_boundary)
            structures = [None] * len(remapped_room_specs)
            for floor_index in structure_order:
                room_spec = remapped_room_specs[floor_index]
                structure = generator.generation_functions.sample_house_structure(
                    interior_boundary=copy.deepcopy(shared_boundary),
                    room_ids=set(room_spec.room_type_map),
                    room_spec=room_spec,
                    interior_boundary_scale=(sampling_vars.interior_boundary_scale),
                )
                if structure is None or not hasattr(structure, "xz_poly_map"):
                    raise InvalidFloorplan(
                        "sample_house_structure returned no HouseStructure"
                    )
                structure.ceiling_height = MULTI_FLOOR_CLEAR_HEIGHT
                if shared_boundary is not None and not np.array_equal(
                    np.asarray(structure.interior_boundary),
                    np.asarray(shared_boundary),
                ):
                    raise InvalidFloorplan(
                        "sample_house_structure did not preserve the shared boundary"
                    )
                structures[floor_index] = structure
                if shared_boundary is None:
                    shared_boundary = copy.deepcopy(structure.interior_boundary)

            structures, core, stair_host_room_ids = _reserve_shared_stair_host_region(
                house_spec=house_spec,
                remapped_room_specs=remapped_room_specs,
                room_id_maps=room_id_maps,
                structures=structures,
                interior_boundary_scale=sampling_vars.interior_boundary_scale,
            )
            partial_houses = [
                PartialHouse.from_structure_and_room_spec(
                    house_structure=structure,
                    room_spec=room_spec,
                )
                for structure, room_spec in zip(structures, remapped_room_specs)
            ]
            return MultiFloorPartialHouse(
                house_spec=house_spec,
                partial_houses=partial_houses,
                room_id_maps=[dict(mapping) for mapping in room_id_maps],
                stair_core=core,
                stair_host_room_ids=stair_host_room_ids,
                interior_boundary=shared_boundary,
            )
        except (InvalidFloorplan, MultiFloorGeometryError) as error:
            last_error = error

    raise InvalidMultiFloorPlan(
        "Failed to generate a valid aligned {}-floor structure in {} complete "
        "attempts: {}".format(len(house_spec.floors), max_attempts, last_error)
    ) from last_error


def _normalize_door_polygons(
    value: Any, room_ids: Iterable[int]
) -> Dict[int, List[Polygon]]:
    normalized = defaultdict(list)
    if isinstance(value, Mapping):
        for room_number, polygons in value.items():
            normalized[int(room_number)].extend(polygons or [])
    for room_number in room_ids:
        normalized[room_number]
    return normalized


def _is_exterior_opening(opening: Mapping[str, Any]) -> bool:
    return any(
        isinstance(opening.get(key), str) and "exterior" in opening.get(key)
        for key in ("wall0", "wall1")
    )


def _apply_house_skybox_policy(source: PartialHouse, target: PartialHouse) -> None:
    """Reuse the ground-floor skybox and point-light policy on another floor."""

    source_parameters = source.procedural_parameters
    target_parameters = target.procedural_parameters
    for key, value in source_parameters.items():
        if key.startswith("skybox"):
            target_parameters[key] = copy.deepcopy(value)

    source_point_light = next(
        (
            light
            for light in source_parameters.get("lights", [])
            if light.get("type") != "directional"
        ),
        None,
    )
    if source_point_light is None:
        return
    for target_light in target_parameters.get("lights", []):
        if target_light.get("type") == "directional":
            continue
        for key, value in source_point_light.items():
            if key not in {"id", "position"}:
                target_light[key] = copy.deepcopy(value)


def _run_floor_door_stage(
    generator: Any,
    partial_house: PartialHouse,
    floor_index: int,
) -> Dict[int, List[Polygon]]:
    gfs = generator.generation_functions
    full_boundary_groups = partial_house.house_structure.boundary_groups
    if floor_index == 0:
        door_boundary_groups = full_boundary_groups
    else:
        door_boundary_groups = {
            group: walls
            for group, walls in full_boundary_groups.items()
            if OUTDOOR_ROOM_ID not in group
        }

    partial_house.house_structure.boundary_groups = door_boundary_groups
    try:
        door_polygons = gfs.add_doors(
            partial_house=partial_house,
            controller=generator.controller,
            pt_db=generator.pt_db,
            split=generator.split,
        )
    finally:
        partial_house.house_structure.boundary_groups = full_boundary_groups
    if partial_house.doors is None:
        partial_house.doors = []
    if floor_index > 0:
        partial_house.doors = [
            door for door in partial_house.doors if not _is_exterior_opening(door)
        ]
    return _normalize_door_polygons(
        door_polygons, partial_house.room_spec.room_type_map
    )


def _run_floor_generation_stages(
    generator: Any,
    partial_house: PartialHouse,
    floor_index: int,
    stair_host_room_id: int,
    stair_core: StairCore,
    sampling_vars: SamplingVars,
    skybox_source: Optional[PartialHouse],
    door_polygons: Optional[Dict[int, List[Polygon]]] = None,
) -> None:
    gfs = generator.generation_functions
    full_boundary_groups = partial_house.house_structure.boundary_groups
    if door_polygons is None:
        door_polygons = _run_floor_door_stage(
            generator=generator,
            partial_house=partial_house,
            floor_index=floor_index,
        )
    randomize_wall_and_floor_materials(partial_house, pt_db=generator.pt_db)
    partial_house.advance_sampling_stage()

    floor_polygons = get_floor_polygons(
        xz_poly_map=partial_house.house_structure.xz_poly_map
    )
    gfs.add_lights(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        floor_polygons=floor_polygons,
        ceiling_height=MULTI_FLOOR_CLEAR_HEIGHT,
    )
    partial_house.advance_sampling_stage()

    if skybox_source is None:
        gfs.add_skybox(
            partial_house=partial_house,
            controller=generator.controller,
            pt_db=generator.pt_db,
            split=generator.split,
        )
    else:
        _apply_house_skybox_policy(skybox_source, partial_house)
    partial_house.advance_sampling_stage()

    gfs.add_exterior_walls(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        boundary_groups=full_boundary_groups,
    )
    partial_house.advance_sampling_stage()

    gfs.add_rooms(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        floor_polygons=floor_polygons,
        room_type_map=partial_house.room_spec.room_type_map,
        door_polygons=door_polygons,
    )
    if partial_house.rooms is None or stair_host_room_id not in partial_house.rooms:
        raise InvalidMultiFloorPlan(
            "add_rooms did not create stair host room {} on floor {}.".format(
                stair_host_room_id, floor_index
            )
        )
    # Keep furniture away from the front and both side approaches at every
    # landing. The runtime mesh exposes all three physical directions, so the
    # generated room must preserve them rather than choosing one entrance.
    stair_object_reservation = box(*stair_core.bounds.as_tuple()).buffer(
        _STAIR_OBJECT_CLEARANCE,
        join_style=2,
    )
    partial_house.rooms[stair_host_room_id].open_polygon.subtract(
        stair_object_reservation
    )
    partial_house.advance_sampling_stage()

    gfs.add_floor_objects(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        max_floor_objects=sampling_vars.max_floor_objects,
    )
    if partial_house.objects is None:
        partial_house.objects = []
    floor_objects = list(partial_house.objects)
    gfs.randomize_object_colors(objects=floor_objects, pt_db=generator.pt_db)
    gfs.randomize_object_states(objects=floor_objects, pt_db=generator.pt_db)
    partial_house.advance_sampling_stage()

    gfs.add_wall_objects(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        rooms=partial_house.rooms,
        boundary_groups=full_boundary_groups,
        room_type_map=partial_house.room_spec.room_type_map,
        ceiling_height=MULTI_FLOOR_CLEAR_HEIGHT,
    )
    if partial_house.windows is None:
        partial_house.windows = []
    wall_objects = list(partial_house.objects[len(floor_objects) :])
    gfs.randomize_object_colors(objects=wall_objects, pt_db=generator.pt_db)
    gfs.randomize_object_states(objects=wall_objects, pt_db=generator.pt_db)
    partial_house.advance_sampling_stage()

    gfs.add_small_objects(
        partial_house=partial_house,
        controller=generator.controller,
        pt_db=generator.pt_db,
        split=generator.split,
        rooms=partial_house.rooms,
    )
    small_objects = list(
        partial_house.objects[len(floor_objects) + len(wall_objects) :]
    )
    gfs.randomize_object_colors(objects=small_objects, pt_db=generator.pt_db)
    gfs.randomize_object_states(objects=small_objects, pt_db=generator.pt_db)
    partial_house.advance_sampling_stage()
    assign_layer_to_rooms(partial_house=partial_house)


# Opening ``assetPosition``/``holePolygon`` values (including their min/max
# vectors) are wall-local and must remain unchanged when the owning wall is
# lifted. These are the fields whose vectors are expressed in world space.
_WORLD_VECTOR_FIELDS = {
    "position",
    "polygon",
    "floorPolygon",
}


def _lift_world_vectors(
    value: Any,
    base_y: float,
    field_name: str = "",
    lifted_vectors: Optional[Set[int]] = None,
) -> None:
    if lifted_vectors is None:
        lifted_vectors = set()
    if isinstance(value, dict):
        if (
            field_name in _WORLD_VECTOR_FIELDS
            and "y" in value
            and isinstance(value["y"], (int, float))
            and not isinstance(value["y"], bool)
            and id(value) not in lifted_vectors
        ):
            lifted_vectors.add(id(value))
            value["y"] = float(value["y"]) + base_y
        for key, child in value.items():
            _lift_world_vectors(
                child,
                base_y=base_y,
                field_name=key,
                lifted_vectors=lifted_vectors,
            )
    elif isinstance(value, list):
        for child in value:
            _lift_world_vectors(
                child,
                base_y=base_y,
                field_name=field_name,
                lifted_vectors=lifted_vectors,
            )


def _qualify_id(identifier: Any, kind: str, floor_index: int) -> str:
    parts = str(identifier).split("|")
    if parts and parts[0] == kind:
        parts = parts[1:]
    parts = [part for part in parts if part]
    if not parts:
        parts = ["item"]
    return floor_qualified_id(kind, floor_index, *parts)


def _qualify_floor_ids(partial_house: PartialHouse, floor_index: int) -> None:
    owner_floor_id = floor_id(floor_index)
    wall_id_map = {}
    for wall in partial_house.walls or []:
        old_id = wall["id"]
        wall["id"] = _qualify_id(old_id, "wall", floor_index)
        wall["floorId"] = owner_floor_id
        wall_id_map[old_id] = wall["id"]

    for kind, openings in (
        ("door", partial_house.doors or []),
        ("window", partial_house.windows or []),
    ):
        for opening in openings:
            opening["id"] = _qualify_id(opening["id"], kind, floor_index)
            opening["floorId"] = owner_floor_id
            for wall_key in ("wall0", "wall1"):
                old_wall_id = opening.get(wall_key)
                if old_wall_id is not None:
                    opening[wall_key] = wall_id_map.get(
                        old_wall_id,
                        _qualify_id(old_wall_id, "wall", floor_index),
                    )

    def set_object_floor_ids(objects: Iterable[Dict[str, Any]]) -> None:
        for obj in objects:
            obj["floorId"] = owner_floor_id
            set_object_floor_ids(obj.get("children", []))

    set_object_floor_ids(partial_house.objects or [])
    for light in partial_house.procedural_parameters.get("lights", []):
        if light.get("type") != "directional":
            light["id"] = _qualify_id(light.get("id", "light"), "light", floor_index)
            light["floorId"] = owner_floor_id


def _merge_surface_rectangles(
    rectangles: Sequence[Polygon], axis: str
) -> List[Polygon]:
    """Merge cells that share a complete edge along one axis."""

    if axis not in {"x", "z"}:
        raise ValueError("surface rectangle merge axis must be 'x' or 'z'")
    grouped = defaultdict(list)
    for rectangle in rectangles:
        min_x, min_z, max_x, max_z = rectangle.bounds
        key = (min_z, max_z) if axis == "x" else (min_x, max_x)
        grouped[key].append((min_x, min_z, max_x, max_z))

    merged = []
    sort_index = 0 if axis == "x" else 1
    near_index = 2 if axis == "x" else 3
    far_index = 0 if axis == "x" else 1
    for bounds in sorted(grouped):
        ordered = sorted(grouped[bounds], key=lambda item: item[sort_index])
        current = list(ordered[0])
        for candidate in ordered[1:]:
            if abs(current[near_index] - candidate[far_index]) <= _GEOMETRY_EPSILON:
                current[near_index] = candidate[near_index]
            else:
                merged.append(box(*current))
                current = list(candidate)
        merged.append(box(*current))
    return merged


def _polygon_parts(
    geometry: Any, preferred_merge_axis: str = "x"
) -> List[Polygon]:
    if geometry.is_empty:
        return []
    xs, zs = _geometry_coordinates(geometry)
    xs = sorted(set(xs))
    zs = sorted(set(zs))
    if len(xs) < 2 or len(zs) < 2:
        return []

    parts = []
    for min_x, max_x in zip(xs, xs[1:]):
        for min_z, max_z in zip(zs, zs[1:]):
            if (max_x - min_x) * (max_z - min_z) <= _GEOMETRY_EPSILON:
                continue
            piece = box(min_x, min_z, max_x, max_z)
            if geometry.covers(piece):
                parts.append(piece)

    covered_area = sum(piece.area for piece in parts)
    if abs(covered_area - geometry.area) > max(_GEOMETRY_EPSILON, geometry.area * 1e-7):
        raise InvalidMultiFloorPlan(
            "Physical floor and ceiling surfaces must be orthogonal polygons."
        )
    merge_axes = (
        preferred_merge_axis,
        "z" if preferred_merge_axis == "x" else "x",
    )
    while True:
        previous_count = len(parts)
        for axis in merge_axes:
            parts = _merge_surface_rectangles(parts, axis)
        if len(parts) == previous_count:
            break
    parts.sort(key=lambda polygon: (polygon.bounds, polygon.area))
    return parts


def _polygon_vectors(polygon: Polygon, y: float) -> List[Dict[str, float]]:
    polygon = orient(polygon, sign=1.0)
    return [
        {"x": float(x), "y": float(y), "z": float(z)}
        for x, z in list(polygon.exterior.coords)[:-1]
    ]


def _is_serializable_surface_piece(polygon: Polygon) -> bool:
    """Return whether both rectangle spans survive engine validation."""

    min_x, min_z, max_x, max_z = polygon.bounds
    return max_x - min_x > _MIN_SURFACE_SPAN and max_z - min_z > _MIN_SURFACE_SPAN


def _surface_objects(
    geometry: Polygon,
    floor_index: int,
    room_number: int,
    surface_type: str,
    material: Optional[Mapping[str, Any]],
    opening: Optional[Polygon],
) -> List[Dict[str, Any]]:
    physical_geometry = geometry if opening is None else geometry.difference(opening)
    y = (
        floor_base_y(floor_index)
        if surface_type == "floor"
        else floor_ceiling_y(floor_index)
    )
    preferred_merge_axis = "x"
    if opening is not None:
        min_x, min_z, max_x, max_z = opening.bounds
        if max_x - min_x > max_z - min_z:
            preferred_merge_axis = "z"
    polygons = [
        polygon
        for polygon in _polygon_parts(
            physical_geometry,
            preferred_merge_axis=preferred_merge_axis,
        )
        if _is_serializable_surface_piece(polygon)
    ]
    surfaces = []
    for piece_index, polygon in enumerate(polygons):
        surface = {
            "id": floor_qualified_id(
                "{}-surface".format(surface_type),
                floor_index,
                "room-{}".format(room_number),
                "piece-{}".format(piece_index),
            ),
            "floorId": floor_id(floor_index),
            "roomId": room_id(room_number),
            "surfaceType": surface_type,
            "polygon": _polygon_vectors(polygon, y),
        }
        if surface_type == "floor":
            surface["slabThickness"] = MULTI_FLOOR_SLAB_THICKNESS
        if material is not None:
            surface["material"] = copy.deepcopy(dict(material))
        surfaces.append(surface)
    if not surfaces:
        raise InvalidMultiFloorPlan(
            "Stair opening removed the entire {} surface for room {}.".format(
                surface_type, room_number
            )
        )
    return surfaces


def _add_schema2_room_surfaces(
    partial_house: PartialHouse,
    floor_index: int,
    num_floors: int,
    stair_host_room_id: int,
    stair_core: StairCore,
) -> Dict[int, Dict[str, Any]]:
    polygons = _room_polygons([partial_house.house_structure])[0]
    room_dict_map = {
        int(room["id"].rsplit("|", 1)[1]): room for room in partial_house.room_types
    }
    ceiling_material = partial_house.procedural_parameters.get("ceilingMaterial")
    core_polygon = box(*stair_core.bounds.as_tuple())
    floor_opening_polygon = box(
        *stair_floor_opening(stair_core, floor_index).as_tuple()
    )
    metadata = {}
    for room_number, polygon in sorted(polygons.items()):
        room = room_dict_map[room_number]
        floor_opening = (
            floor_opening_polygon if room_number == stair_host_room_id else None
        )
        ceiling_opening = (
            core_polygon
            if floor_index < num_floors - 1 and room_number == stair_host_room_id
            else None
        )
        surface_metadata = {
            "floorId": floor_id(floor_index),
            "floorPolygon": _polygon_vectors(polygon, floor_base_y(floor_index)),
            "floorPolygons": _surface_objects(
                geometry=polygon,
                floor_index=floor_index,
                room_number=room_number,
                surface_type="floor",
                material=room.get("floorMaterial"),
                opening=floor_opening,
            ),
            "ceilings": _surface_objects(
                geometry=polygon,
                floor_index=floor_index,
                room_number=room_number,
                surface_type="ceiling",
                material=ceiling_material,
                opening=ceiling_opening,
            ),
        }
        room.update(copy.deepcopy(surface_metadata))
        metadata[room_number] = surface_metadata
    return metadata


def _merge_procedural_parameters(
    partial_houses: Sequence[PartialHouse],
) -> Dict[str, Any]:
    merged = copy.deepcopy(partial_houses[0].procedural_parameters)
    merged["lights"] = []
    directional_light = None
    point_lights = []
    for partial_house in partial_houses:
        for light in partial_house.procedural_parameters.get("lights", []):
            if light.get("type") == "directional":
                if directional_light is None:
                    directional_light = light
            else:
                point_lights.append(light)
    if directional_light is not None:
        merged["lights"].append(directional_light)
    merged["lights"].extend(point_lights)
    return merged


def _assemble_house(
    context: MultiFloorPartialHouse,
) -> House:
    num_floors = len(context.partial_houses)
    all_rooms = []
    all_doors = []
    all_windows = []
    all_objects = []
    all_walls = []
    procedural_rooms = {}
    room_floor_map = {}
    floor_records = []

    for floor_index, partial_house in enumerate(context.partial_houses):
        base_y = floor_base_y(floor_index)
        lifted_vectors: Set[int] = set()
        for world_geometry in (
            partial_house.room_types,
            partial_house.walls,
            partial_house.doors,
            partial_house.windows,
            partial_house.objects,
            partial_house.procedural_parameters,
        ):
            _lift_world_vectors(
                world_geometry,
                base_y,
                lifted_vectors=lifted_vectors,
            )
        _qualify_floor_ids(partial_house, floor_index)

        room_surfaces = _add_schema2_room_surfaces(
            partial_house=partial_house,
            floor_index=floor_index,
            num_floors=num_floors,
            stair_host_room_id=context.stair_host_room_ids[floor_index],
            stair_core=context.stair_core,
        )
        floor_records.append(
            make_floor_record(
                floor_index=floor_index,
                room_id_map=context.room_id_maps[floor_index],
                room_surfaces=room_surfaces,
            )
        )
        all_rooms.extend(partial_house.room_types or [])
        all_doors.extend(partial_house.doors or [])
        all_windows.extend(partial_house.windows or [])
        all_objects.extend(partial_house.objects or [])
        all_walls.extend(partial_house.walls or [])
        procedural_rooms.update(partial_house.rooms or {})
        for room_number in partial_house.room_spec.room_type_map:
            room_floor_map[room_number] = base_y

    data = {
        "schema": MULTI_FLOOR_SCHEMA,
        "rooms": all_rooms,
        "doors": all_doors,
        "windows": all_windows,
        "objects": all_objects,
        "walls": all_walls,
        "proceduralParameters": _merge_procedural_parameters(context.partial_houses),
        "floors": floor_records,
        "verticalConnectors": build_vertical_connectors(
            core=context.stair_core,
            stair_host_room_numbers=context.stair_host_room_ids,
            asset_id=MULTI_FLOOR_STAIR_ASSET_ID,
        ),
        "stairCoreReservations": stair_core_reservations(
            core=context.stair_core, num_floors=num_floors
        ),
    }
    return House(
        data=data,
        rooms=procedural_rooms,
        interior_boundary=context.interior_boundary,
        room_spec=context.partial_houses[0].room_spec,
        room_floor_map=room_floor_map,
        house_spec_id=context.house_spec.house_spec_id,
        floor_room_spec_ids=[
            floor.room_spec.room_spec_id for floor in context.house_spec.floors
        ],
    )


def _select_house_spec(generator: Any) -> HouseSpec:
    if generator.house_spec is not None:
        return copy.deepcopy(generator.house_spec)
    if generator.house_spec_sampler is None:
        raise InvalidMultiFloorPlan(
            "Multi-floor generation requires a HouseSpec or HouseSpecSampler."
        )
    house_spec = generator.house_spec_sampler.sample(
        num_floors=generator.num_floors, rng=random
    )
    if not isinstance(house_spec, HouseSpec):
        raise TypeError("house_spec_sampler.sample() must return a HouseSpec.")
    if (
        generator.num_floors is not None
        and house_spec.num_floors != generator.num_floors
    ):
        raise InvalidMultiFloorPlan(
            "HouseSpecSampler returned {} floors for a {}-floor request.".format(
                house_spec.num_floors, generator.num_floors
            )
        )
    return house_spec


def sample_multifloor_house(
    generator: Any,
    sampling_vars: Optional[SamplingVars] = None,
    return_partial_houses: bool = False,
) -> Tuple[House, Dict[NextSamplingStage, MultiFloorPartialHouse]]:
    """Generate and serialize one complete schema-2 multi-floor house."""

    if generator.controller is None:
        generator.controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)

    # Capability detection deliberately precedes every operation that samples
    # ProcTHOR randomness, including selection of an implicit generator seed.
    ensure_schema2_controller(generator.controller)
    if generator.seed is None:
        seed = random.randint(0, 2**15)
        logging.debug("Using seed %s", seed)
        generator.set_seed(seed)
    else:
        # The constructor has already initialized Python/NumPy randomness for
        # an explicit seed. Reset only the engine here; reseeding Python would
        # make repeated dataset attempts replay the identical failed house.
        generator.controller.step(
            action="SetRandomSeed",
            seed=generator.seed,
            renderImage=False,
        )

    house_spec = _select_house_spec(generator)
    sampling_vars = SamplingVars.sample() if sampling_vars is None else sampling_vars
    context: Optional[MultiFloorPartialHouse] = None
    door_clearance_polygons: List[Dict[int, List[Polygon]]] = []
    last_error: Optional[Exception] = None
    for _ in range(MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS):
        try:
            candidate_context = sample_complete_multifloor_structure(
                generator=generator,
                house_spec=house_spec,
                sampling_vars=sampling_vars,
                max_attempts=1,
            )
            candidate_door_polygons = [
                _run_floor_door_stage(
                    generator=generator,
                    partial_house=partial_house,
                    floor_index=floor_index,
                )
                for floor_index, partial_house in enumerate(
                    candidate_context.partial_houses
                )
            ]
            if candidate_context.partial_houses:
                core, stair_host_room_ids = _validate_reserved_stair_core(
                    core=candidate_context.stair_core,
                    host_ids=candidate_context.stair_host_room_ids,
                    structures=[
                        partial_house.house_structure
                        for partial_house in candidate_context.partial_houses
                    ],
                    door_clearance_polygons=candidate_door_polygons,
                )
                candidate_context.stair_core = core
                candidate_context.stair_host_room_ids = stair_host_room_ids
            context = candidate_context
            door_clearance_polygons = candidate_door_polygons
            break
        except (
            InvalidFloorplan,
            InvalidMultiFloorPlan,
            MultiFloorGeometryError,
        ) as error:
            last_error = error
            if getattr(generator, "house_spec", None) is None:
                house_spec = _select_house_spec(generator)
    if context is None:
        raise InvalidMultiFloorPlan(
            "Failed to generate a doorway-safe aligned {}-floor house in {} "
            "complete attempts: {}".format(
                len(house_spec.floors),
                MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS,
                last_error,
            )
        ) from last_error

    contexts = {}
    if return_partial_houses:
        contexts[NextSamplingStage.DOORS] = copy.deepcopy(context)

    for floor_index, (partial_house, floor_door_polygons) in enumerate(
        zip(context.partial_houses, door_clearance_polygons)
    ):
        _run_floor_generation_stages(
            generator=generator,
            partial_house=partial_house,
            floor_index=floor_index,
            stair_host_room_id=context.stair_host_room_ids[floor_index],
            stair_core=context.stair_core,
            sampling_vars=sampling_vars,
            skybox_source=(None if floor_index == 0 else context.partial_houses[0]),
            door_polygons=floor_door_polygons,
        )
    context.next_sampling_stage = NextSamplingStage.COMPLETE
    house = _assemble_house(context)
    contexts[NextSamplingStage.COMPLETE] = context
    return house, contexts


__all__ = [
    "MultiFloorPartialHouse",
    "ensure_schema2_controller",
    "sample_complete_multifloor_structure",
    "sample_multifloor_house",
]
