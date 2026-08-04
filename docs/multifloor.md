# Multi-floor generation

ProcTHOR can generate two- and three-floor residential scenes with one aligned
footprint and a physically walkable straight stair flight between each pair of
adjacent floors. Multi-floor houses use procedural-house schema `2.0.0` and
require the matching AI2-THOR integration described below.

## Generate a built-in residential house

Pass `num_floors=2` or `num_floors=3` to `HouseGenerator`. The built-in sampler
puts a kitchen and living room on the ground floor, optionally adds a ground
bathroom, and gives each upper floor a living-room stair host, bathroom, and one
or two bedrooms.

```python
from ai2thor.controller import Controller
from procthor.generation import HouseGenerator

controller = Controller(
    local_executable_path="/path/to/patched/ai2thor-build",
    scene="Procedural",
    quality="Low",
)

two_floor_generator = HouseGenerator(
    split="train",
    seed=1234,
    num_floors=2,
    controller=controller,
)
two_floor_house, _ = two_floor_generator.sample()

three_floor_generator = HouseGenerator(
    split="train",
    seed=5678,
    num_floors=3,
    controller=controller,
)
three_floor_house, _ = three_floor_generator.sample()
```

The supplied controller must use the patched build below and advertise schema
`2.0.0`. Before consuming generation randomness, ProcTHOR calls
`GetSupportedHouseSchemas`; if schema 2 is not reported, generation raises
`MultiFloorCompatibilityError` immediately.

## Supply an explicit house specification

`RoomSpec` remains the specification for one horizontal level. `FloorSpec`
wraps it with an optional floor-local stair-host room ID, and `HouseSpec`
combines exactly two or three floors under one shared footprint. `dims` may be
a fixed pair or a zero-argument callable. When it is omitted, the most populated
floor establishes the boundary on each attempt and every other floor reuses it.

```python
from procthor.generation import FloorSpec, HouseGenerator, HouseSpec
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import LeafRoom

ground = RoomSpec(
    room_spec_id="custom-ground",
    sampling_weight=1,
    spec=[
        LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
        LeafRoom(room_id=3, ratio=4, room_type="LivingRoom"),
    ],
)
upper = RoomSpec(
    room_spec_id="custom-upper",
    sampling_weight=1,
    spec=[
        LeafRoom(room_id=2, ratio=3, room_type="LivingRoom"),
        LeafRoom(room_id=3, ratio=2, room_type="Bedroom"),
        LeafRoom(room_id=4, ratio=1, room_type="Bathroom"),
    ],
)

house_spec = HouseSpec(
    house_spec_id="custom-two-floor",
    dims=(14, 10),
    floors=[
        FloorSpec(room_spec=ground, stair_host_room_id=3),
        FloorSpec(room_spec=upper, stair_host_room_id=2),
    ],
)

generator = HouseGenerator(
    split="train",
    seed=1234,
    house_spec=house_spec,
    controller=controller,
)
house, _ = generator.sample()
```

An explicit `house_spec` cannot be combined with `num_floors`, `room_spec`,
`room_spec_sampler`, `partial_house`, or `house_spec_sampler`. Input room IDs
are local to their floor; schema-2 assembly maps them deterministically to
globally unique room numbers. If `stair_host_room_id` is supplied, it must
identify a `LivingRoom` or `Bedroom` leaf room on that floor.

## Geometry contract

The first schema-2 implementation intentionally uses one fixed contract:

- two or three floors with a shared aligned footprint;
- `3.0 m` floor-to-floor pitch;
- `0.2 m` intermediate slab and `2.8 m` clear room height;
- one shared `1.2 m x 6.5 m` reserved stair core;
- a straight `1.0 m` wide flight with a `4.5 m` run and `3.0 m` rise; and
- for three floors, the second flight is rotated `180°` relative to the first.

The matching public constants are `FLOOR_TO_FLOOR_HEIGHT`,
`MULTI_FLOOR_SLAB_THICKNESS`, `MULTI_FLOOR_CLEAR_HEIGHT`, `STAIR_CORE_WIDTH`,
`STAIR_CORE_LENGTH`, `STAIR_WIDTH`, and `STAIR_RUN` in `procthor.constants`.
Stairs occupy existing living rooms (falling back to bedrooms where needed),
not a synthetic stairwell room. Exterior doors are generated only on the ground
floor; an agent starting room may be selected from any floor.

The generator locates the shared core during bounded complete-structure
sampling and reserves it before furnishing. It retries at most
`MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS` (`50`) complete aligned structures.
Exhausting that bound raises `InvalidMultiFloorPlan` rather than returning a
partially valid house.

Each aligned attempt samples floor doors first, subtracts the exact padded
doorway/open-wall clearance polygons from the candidate stair-host regions, and
then refits the shared core across those doorway-safe regions. If no shared
doorway-safe fit remains, the generator rejects and retries the entire aligned
attempt.

Immediately before floor-object placement, the stair host room also reserves a
square-cornered `0.8 m` apron around the entire core. The runtime can connect
through either lateral landing edge or the exposed end edge and samples
room-side points at offsets up to `0.8 m`; preserving this apron prevents
furniture from blocking every valid egress.

## Schema 2 output

Single-floor houses remain on schema `1.0.0`; only multi-floor houses emit
schema `2.0.0` (`MULTI_FLOOR_SCHEMA`). The schema-2 additions are:

- top-level `schema == "2.0.0"` and `metadata.schema == "2.0.0"`;
- top-level `floors`, including each level's index, `baseY`, `ceilingY`, slab
  thickness, global room IDs, and input-to-global room-ID map;
- top-level `verticalConnectors`, including the stair asset transform, adjacent
  floors and rooms, landings, and slab-opening polygons;
- room `floorId` and physical `floorPolygons`, while retaining the semantic
  `floorPolygon`; and
- explicit, actual-world-space ceiling pieces.

Rooms, walls, doors, windows, lights, and objects otherwise remain flat arrays.
Structural IDs for exteriors, surfaces, openings, landings, and connectors are
deterministic and floor-qualified.

Schema-2 generation currently runs as one complete operation:
`next_sampling_stage` must remain `NextSamplingStage.STRUCTURE`, and legacy
`PartialHouse` resume is rejected. With `return_partial_houses=True`, the
returned mapping contains the aligned pre-generation context at `DOORS` and
the assembled context at `COMPLETE`.

## Required AI2-THOR integration

The maintained engine integration is pinned to AI2-THOR commit
`24f79883b4889e3f0e6f4ae301808b9025872dfc`, also exposed as
`MULTI_FLOOR_AI2THOR_COMMIT`. Apply
[`integrations/ai2thor/ai2thor-schema2-multifloor.patch`](../integrations/ai2thor/ai2thor-schema2-multifloor.patch)
to that revision and rebuild AI2-THOR. See the companion
[`integrations/ai2thor/README.md`](../integrations/ai2thor/README.md) and
[`stair-asset-contract.json`](../integrations/ai2thor/stair-asset-contract.json)
for engine-side schema behavior and the required curated stair prefab contract.

The patched engine advertises supported house schemas, creates schema-2 slabs
and ceilings at their supplied world-space elevations, places structural stairs
before navmesh construction, and replaces the stair prefab's invisible ramp mesh
with one connected lower-landing/ramp/upper-landing collider. Navmesh baking
excludes trigger volumes, caps physics-collider `minRegionArea` at `0.05`,
and tags physical floor surfaces for height raycasts. Every internal doorway
receives a sampled bidirectional zero-width link per active agent type so the
two room surfaces remain connected while physical colliders still gate motion.
Required bidirectional links connect room surfaces to inset ramp anchors for
every active agent type. Additional locally adjacent links are sampled at three
positions across both lateral edges and the exposed end edge of each landing
where both surfaces exist. A short link bridges the ramp mesh seam, and scene
creation requires a complete lower-to-upper navmesh path. Standard movement
uses physical floor/stair heights plus bounded
forward lookahead to cross those links and requires support beneath the
horizontal action target. The lower ceiling and upper floor retain full-core
1.2 m by 6.5 m openings. An unpatched engine is rejected by the capability
preflight rather than silently flattening the scene.

## Generate the mixed-floor dataset

The dataset script samples `50%` single-floor, `35%` two-floor, and `15%`
three-floor houses. Point it at the patched executable so one shared controller
can create both schema versions:

```shell
export PROCTHOR_SCHEMA2_EXECUTABLE=/path/to/patched/ai2thor-executable
uv run --with-editable . python scripts/generate_procthor_10k_dataset.py
```

Retries retain the selected floor count and complete room specification.

## Legacy compatibility

Omitting all multi-floor arguments follows the original single-floor pipeline
exactly. Passing `num_floors=1` is an explicit alias for that same path. Both
cases retain schema `1.0.0`, the legacy output shape, generation order, and seed
behavior; they do not perform the schema-2 capability preflight.
