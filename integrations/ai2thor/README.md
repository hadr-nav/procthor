# AI2-THOR schema-2 multi-floor integration

ProcTHOR schema 2 requires a coordinated AI2-THOR change. This directory pins
that engine half so generated houses fail fast on older builds instead of being
silently flattened onto y=0.

## Pinned base

- Repository: `allenai/ai2thor`
- Commit: `24f79883b4889e3f0e6f4ae301808b9025872dfc`
- Machine-readable manifest: `manifest.json`
- Source patch: `ai2thor-schema2-multifloor.patch`
- Curated prefab contract: `stair-asset-contract.json`

Apply from a clean checkout at the pinned commit:

```shell
git rev-parse HEAD
git apply --check /path/to/procthor/integrations/ai2thor/ai2thor-schema2-multifloor.patch
git apply /path/to/procthor/integrations/ai2thor/ai2thor-schema2-multifloor.patch
```

The first command must print the pinned SHA. Do not apply the patch to another
revision without reviewing every hunk and rerunning the Unity tests below.

## What the patch changes

The schema-1 constant remains `1.0.0`, and its existing floor, global collider,
ceiling-height, and reachable-position code remains on the legacy branch.
Schema `2.0.0` adds:

- `GetSupportedHouseSchemas`, returning `["1.0.0", "2.0.0"]` without
  consuming random state;
- rich room floor surfaces tagged `StructureObjectTag.Floor` and explicit
  ceiling surfaces in actual world y;
- top-level floor records and vertical-connector records matching ProcTHOR's
  serializer;
- preflight validation of floor/connector topology, surface non-overlap,
  opening clearance, flattened copies, and prefab geometry before scene mutation;
- one slab collider and receptacle per rectangular semantic floor surface,
  plus one consolidated navmesh-only floor collider per storey, leaving the
  complete 1.2 m by 6.5 m reserved stair core open in the lower ceiling and
  upper floor instead of covering it with a global rectangle;
- explicit per-piece ceiling meshes instead of one global-height ceiling;
- structural stair spawning before navmesh baking, including replacement of the
  authored ramp collider mesh with one lower-landing/ramp/upper-landing surface
  extended through the exposed end and both lateral edges at every landing;
- all procedural navmeshes collected from the same non-trigger physics
  colliders used by movement, with every schema-2
  effective agent slope clamped at least 0.5 degrees above the serialized ramp
  slope; the configured disconnected-region threshold is retained so tiny
  isolated islands are removed instead of being preserved for the stair;
- a link-free physical navmesh: no synthetic doorway, landing, ramp-seam, or
  connector links are created;
- `GetNavMeshConnectivity`, which selects one agent type, covers every baked
  triangle, and reports exact path-connected component sizes;
- navmesh-based schema-2 reachable positions containing distinct floor y
  values;
- schema-2 shortest-path endpoint sampling preserves each requested y instead
  of flattening both endpoints to one floor;
- optional `targetY` is propagated through remote `MoveAhead`, allowing
  standard movement to follow the height of the continuous physical navmesh;
  and
- a `VerticalConnectorAsset` marker that validates the ramp, exact landing
  anchors, and both 1.2 m by 1.0 m platform colliders.

The schema-2 wire types are intentionally richer than schema 1:

- `room.floorPolygons`: surface objects with `id`, `floorId`, `roomId`,
  `surfaceType`, `polygon`, `slabThickness`, and optional `material`;
- `room.ceilings`: explicit surface objects with the same ownership fields,
  polygon, and material;
- `floors[*].floorSurfaces` and `ceilingSurfaces`: flattened copies whose
  complete records must exactly match their rooms; and
- `verticalConnectors[*]`: `connectorType`, transform, adjacent floor/room
  IDs, `assetContract`, `landingPolygons`, and `openingPolygons`.

Schema 2 currently accepts only non-overlapping, axis-aligned rectangular
physical surface pieces. ProcTHOR decomposes a rectangular stair opening into
those pieces before serialization.

Before floor-object placement, ProcTHOR subtracts a square-cornered `0.8 m`
buffer around the complete reserved core from the stair host room's open
polygon. This keeps the front and both lateral approaches physically clear.
The reservation is applied while constraining the grid partition on every
floor, and the partition is rejected if reassigning it would empty or split any
room.

## Curated production asset

Author and check in a Unity prefab named
`Staircase_Straight_3m_1m_4_5m`. The JSON contract is normative. In
particular:

1. The prefab origin is the shared core center at the lower floor's base y.
2. Local +z is the direction of ascent.
3. The visual staircase fits the 1.2 m by 6.5 m reserved envelope.
4. The prefab root has identity local scale. Its disabled-renderer
   `walkableSurface` is a direct child at an identity local transform and owns
   one enabled, non-trigger ramp `MeshCollider` with an exact 1.0 m width,
   3.0 m rise, and 4.5 m run.
5. Lower and upper platform objects are direct `walkableSurface` children whose
   transforms are their landing anchors. Each owns a 1.2 m by 0.2 m by 1.0 m
   `BoxCollider` centered at `(0, -0.1, 0)`, placing its top face on the anchor
   plane. Renderers on the ramp and platform hierarchy are absent or disabled.
   These colliders remain physical at runtime but are ignored as independent
   navmesh sources; the engine replaces the authored ramp mesh with one
   connected landing/ramp/landing collision mesh before baking.
6. The root has `VerticalConnectorAsset`, with references to the ramp, both
   platform objects, and both landing anchors.
7. Visual steps and rails do not contribute walkable navmesh geometry. The
   patch marks the root Not Walkable and the walkable hierarchy Walkable.
8. Register the prefab in the Procedural scene's
   `ProceduralAssetDatabase.prefabs` list so `AssetInDatabase` and
   `CreateHouse` can resolve its exact asset ID.

The prefab, model, materials, generated `.meta` files, and serialized
Procedural-scene GUID reference are Unity assets and are not representable in
this source-only patch. They must be added in the AI2-THOR repository as one
reviewed asset change.

## Required engine verification

Run these checks in the patched AI2-THOR checkout with the production prefab
registered:

1. Unity EditMode: schema support action returns exactly 1.0.0 and 2.0.0.
2. Unity PlayMode: the existing schema-1 procedural tests pass unchanged.
3. Unity PlayMode: create the same two-floor fixture with the agent starting
   once on each level; assert floor meshes, slab colliders, receptacles,
   ceilings, walls, lights, objects, doors, windows, and agent pose retain
   their serialized y, and every physical schema-2 floor surface carries
   `StructureObjectTag.Floor`.
4. Unity PlayMode: raycast through both full-core serialized opening polygons
   and assert no slab or ceiling collider covers either 1.2 m by 6.5 m opening.
5. Unity PlayMode: assert the stair instance exists before
   `NavMeshSurfaceExtended.BuildNavMesh`, its runtime collider is the connected
   landing/ramp/landing mesh, its separate platform colliders are ignored by the
   bake, its visual hierarchy is Not Walkable, trigger colliders are excluded,
   the surface uses `PhysicsColliders`, and every effective agent slope is at
   least `34.1900675` degrees while retaining the configured `minRegionArea`.
   Assert the connected mesh includes a `0.6 m` end apron and two
   `0.6 m x 1.0 m` side aprons at both landings. Assert every room-floor piece
   is represented in exactly one consolidated floor source for its storey and
   the result has no runtime `NavMeshLink` instances.
6. Controller integration: `GetNavMeshConnectivity` returns one component and
   a component size equal to the complete selected-navmesh triangle count.
7. Controller integration: `GetReachablePositions` from either floor returns
   positions near both base y values and `GetShortestPath` completes across
   the stair and at least 0.5 m beyond the complete stair envelope; repeated
   standard movement actions leave the stair for the upper room and traverse it
   in both directions. Repeat the path assertion from the front, left, and right
   approach of both landings.
8. Repeat for three floors and assert both flights retain the shared parallel
   orientation and the middle floor remains connected to each flight.
9. Negative cases: missing prefab, scaled or mis-parented ramp, malformed or
   offset platform colliders, duplicate/missing adjacent connector pairs,
   overlapping or opening-covering surfaces, inconsistent flattened copies,
   invalid materials, non-adjacent floor IDs, and unsupported schema all fail
   before scene mutation.

This artifact is prepared against the pinned source checkout and clean-applies
there. Runtime verification evidence for the current integration is recorded in
the repository change that updates this artifact; the separate EditMode and
PlayMode suites above remain required.
