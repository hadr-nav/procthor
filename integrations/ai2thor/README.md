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
- one slab collider and receptacle per rectangular floor surface, leaving the
  complete 1.2 m by 6.5 m reserved stair core open in the lower ceiling and
  upper floor instead of covering it with a global rectangle;
- explicit per-piece ceiling meshes instead of one global-height ceiling;
- structural stair spawning before navmesh baking, including replacement of the
  authored ramp collider mesh with one shared-vertex lower-landing/ramp/upper-
  landing surface;
- schema-2 navmeshes collected from non-trigger physics colliders with every
  effective agent slope clamped at least 0.5 degrees above the serialized ramp
  slope and `minRegionArea` capped at `0.05`;
- sampled, bidirectional zero-width links across every internal doorway for
  each active navmesh agent type, restoring room-to-room connectivity while
  leaving physical door and wall colliders authoritative for movement;
- required bidirectional zero-width links from room surfaces to inset ramp
  anchors, additional locally adjacent links sampled at three positions across
  both lateral edges and the exposed end edge of each landing, and a short link
  across the ramp mesh seam for every active navmesh agent type; candidate
  landing links that cross a physical wall are rejected using explicit
  `StructureObjectTag.Wall` tags, and scene creation rejects a connector unless
  the resulting navmesh has a complete
  lower-to-upper path;
- navmesh-based schema-2 reachable positions containing distinct floor y
  values;
- schema-2 shortest-path endpoint sampling preserves each requested y instead
  of flattening both endpoints to one floor;
- optional `targetY` is propagated through remote `MoveAhead`, and schema-2
  discrete movement is projected using physical floor/stair height
  raycasts, with bounded `0.25`/`0.5`/`0.75` m forward lookahead and a
  local path-length cap of `5×` the action distance plus `0.1` m, allowing
  standard movement actions to cross landing links and ascend or descend only
  when the horizontal action target has physical floor or stair support; and
- a `VerticalConnectorAsset` marker that validates the ramp, exact landing
  anchors, and both 1.2 m by 1.0 m platform colliders, then owns the runtime
  navmesh-link lifecycle.

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
polygon. This keeps every possible room-side landing egress clear for the
runtime link sampler, including its farthest `0.8 m` query offset.

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
   platform objects, and both landing anchors. It also owns every runtime
   navmesh link and removes valid instances when destroyed.
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
   least `34.1900675` degrees while `minRegionArea <= 0.05`. After baking,
   assert required room-to-ramp links exist, additional locally adjacent links
   are created wherever both sides sample successfully at three positions
   across each landing edge, the ramp mesh seam is bridged, and the result is a
   complete lower-to-upper navmesh path for every active agent type.
6. Unity PlayMode: assert every internal doorway has one sampled bidirectional
   link per active agent type and that links are removed with their owning door.
7. Controller integration: `GetReachablePositions` from either floor returns
   positions near both base y values and `GetShortestPath` completes across
   the stair and at least 0.5 m beyond the complete stair envelope; repeated
   standard movement actions leave the stair for the upper room and traverse it
   in both directions while physical floor/stair raycasts drive the returned
   agent y. Exercise direct and link-lookahead projections, reject unsupported
   horizontal targets, and reject paths over the configured `5× + 0.1 m` local
   cap.
8. Repeat for three floors and assert the second stair is yaw-rotated 180
   degrees.
9. Negative cases: missing prefab, scaled or mis-parented ramp, malformed or
   offset platform colliders, duplicate/missing adjacent connector pairs,
   overlapping or opening-covering surfaces, inconsistent flattened copies,
   invalid materials, non-adjacent floor IDs, and unsupported schema all fail
   before scene mutation.

This artifact was prepared against the pinned source checkout and clean-applies
there. The patched Unity player was built and exercised through AI2-THOR on 100
ProcTHOR scenes; all 100 completed A* waypoint following with zero collisions.
The separate EditMode and PlayMode suites above remain required.
