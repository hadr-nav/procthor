"""Integrity tests for the pinned schema-2 AI2-THOR integration artifacts."""

import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = REPO_ROOT / "integrations" / "ai2thor"
MANIFEST_PATH = INTEGRATION_ROOT / "manifest.json"
PATCH_PATH = INTEGRATION_ROOT / "ai2thor-schema2-multifloor.patch"
CONTRACT_PATH = INTEGRATION_ROOT / "stair-asset-contract.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IntegrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        cls.contract = json.loads(CONTRACT_PATH.read_text())
        cls.patch = PATCH_PATH.read_text()

    def test_manifest_hashes_and_base_commit_match_artifacts(self):
        self.assertEqual(
            self.manifest["patch"]["sha256"],
            sha256(PATCH_PATH),
        )
        self.assertEqual(
            self.manifest["stairAsset"]["contractSha256"],
            sha256(CONTRACT_PATH),
        )
        self.assertEqual(
            self.contract["engineBaseCommit"],
            self.manifest["engineBaseCommit"],
        )
        self.assertEqual(
            self.contract["assetId"],
            self.manifest["stairAsset"]["assetId"],
        )

    def test_patch_scope_and_navigation_contract_are_synchronized(self):
        patched_paths = []
        for line in self.patch.splitlines():
            if not line.startswith("diff --git "):
                continue
            _, _, left, right = line.split()
            self.assertEqual(left[2:], right[2:])
            patched_paths.append(left[2:])
        self.assertEqual(
            patched_paths,
            self.manifest["patchedSourceFiles"],
        )

        for required_source in (
            "GetSupportedHouseSchemas",
            "tryProjectMultiFloorMovementTarget",
            "NavMeshCollectGeometry.PhysicsColliders",
            "minimumAgentSlope",
            "validRampSurface",
            "validateOpeningClear",
            "sameFloorSurface",
            "connectorLowerFloorIndices",
            "lowerLandingSurface",
            "upperLandingSurface",
            "createConnectedStairWalkableMesh",
            "ConnectedStairLandingRampCollider",
            "var collider = source.component as Collider;",
            "tagObjectNavmesh(marker.lowerLandingSurface, ignore: true)",
            "tagObjectNavmesh(marker.upperLandingSurface, ignore: true)",
            "addDoorNavMeshLinks",
            "addVerticalConnectorNavMeshLinks",
            "AddRuntimeNavMeshLink",
            "runtimeNavMeshLinks",
            "settings.minRegionArea = Mathf.Min",
            "floorStructure.WhatIsMyStructureObjectTag = StructureObjectTag.Floor",
            "tryGetMultiFloorWalkableSurfaceY",
            "usedLinkLookAhead",
            "rampInset",
            "lowerMidQuery",
            "upperMidQuery",
            "sampleRampAdjacentRoomNavMeshHits",
            "sampleLandingBoundaryNavMeshLinks",
            "lowerRampRoomHits",
            "upperRampRoomHits",
            "lowerLandingLinks",
            "upperLandingLinks",
            "NavMesh.CalculatePath",
            "NavMeshPathStatus.PathComplete",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, self.patch)

        self.assertEqual(
            self.contract["walkableRamp"]["collider"]["width"],
            1.0,
        )
        self.assertEqual(
            self.contract["component"]["prefabRootLocalScale"],
            {"x": 1.0, "y": 1.0, "z": 1.0},
        )
        self.assertEqual(self.contract["walkableRamp"]["parent"], "prefabRoot")
        platforms = self.contract["landingPlatforms"]
        self.assertEqual(platforms["parentRelationship"], "direct")
        self.assertEqual(platforms["height"], 0.2)
        self.assertEqual(
            platforms["colliders"]["localCenter"],
            {"x": 0.0, "y": -0.1, "z": 0.0},
        )
        self.assertEqual(platforms["colliders"]["topFaceLocalY"], 0.0)
        self.assertTrue(platforms["physicalCollisionRetained"])
        self.assertEqual(platforms["runtimeNavMeshSource"], "ignored")
        runtime_collider = self.contract["runtimeWalkableCollider"]
        self.assertTrue(runtime_collider["replacesAuthoredRampMesh"])
        self.assertEqual(
            runtime_collider["sharedVertexSurfaceSegments"],
            ["lowerLanding", "ramp", "upperLanding"],
        )
        navigation = self.contract["navigation"]
        self.assertEqual(navigation["minimumAgentSlopeDegrees"], 34.1900675)
        self.assertEqual(navigation["maximumMinRegionArea"], 0.05)
        self.assertEqual(navigation["floorStructureObjectTag"], "Floor")
        self.assertTrue(navigation["triggerCollidersExcludedFromBake"])
        height_raycast = navigation["physicalSurfaceHeightRaycast"]
        self.assertEqual(height_raycast["originHeightAboveAgent"], 0.5)
        self.assertEqual(height_raycast["distance"], 2.5)
        self.assertEqual(
            height_raycast["recognizedConnectorSurfaces"],
            ["walkableSurface", "lowerLandingSurface", "upperLandingSurface"],
        )
        look_ahead = navigation["movementLookAhead"]
        self.assertEqual(look_ahead["distances"], [0.25, 0.5, 0.75])
        self.assertEqual(look_ahead["sampleRadius"], 0.25)
        self.assertTrue(look_ahead["requiresPhysicalSurfaceAtHorizontalTarget"])
        self.assertEqual(look_ahead["maximumPathLengthMultiplier"], 5.0)

        links = navigation["landingBoundaryLinks"]
        self.assertEqual(
            links["adjacentEdgesPerLanding"],
            {"lateral": 2, "exposedEnd": 1},
        )
        self.assertTrue(links["connectsEverySampledAdjacentEdge"])
        self.assertEqual(links["landingLongitudinalFractions"], [-0.25, 0.0, 0.25])
        self.assertEqual(links["landingLateralFractions"], [-0.25, 0.0, 0.25])
        self.assertEqual(
            links["rampQueryInset"],
            {"minimum": 0.6, "agentRadiusPadding": 0.4},
        )
        self.assertEqual(
            links["landingSurfaceInset"],
            {"agentRadiusPadding": 0.05},
        )
        self.assertEqual(
            links["adjacentEdgeQueryInset"],
            {"agentRadiusPadding": 0.2},
        )
        self.assertEqual(
            links["internalRampSeamQueryInset"],
            {"agentRadiusPadding": 0.1},
        )
        self.assertEqual(links["sampleDistance"], 0.2)
        self.assertTrue(links["roomToRampLinksRequired"])
        self.assertTrue(links["landingBoundaryLinksRequireBothSamples"])
        self.assertTrue(links["internalRampSeamLink"])
        self.assertEqual(links["width"], 0.0)
        self.assertTrue(links["bidirectional"])
        self.assertTrue(links["completeLowerToUpperPathRequired"])
        self.assertEqual(
            links["failureDiagnostics"],
            [
                "lowerRampSampled",
                "lowerMidRampSampled",
                "upperMidRampSampled",
                "upperRampSampled",
                "lowerRampRoomCount",
                "upperRampRoomCount",
                "lowerLandingLinkCount",
                "upperLandingLinkCount",
            ],
        )
        self.assertEqual(
            self.contract["generatorObjectClearance"],
            {
                "distanceFromReservedCore": 0.8,
                "bufferJoinStyle": "mitre",
                "appliedBeforeFloorObjectPlacement": True,
                "purpose": (
                    "preserve room-side landing egress along every exposed core edge"
                ),
            },
        )
        self.assertEqual(
            self.contract["slabOpenings"],
            {"coverEntireReservedCore": True, "width": 1.2, "length": 6.5},
        )

        for opening in self.contract["connectorRecordExample"]["openingPolygons"]:
            xs = [point["x"] for point in opening["polygon"]]
            zs = [point["z"] for point in opening["polygon"]]
            self.assertAlmostEqual(max(xs) - min(xs), 1.2)
            self.assertAlmostEqual(max(zs) - min(zs), 6.5)

    def test_runtime_landing_links_are_sampled_for_active_agent_types(self):
        link_method = self.patch.split(
            "private static List<NavMeshHit> sampleRampAdjacentRoomNavMeshHits", 1
        )[1].split("private static void createMultiFloorRoomSurfaces", 1)[0]

        for required_source in (
            "GameObject.FindObjectsOfType<NavMeshSurfaceExtended>()",
            ".Where(surface => surface.isActiveAndEnabled)",
            "new NavMeshQueryFilter()",
            "agentBuildSettings",
            "var rampInset = Mathf.Max(",
            "0.6f,",
            "agentBuildSettings.agentRadius + 0.4f",
            "agentRadius + 0.2f",
            "agentBuildSettings.agentRadius + 0.1f",
            "var landingInset = agentRadius + 0.05f",
            "new[] { -1.0f, 1.0f }",
            "new[] { -0.25f, 0.0f, 0.25f }",
            "const float sampleDistance = 0.2f",
            "lowerMidQuery",
            "upperMidQuery",
            "lowerRampRoomHits",
            "upperRampRoomHits",
            "lowerLandingLinks",
            "upperLandingLinks",
            "NavMesh.SamplePosition",
            "links.Add((landingHit, roomHit))",
            "foreach (var roomHit in lowerRampRoomHits)",
            "foreach (var link in lowerLandingLinks)",
            "foreach (var link in upperLandingLinks)",
            "foreach (var roomHit in upperRampRoomHits)",
            "width = 0.0f",
            "bidirectional = true",
            "lowerRampRoomHits.Count == 0",
            "|| upperRampRoomHits.Count == 0",
            "NavMesh.CalculatePath",
            "path.status != NavMeshPathStatus.PathComplete",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, link_method)
        self.assertNotIn("|| lowerLandingLinks.Count == 0", link_method)
        self.assertNotIn("|| upperLandingLinks.Count == 0", link_method)

    def test_internal_doorways_receive_owned_bidirectional_links(self):
        link_method = self.patch.split("private static void addDoorNavMeshLinks", 1)[
            1
        ].split("private static void createMultiFloorRoomSurfaces", 1)[0]

        for required_source in (
            "doors.Where(door =>",
            "!String.IsNullOrEmpty(door.room1) && door.room0 != door.room1",
            "GameObject.FindObjectsOfType<NavMeshSurfaceExtended>()",
            "new NavMeshQueryFilter()",
            "NavMesh.SamplePosition",
            "owner.AddRuntimeNavMeshLink",
            "width = 0.0f",
            "bidirectional = true",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, link_method)

        self.assertIn(
            "addDoorNavMeshLinks(house.doors, walls)",
            self.patch,
        )
        self.assertIn(
            "Could not create a NavMesh link for doorway",
            self.patch,
        )
        self.assertIn("runtimeNavMeshLinks", self.patch)

    def test_multifloor_movement_uses_physical_heights_and_bounded_lookahead(self):
        height_method = self.patch.split(
            "private bool tryGetMultiFloorWalkableSurfaceY", 1
        )[1].split("private bool tryProjectMultiFloorMovementTarget", 1)[0]
        movement_method = self.patch.split(
            "private bool tryProjectMultiFloorMovementTarget", 1
        )[1].split("// XXX revisit", 1)[0]

        self.assertIn(
            ".RaycastAll(origin, Vector3.down, 2.5f, layerMask)", height_method
        )
        self.assertIn("StructureObjectTag.Floor", height_method)
        self.assertIn("connector.walkableSurface", height_method)
        self.assertIn("controllerScaleY", movement_method)
        self.assertIn("controllerBottomY", movement_method)
        self.assertIn("TransformPoint(m_CharacterController.center).y", movement_method)
        self.assertIn(
            "m_CharacterController.height * controllerScaleY / 2.0f",
            movement_method,
        )
        self.assertNotIn("m_CharacterController.bounds.min.y", movement_method)
        self.assertIn("0.35f", movement_method)
        self.assertNotIn("1.25f", movement_method)
        self.assertIn("usedLinkLookAhead", movement_method)
        self.assertIn("new[] { 0.25f, 0.5f, 0.75f }", movement_method)
        self.assertIn("tryGetMultiFloorWalkableSurfaceY", movement_method)
        self.assertIn("hasPhysicalSurfaceHeights", movement_method)
        self.assertIn(
            "usedLinkLookAhead && !hasTargetPhysicalSurface",
            movement_method,
        )
        self.assertIn(
            "targetPhysicalSurfaceY - currentPhysicalSurfaceY",
            movement_method,
        )
        self.assertIn("Vector2.Distance(targetXZ, hitXZ) > 0.3f", movement_method)
        self.assertIn("horizontalDistance * 5.0f + 0.1f", movement_method)
        self.assertIn(
            "transform.position.y - startHit.position.y",
            movement_method,
        )


if __name__ == "__main__":
    unittest.main()
