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
        self.assertEqual(self.manifest["patch"]["sha256"], sha256(PATCH_PATH))
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
        self.assertEqual(patched_paths, self.manifest["patchedSourceFiles"])

        for required_source in (
            "GetSupportedHouseSchemas",
            "GetNavMeshConnectivity",
            "NavMeshConnectivity",
            "NavMesh.CalculateTriangulation",
            "componentSizes",
            "connected = componentSizes.Count == 1",
            "NavMeshCollectGeometry.PhysicsColliders",
            "minimumAgentSlope",
            "validRampSurface",
            "validateOpeningClear",
            "sameFloorSurface",
            "connectorLowerFloorIndices",
            "lowerLandingSurface",
            "upperLandingSurface",
            "landingEgressDepth",
            "createConnectedStairWalkableMesh",
            "ConnectedStairLandingRampCollider",
            "createMultiFloorNavMeshFloors",
            "MultiFloorNavMeshFloors",
            "ConsolidatedNavMeshFloor",
            "mesh.CombineMeshes",
            "tagObjectNavmesh(surfaceObject, ignore: true)",
            "tagObjectNavmesh(marker.lowerLandingSurface, ignore: true)",
            "tagObjectNavmesh(marker.upperLandingSurface, ignore: true)",
            "floorStructure.WhatIsMyStructureObjectTag = StructureObjectTag.Floor",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, self.patch)

        for forbidden_source in (
            "NavMeshLinkData",
            "addDoorNavMeshLinks",
            "addVerticalConnectorNavMeshLinks",
            "AddRuntimeNavMeshLink",
            "sampleLandingBoundaryNavMeshLinks",
            "usedLinkLookAhead",
        ):
            with self.subTest(forbidden_source=forbidden_source):
                self.assertNotIn(forbidden_source, self.patch)

        dimensions = self.contract["dimensions"]
        self.assertEqual(dimensions["landingEgressDepth"], 0.6)
        self.assertEqual(self.contract["walkableRamp"]["collider"]["width"], 1.0)
        self.assertEqual(
            self.contract["component"]["prefabRootLocalScale"],
            {"x": 1.0, "y": 1.0, "z": 1.0},
        )

        platforms = self.contract["landingPlatforms"]
        self.assertEqual(platforms["parentRelationship"], "direct")
        self.assertEqual(platforms["height"], 0.2)
        self.assertEqual(
            platforms["colliders"]["localCenter"],
            {"x": 0.0, "y": -0.1, "z": 0.0},
        )
        self.assertTrue(platforms["physicalCollisionRetained"])
        self.assertEqual(platforms["runtimeNavMeshSource"], "ignored")

        runtime_collider = self.contract["runtimeWalkableCollider"]
        self.assertTrue(runtime_collider["replacesAuthoredRampMesh"])
        self.assertEqual(runtime_collider["landingEgressDepth"], 0.6)
        self.assertEqual(runtime_collider["sideApronLength"], 1.0)
        self.assertEqual(
            runtime_collider["physicalSurfaceSegments"],
            [
                "lowerLanding",
                "ramp",
                "upperLanding",
                "lowerFrontApron",
                "lowerLeftApron",
                "lowerRightApron",
                "upperFrontApron",
                "upperLeftApron",
                "upperRightApron",
            ],
        )

        navigation = self.contract["navigation"]
        self.assertEqual(navigation["minimumAgentSlopeDegrees"], 34.1900675)
        self.assertTrue(navigation["retainConfiguredMinRegionArea"])
        self.assertEqual(navigation["floorStructureObjectTag"], "Floor")
        self.assertTrue(navigation["triggerCollidersExcludedFromBake"])
        self.assertEqual(
            navigation["landingApproaches"]["requiredPerLanding"],
            ["front", "left", "right"],
        )
        self.assertFalse(
            navigation["landingApproaches"]["syntheticNavMeshLinks"]
        )
        self.assertTrue(
            navigation["floorSources"]["oneConsolidatedColliderPerFloor"]
        )
        self.assertTrue(
            navigation["floorSources"]["semanticFloorSourcesIgnoredByNavMesh"]
        )
        topology = navigation["wholeNavMeshValidation"]
        self.assertEqual(topology["action"], "GetNavMeshConnectivity")
        self.assertEqual(topology["requiredComponentCount"], 1)
        self.assertTrue(topology["datasetRejectsFailure"])

        self.assertEqual(
            self.contract["connectorRecordExample"]["assetContract"][
                "landingEgressDepth"
            ],
            0.6,
        )
        self.assertEqual(
            self.contract["generatorObjectClearance"]["distanceFromReservedCore"],
            0.8,
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

    def test_runtime_stair_mesh_has_three_physical_approaches_per_landing(self):
        method = self.patch.split(
            "private static Mesh createConnectedStairWalkableMesh", 1
        )[1].split("private static GameObject createMultiFloorNavMeshFloors", 1)[0]

        self.assertEqual(method.count("addHorizontalQuad("), 6)
        self.assertIn("var egressDepth = contract.landingEgressDepth", method)
        self.assertIn("lowerLandingCenter - landingDepth / 2.0f", method)
        self.assertIn("upperLandingCenter + landingDepth / 2.0f", method)
        self.assertIn("-halfEnvelope - egressDepth", method)
        self.assertIn("halfLandingWidth + egressDepth", method)

    def test_floor_navmesh_source_eliminates_room_and_door_seams(self):
        method = self.patch.split(
            "private static GameObject createMultiFloorNavMeshFloors", 1
        )[1].split("private static GameObject spawnVerticalConnectors", 1)[0]

        for required_source in (
            ".SelectMany(room => room.floorPolygons)",
            ".GroupBy(surface => surface.floorId)",
            "GenerateWorldSpaceFloorMesh(surface.polygon)",
            "mesh.CombineMeshes(combines, true, true, false)",
            "floorObject.AddComponent<MeshCollider>()",
            "StructureObjectTag.Floor",
            'tagObjectNavmesh(floorObject, "Walkable")',
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, method)
        self.assertIn(
            "house.metadata.navMeshes,\n+                minimumAgentSlope,\n+                true",
            self.patch,
        )

    def test_connectivity_action_classifies_the_complete_selected_navmesh(self):
        method = self.patch.split("public void GetNavMeshConnectivity", 1)[1].split(
            "public void CreateHouse", 1
        )[0]

        for required_source in (
            "activateOnlyNavmeshSurface",
            "NavMesh.CalculateTriangulation()",
            "var representatives = new Vector3[triangleCount]",
            "new HashSet<int>(Enumerable.Range(0, triangleCount))",
            "NavMesh.CalculatePath",
            "NavMeshPathStatus.PathComplete",
            "componentSizes.Sort",
            "componentCount = componentSizes.Count",
            "connected = componentSizes.Count == 1",
            "activateAllNavmeshSurfaces",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, method)


if __name__ == "__main__":
    unittest.main()
