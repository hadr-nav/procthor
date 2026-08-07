USE_ITHOR_SPLITS = False
"""Determines if the iTHOR object splits should be used spawning objects."""

OPENNESS_RANDOMIZATIONS = {
    "Box": {"population": [0, 1], "weights": [0.5, 0.5]},
    "Laptop": {"population": [0, 1, "any"], "weights": [0.4, 0.4, 0.2]},
}
"""Parameters that specify the openness randomization of an object.

Currently assumes that opening the object roughly doesn't influcence any
other objects in the scene (e.g., there are no objects on top of the object,
opening doesn't cause the size of the object to expand in the x/z direction.)
"""

SCHEMA = "1.0.0"
"""The schema version of the json file to create the house."""

MULTI_FLOOR_SCHEMA = "2.0.0"
"""The schema version used only for houses with multiple floors."""

MULTI_FLOOR_AI2THOR_COMMIT = "24f79883b4889e3f0e6f4ae301808b9025872dfc"
"""AI2-THOR revision against which the schema-2 integration patch is maintained."""

FLOOR_TO_FLOOR_HEIGHT = 3.0
"""Vertical distance, in meters, between adjacent generated floors."""

MULTI_FLOOR_CLEAR_HEIGHT = 2.8
"""Usable room height below the intermediate slab."""

MULTI_FLOOR_SLAB_THICKNESS = 0.2
"""Thickness, in meters, of schema-2 floor and ceiling surfaces."""

MULTI_FLOOR_MAX_STRUCTURE_ATTEMPTS = 50
"""Maximum complete structure attempts before multi-floor generation fails."""

STAIR_CORE_WIDTH = 1.2
STAIR_CORE_LENGTH = 6.5
STAIR_WIDTH = 1.0
STAIR_RUN = 4.5
STAIR_LANDING_EGRESS_DEPTH = 0.6
MULTI_FLOOR_STAIR_ASSET_ID = "Staircase_Straight_3m_1m_4_5m"
"""Reference stair contract shared by ProcTHOR and the AI2-THOR patch."""

MARGIN = {
    "middle": 0.35,
    "edge": {"front": 0.5, "back": 0, "sides": 0},
    "corner": {"front": 0.5, "back": 0, "sides": 0},
}
"""The margin between different objects."""

PADDING_AGAINST_WALL = 0.05
"""Padding, or extra space, added to each object.

This helps keep objects from colliding into the wall.
"""

P_CHOOSE_ASSET_GROUP = 0.6
"""The probability of choosing a semantic asset group over a standalone asset."""

MAX_INTERSECTING_OBJECT_RETRIES = 5
"""Number of retries to sample from asset group if any objects within it collide."""

P_W1_ASSET_SKIPPED = 0.8
"""Probability of skipping a weight 1 asset, when there are only weight 1 assets available.

Avoids the problem of weight 1 assets always appearing in rooms when max_floor_objects
is large.

Note that this number is often compounded, relative to max_floor_objects and
the number of w2 and asset groups available.
"""

P_CHOOSE_EDGE = 0.7
"""Probability of placing an object at the edge of a room.

When sampling a rectangle that is at the edge of the room, this denotes the
probability that the sampled object should be placed at the edge of the rectangle
instead of in the middle.
"""

P_LARGEST_RECTANGLE = 0.8
"""Probability that the largest possible rectangle gets chosen.

Among all possible rectangles with which to place an object, this denotes the
probability of choosing the largest remaining one.
"""

MIN_RECTANGLE_SIDE_SIZE = 0.5
"""The minimum rectangle size per side, in meters, that can be chosen."""

PROCTHOR_INITIALIZATION = dict(branch="main", scene="Procedural")
"Base AI2-THOR initialization parameters for ProcTHOR."

FLOOR_Y = 0
"""Position of the floor in meters."""

OUTDOOR_ROOM_ID = 1
"""The roomId of the entries in the matrix outside of the generated house."""

EMPTY_ROOM_ID = 0
"""The roomId of the entries in the matrix that are empty."""
