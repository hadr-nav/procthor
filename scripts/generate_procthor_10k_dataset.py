import os
import random
from datetime import datetime
from multiprocessing import Pool, Value
from time import sleep

import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from procthor.generation import (
    HouseGenerator,
    PROCTHOR10K_ROOM_SPEC_SAMPLER,
    PROCTHOR_MULTIFLOOR_HOUSE_SPEC_SAMPLER,
)

from procthor.generation.multifloor_generation import (
    ensure_schema2_controller,
)
from procthor.utils.types import InvalidFloorplan, InvalidMultiFloorPlan

print("Starting at", datetime.now())


split = "train"
processes = 60
counter = Value("i", 5400 - processes)
n_gpus = torch.cuda.device_count()

# n_gpus = 1
# house_generators = [
#     HouseGenerator(
#         split=split,
#         controller=Controller(
#             x_display=f":0.{i}", quality="Low", **PROCTHOR_INITIALIZATION
#         ),
#     )
#     for i in range(n_gpus)
# ]

# house_generator = HouseGenerator(split=split)
controllers = {}

FLOOR_COUNT_POPULATION = [1, 2, 3]
FLOOR_COUNT_WEIGHTS = [0.50, 0.35, 0.15]
"""Default dataset mix: 50% one floor, 35% two floors, and 15% three floors."""


SCHEMA2_EXECUTABLE_PATH = os.environ.get("PROCTHOR_SCHEMA2_EXECUTABLE")
if not SCHEMA2_EXECUTABLE_PATH:
    raise RuntimeError(
        "Set PROCTHOR_SCHEMA2_EXECUTABLE to the patched AI2-THOR executable "
        "before generating the default mixed-floor dataset."
    )


def generate_house(i: int) -> None:
    global counter
    global n_gpus

    pid = os.getpid()
    print(f"Using {pid}")
    if pid not in controllers:
        gpu_i = pid % n_gpus
        controller = Controller(
            gpu_device=gpu_i,
            platform=CloudRendering,
            quality="Low",
            scene="Procedural",
            local_executable_path=SCHEMA2_EXECUTABLE_PATH,
        )
        ensure_schema2_controller(controller)
        controllers[pid] = controller

    # Choose the floor count and complete spec once so retries do not bias the
    # requested floor distribution or room composition.
    floor_count = random.choices(
        FLOOR_COUNT_POPULATION, weights=FLOOR_COUNT_WEIGHTS, k=1
    )[0]
    if floor_count == 1:
        room_spec = PROCTHOR10K_ROOM_SPEC_SAMPLER.sample()
        house_generator = HouseGenerator(
            controller=controllers[pid], split=split, room_spec=room_spec
        )
    else:
        house_spec = PROCTHOR_MULTIFLOOR_HOUSE_SPEC_SAMPLER.sample(
            num_floors=floor_count
        )
        house_generator = HouseGenerator(
            controller=controllers[pid], split=split, house_spec=house_spec
        )

    while True:
        try:
            house, _ = house_generator.sample()
        except (InvalidFloorplan, InvalidMultiFloorPlan):
            # Retry bounded geometry failures without resampling the requested
            # floor count, RoomSpec, or HouseSpec.
            continue
        house.validate(house_generator.controller)
        if house.data["metadata"]["warnings"]:
            # Retry geometry/furnishing with the same RoomSpec or HouseSpec.
            continue
        break

    with counter.get_lock():
        counter.value += 1
    sleep(0.1)

    print(i, counter.value)

    house.to_json(f"big-dataset/{split}/{pid}-{counter.value}.json.gz", compressed=True)


with Pool(processes=processes) as p:
    r = p.map(generate_house, range(10_000_000_000))
