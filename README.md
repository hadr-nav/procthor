<h1 align="center">
  🏘️ ProcTHOR
</h1>

<h3 align="center"><em>Scaling Embodied AI by Procedurally Generating Interactive 3D Environments</em></h3>

![procthor-cover(1)](https://user-images.githubusercontent.com/28768645/191896912-58a2234a-ed50-40b6-a534-348db7260756.jpg)

ProcTHOR procedurally generates interactive, diverse, and semantically plausible houses that are compatible with AI2-THOR.

## Example

Install `procthor` with PyPi:
```bash
pip install procthor
```

And then run the example script to generate a new house:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
python scripts/example.py
```

## Multi-floor scenes

With the schema-2 AI2-THOR integration installed, generate a two- or three-floor
residential scene by passing `num_floors`:

```python
from ai2thor.controller import Controller
from procthor.generation import HouseGenerator

# This must use the patched AI2-THOR schema-2 build linked below.
controller = Controller(
    local_executable_path="/path/to/patched/ai2thor-build",
    scene="Procedural",
    quality="Low",
)
generator = HouseGenerator(
    split="train", seed=1234, num_floors=2, controller=controller
)
house, _ = generator.sample()
```

Explicit per-floor layouts are available through `FloorSpec` and `HouseSpec`.
Omitting multi-floor arguments—or passing `num_floors=1`—keeps the exact legacy
schema-1 single-floor path. See the [multi-floor guide](docs/multifloor.md) for
the engine patch, fixed stair geometry, schema fields, and complete examples.

## Citation

This code is used to generate houses for the [ProcTHOR](https://procthor.allenai.org/) paper:

```bibtex
@inproceedings{procthor,
  author={Matt Deitke and Eli VanderBilt and Alvaro Herrasti and
          Luca Weihs and Jordi Salvador and Kiana Ehsani and
          Winson Han and Eric Kolve and Ali Farhadi and
          Aniruddha Kembhavi and Roozbeh Mottaghi},
  title={{ProcTHOR: Large-Scale Embodied AI Using Procedural Generation}},
  booktitle={NeurIPS},
  year={2022},
  note={Outstanding Paper Award}
}
```

## 👋 Our Team

ProcTHOR is an open-source project built by the [PRIOR team](//prior.allenai.org) at the [Allen Institute for AI](//allenai.org) (AI2).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

<br />

<a href="//prior.allenai.org">
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/allenai/ai2thor/main/doc/static/ai2-prior.svg" /></p>
</a>
