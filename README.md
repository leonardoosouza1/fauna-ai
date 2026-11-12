# iara-brain (Project Fauna)

A multi-agent ecosystem simulator in JAX. Agents with liquid neural networks, Closed-form Continuous-time
(CfC) cells wired as a Neural Circuit Policy (NCP) inspired by *C. elegans*, sense, eat, hunt, reproduce
and evolve in a continuous 2D world. A Python socket server streams the world state to a React/three.js
viewer. The target hardware is a consumer AMD GPU running ROCm.

**Status: early scaffold (February 2026), no experiments recorded.** The modules and their unit tests
exist. Nobody has run or measured evolution end to end yet, so this README contains no result numbers.
Active research on CfC/neuromodulated agents has since moved to other repositories in the same program.

## What is implemented

| area | module (`backend/src/`) | contents |
|---|---|---|
| brain | `systems/brain/cfc_cell.py`, `ncp.py`, `policy.py` | CfC cell (Hasani et al. 2022) in Flax; sparse NCP wiring (sensory → inter → command → motor); `AgentBrain` = MLP encoder + CfC layers + tanh motor head |
| workspace | `systems/brain/workspace.py` | Global-Workspace-style cross-attention / consensus fusion and a modulator |
| plasticity | `systems/brain/plasticity.py` | Hebbian update, Oja's rule, homeostatic scaling |
| world | `systems/physics.py`, `sensors.py`, `environment.py` | forces, speed clamp, bounds, neighbours, collisions; proprioception and food-gradient sensors; seasons, vegetation, scent map |
| ecology | `systems/metabolism.py`, `predation.py`, `core/species.py` | energy cost, starvation, reproduction threshold; Holling type II/III functional responses; multi-species prey matrix |
| learning | `systems/evolution.py`, `training.py`, `core/genome.py` | fitness, tournament / elite selection, genome mutation and crossover; PPO loss, GAE, rollout buffer |
| infra | `engine/engine.py`, `service/`, `adapters/checkpoint.py`, `infrastructure/rocm_check.py` | simulation engine, Socket.IO API (`sim:start/pause/resume/reset/config`, `agent:inspect`), checkpoints, ROCm backend check |

The repository has 75 unit tests in `backend/tests/` (CfC cell, NCP, state, and phase 2–4 systems). I did
not run them while writing this README.

## How it fits together

```
 world state (JAX pytrees, up to --max-agents)
    |  sensors: proprioception, food gradient, scent
    v
 AgentBrain:  MLP encoder -> CfC inter layer -> CfC command layer -> tanh motor
    |  physics, metabolism, predation, vegetation, seasons
    v
 engine step --Socket.IO--> React + three.js viewer (frontend/)
    |
 evolution (selection, mutation, crossover) / PPO
```

## What didn't work (and why)

Nothing has been measured, so there are no negatives yet. The `main.py` service starts the engine and the
socket API, but it has not yet been wired into a recorded evolution run with baselines.

## Hardware / stack

- Developed for an AMD Radeon RX 6750 XT (Navi 22, gfx1031, 12 GB) running as gfx1030 via
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- JAX / Flax / Optax in the `rocm/jax` Docker image (`docker/`), with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` to leave room for the display. The CPU-only `jax[cpu]` also works
  for the tests.
- Frontend: React 19, Vite, @react-three/fiber, Zustand, Socket.IO client.

## Build & run

Backend (CPU):

```bash
cd backend
pip install -r requirements.txt jax[cpu]   # or: pip install -e ".[dev]" plus python-socketio aiohttp
python -m pytest tests/ -v
python -m src.infrastructure.rocm_check    # which JAX backend is active
python -m src.main --port 8765 --max-agents 500 --grid-size 128 --seed 42
```

GPU (ROCm, Docker):

```bash
cd docker
docker compose up -d          # builds from rocm/jax, mounts ../backend, runs src.main on port 8765
docker compose exec fauna bash
```

Frontend:

```bash
cd frontend
yarn install
VITE_API_URL=http://localhost:8765 yarn dev
```

## Roadmap

- [x] Foundation: types, CfC cell, NCP wiring, tests
- [~] Physics, sensors, metabolism, predation, environment: code and unit tests in place, no measured runs
- [~] Global workspace, Hebbian plasticity, PPO: code and unit tests in place, no measured runs
- [ ] First recorded experiment: multi-seed evolution against a random-policy and an MLP-policy baseline
- [ ] Scale with `vmap` over many agents and measure ticks/s on the GPU

The `*.txt` files at the repository root are background design notes (in Portuguese) written before the
code. They are not documentation of what was built.

Author: [leonardoosouza1](https://github.com/leonardoosouza1)

License: TBD (`pyproject.toml` declares MIT, but no LICENSE file is committed yet).
