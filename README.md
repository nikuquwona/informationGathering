# LocalGP · Information gathering

Multi-agent aerial deployment in unknown environments. This repository preserves a 2024 graduation-project implementation and its original experiment archive, with a new offline replay and tested GP corrections.

> **Status:** research prototype. The 2024 code is related to the LocalGP–MAPPO paper; it is not a verified reproduction of the final published method. Historical results have not been regenerated with the corrected code.

## Explore the archive — no dependencies

From the repository root:

```sh
python3 tools/build_replay.py
```

Open `output/replay.html` in a browser. It works offline and includes:

- 45 complete, three-agent archived runs, with playback and frame scrubbing.
- Recorded travel distance, displacement, and sampled inter-agent separation.
- Available final GP mean / standard-deviation snapshots, explicitly kept static during playback.
- A JSON download per run and source-file SHA-256 provenance.

The exporter discovers the archive rather than hard-coding the run count. It validates matrices and equal-length trajectories. It does not invent user positions, coverage, throughput, intermediate GP maps, or algorithm labels for undocumented run numbers. The navigation grid is the repository's current grid; historical per-run configurations were not saved.

## Run the regression tests

Python 3.10 or newer, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
MPLBACKEND=Agg python -m pytest -q
```

A GitHub Actions workflow runs these tests on Python 3.10 and 3.12 and checks that the archive exports. CUDA and model checkpoints are not needed for these checks. `requirements-verified.txt` records the exact scientific/test environment used locally; project dependency ranges live in `pyproject.toml`.

## What changed

- Global GP uncertainty now uses the predicted **standard deviation**, rather than accidentally reusing the mean.
- GP prediction / reward maps use floating-point storage even when the navigation grid is integer-valued.
- Repeated spatial observations use the **latest measurement**. This is an explicit choice for the moving-user environment; older values at unvisited locations still remain, so this is not a spatiotemporal GP.
- Local consensus weights are normalized over experts whose prediction region actually includes a point. Uncovered points retain the kernel prior uncertainty.
- Constructor and reset priors agree; resetting clears reward deltas and the fitted GP. Empty measurement batches and invalid inputs have regression coverage.
- Observation assembly supports arbitrary grid shapes and agent counts, preserving the original identity channels for three agents. Observations use `float32`.
- Environment imports support Gymnasium without requiring PyTorch. This does **not** migrate the legacy environment to Gymnasium's reset/step API.

These corrections change the observations and potentially the rewards. Historical checkpoints and curves must be re-evaluated; there is no claim of an improved trained policy yet.

## Research reference

H. Liu et al., *Autonomous Deployment of Aerial Base Station Without Network-Side Assistance in Emergency Scenarios Based on Multi-Agent Deep Reinforcement Learning*, IEEE TNSM, vol. 23, 2026. DOI: [10.1109/TNSM.2025.3603875](https://doi.org/10.1109/TNSM.2025.3603875).

See [the validation record](docs/validation.md). Read [the implementation audit and next steps](docs/research-notes.md) before running new experiments. The local reference PDF was used for comparison and is not redistributed here.

## Repository guide

- `forth/GPmodel.py`: local and global GP estimators.
- `forth/InformationGatheringEnvironment.py`: legacy multi-agent environment.
- `forth/ppo_discrete.py`, `forth/PPO_dis_main.py`: legacy shared-policy PPO implementation and training loop.
- `forth/groundtruth_move*.py`: fixed user layouts and moving signal fields.
- `forth/path/`, `mu_sig_map/`, `log/`: original experiment archives, preserved unchanged.
- `tools/build_replay.py`, `tools/replay_template.html`: dependency-free offline replay exporter.
- `tests/`: GP, observation, and archive regression tests.

The old training entry points still contain hard-coded CUDA, configuration and rollout issues described in the audit. Installing `.[training]` only installs optional dependencies; it does not certify that the training pipeline is ready or faithfully implements MAPPO.
