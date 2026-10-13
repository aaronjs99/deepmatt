# DeepMAtt

**Deep Manifold Attention: cislunar formation-flying research software**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](paper/main.pdf)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepMAtt is a research codebase for experiments in normalized circular
restricted three-body problem (CR3BP) dynamics, cislunar formation flying,
guidance and control, and learning-based experiment prototypes. It is not
flight software and its archived paper figures are not a substitute for a fresh
reproduction of a stated result.

## What is reusable

The computational core is deliberately independent of a single celestial pair.
Use `CR3BPSystem` to supply a normalized mass parameter and optional reporting
scales; `EARTH_MOON_SYSTEM` remains the default for existing Earth--Moon
experiments.

```python
from deepmatt.core.system import CR3BPSystem
from deepmatt.dynamics.cr3bp import cr3bp_dynamics

synthetic_system = CR3BPSystem(name="synthetic", mass_parameter=0.2)
state_derivative = cr3bp_dynamics(0.0, state, system=synthetic_system)
```

Guidance (`compute_stm`, `find_natural_formation`), relative navigation (`EKF`),
and `SDREController` also accept a keyword-only `system`; existing calls retain
the Earth--Moon default. Supply a reference state appropriate to the chosen
system rather than reusing the Earth--Moon halo initial condition.

`gnc/control/lqr.py` owns the shared acceleration input map and Riccati feedback
calculation. Experiment-specific cost matrices and solver-failure fallbacks stay
with their callers.

The repository separates:

- normalized CR3BP dynamics and local linearization;
- analysis, transfer, formation, navigation, and control experiments;
- optional learning and optimization backends; and
- manuscript assets and historical result files.

## Install

```bash
git clone git@github.com:aaronjs99/cislune.git
cd cislune
python -m pip install -e .
```

Install only the optional capabilities you intend to use:

```bash
# GEKKO-backed optimization experiments
python -m pip install -e ".[optimization]"

# PPO and Transformer experiments
python -m pip install -e ".[learning]"

# Dataframe-backed reporting experiments
python -m pip install -e ".[reporting]"
```

## Run an experiment

### Reproducible GNC simulation

Run a complete guidance, relative navigation, and formation-control experiment
with the basic installation:

```bash
python run.py --mode gnc --config config/gnc.yaml

# Quick headless run; command-line values override the configuration.
python run.py --mode gnc --duration 0.025 --dt 0.01 --seed 7 --no-plot

# The installed entry point supports the same options.
deepmatt --mode gnc --duration 0.3 --seed 7
```

Each invocation creates a unique folder under `results/runs/`. Use
`--output-dir <folder>` to choose another parent directory. Existing runs are
never overwritten. A run contains:

| File | Contents |
| --- | --- |
| `config.json` | Every resolved input, including defaults and the system definition; accepted by `--config` for replay |
| `summary.json` | Completion status, final/RMS/maximum tracking errors, control usage, saturation count, and Python/NumPy/SciPy versions |
| `trajectory.csv` | Time, absolute chief/reference/deputy states, relative navigation estimates, errors, cumulative delta-V, and interval controls |
| `states.npz` | The same complete numerical histories plus navigation covariance matrices; load with `numpy.load(..., allow_pickle=False)` |
| `performance.png` | Tracking, navigation, acceleration, and cumulative delta-V plots; omitted with `--no-plot` |

To repeat a run, pass its `config.json` to `--config`. The same settings and
seed reproduce numerical histories within the same software environment.
Version changes may change floating-point results. Saved runs are local
artifacts; publishing or archiving them is a separate step.

Configuration is safe YAML or JSON. Unknown settings, duplicate keys,
malformed states, non-finite numbers, invalid noise/limits, and more than
100000 integration intervals are rejected before simulation. See
`config/gnc.yaml` for the maintained example; `GNCConfig` lists every setting.
GNC-specific command options are rejected for legacy modes rather than silently
ignored. Invalid inputs exit with status 2, numerical failures with status 1,
and interruption with status 130. Failed saves retain `status: incomplete`;
`completed` is written only after every requested artifact has been saved.

All inputs and raw numerical histories use **normalized rotating-frame CR3BP
units**. State order is `[x, y, z, vx, vy, vz]`. For the default Earth--Moon
system, one length unit is 384400 km and one time unit is 375200 s. A position
measurement standard deviation of `1e-8` therefore means approximately 3.844 m.
Physical position/velocity/delta-V summaries are derived from the system's
reporting scales; unavailable conversions are JSON `null`. `process_noise_variance`
is a discrete covariance added per EKF prediction, so changing the sample step
also changes the accumulation of that configured process uncertainty.

The reference deputy follows the full unforced CR3BP equations from
`relative_offset`. The controlled deputy starts with `initial_relative_error`
and uses sampled LQR feedback around the chief, a position-only relative EKF,
and a noisy actuator. Chief, reference, and deputy share a checked nonlinear
integration over each interval. The EKF uses a frozen local Jacobian with exact
zero-order-hold discretization and a Joseph covariance update. The estimator
knows the limited command, not the actuator's true noise. Actual acceleration
is norm-limited after noise and bias. Position and velocity errors are reported
separately. Both initial and terminal samples are recorded; a shortened last
interval reaches the requested duration exactly. CSV controls apply from a row
to the following row; control cells at the terminal sample are blank.

This explicit unforced reference is not guaranteed to be a bounded or periodic
natural formation. The simulation uses ideal chief-state knowledge and a
synthetic relative-position sensor; it does not model real sensor links,
third-body perturbations, spacecraft mass/propellant, or flight operations.
Delta-V is integrated acceleration magnitude, not a propellant-mass estimate.

For programmatic use, `gnc/config.py` owns validation,
`gnc/integrated_sim.py:run_gnc_simulation` returns numerical results without file
or plotting side effects, and `output/gnc.py:save_gnc_result` owns persistence and
headless plotting. Both the CLI and the legacy `integrated_gnc_sim()` entry use
this same pipeline. No learning or optimization backend is imported by GNC.

### Other experiments

The command layer loads a mode only when selected, so a basic installation can
still inspect the available interface without all optional backends.

```bash
python run.py --mode formation
python run.py --mode stability
python run.py --mode train_ppo
```

Available modes:

| Mode | Purpose |
| --- | --- |
| `gnc` | Configurable, seeded GNC simulation with complete run bundles |
| `adversary` | Adversarial analysis experiment |
| `halo`, `uncontrolled_halo`, `lissajous`, `stability`, `stationkeeping`, `poincare` | Dynamics and orbit-analysis experiments |
| `transfer_earth`, `transfer_leo`, `transfer_fast` | Transfer-design experiments |
| `formation`, `formation_rigid`, `formation_optimize` | Formation-control experiments |
| `train_ppo`, `train_transformer` | Optional learning experiments |
| `benchmark_hybrid` | Hybrid benchmark experiment |
| `generate_results` | Runs the documented result batch; inspect and reproduce each output before relying on it |

## Layout

```text
src/deepmatt/       Python package
  core/             System definitions and shared numerical utilities
  dynamics/         Normalized CR3BP model and Jacobian
  analysis/         Orbit and transfer experiments
  formation/, gnc/  Formation, guidance, navigation, and control experiments
  learning/         Optional learning experiments
  validation/       Benchmark and robustness experiments
paper/              Manuscript source and archival figures
results/            Generated or archived experiment outputs
```

## Scope

The package exposes simulation and analysis interfaces only. Deployment,
hardware-in-the-loop testing, and flight validation require separate evidence,
reviews, and operational controls.

## Development checks

```bash
python -m pip install -e ".[development]"
python -m pytest -c pyproject.toml tests
ruff check src/deepmatt/cli.py src/deepmatt/gnc/config.py src/deepmatt/gnc/integrated_sim.py src/deepmatt/gnc/navigation.py src/deepmatt/gnc/actuators.py src/deepmatt/output/gnc.py tests
```

The tests cover the real short-run workflow, configuration and CLI failures,
replay, independent propagation checks, exact terminal times, isolated seeds,
physical-unit reporting, actuator saturation, estimator accuracy/covariance,
and incomplete-save handling. These software checks do not validate spacecraft
performance. No type checker or repository-wide lint gate is currently configured.

## License

MIT License. See [LICENSE](LICENSE).
