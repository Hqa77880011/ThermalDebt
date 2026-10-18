# ThermalDebt

Implementation of **Thermal-Debt Scheduling for Stable On-Device Visual Inference**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0, pillow 12.3.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Render circle/square/triangle images with disjoint development/evaluation IDs and compute pixel-classifier predictions at three resolutions:

```bash
python finite_data.py generate --output data/generated --seed 13
python finite_data.py profile --source data/generated
thermaldebt calibrate --predictions data/generated/development.csv --config data/generated/finite_profile.json --output data/profile.json
thermaldebt prepare --predictions data/generated/evaluation.csv --output data/session --phase-duration 300 --seed 13
```

The profile declares analytic latency/load/thermal costs; these are model assumptions. External image/prediction logs use native `collect`/`calibrate` with independent development/evaluation IDs.

## Training and inference

```bash
thermaldebt run --data data/session --profile data/profile.json --output outputs/full --policy full
```

## Evaluation

Replay records completed predictions, deadlines, energy/debt and modeled thermal states.

## Comparisons and ablations

```bash
thermaldebt run --data data/session --profile data/profile.json --output outputs/low --policy fixed_low
thermaldebt run --data data/session --profile data/profile.json --output outputs/high --policy fixed_high
thermaldebt run --data data/session --profile data/profile.json --output outputs/greedy --policy greedy
thermaldebt run --data data/session --profile data/profile.json --output outputs/threshold --policy threshold
thermaldebt run --data data/session --profile data/profile.json --output outputs/energy --policy energy_mpc
thermaldebt run --data data/session --profile data/profile.json --output outputs/temperature --policy temperature_mpc
thermaldebt run --data data/session --profile data/profile.json --output outputs/no_switch --policy no_switching
thermaldebt run --data data/session --profile data/profile.json --output outputs/no_queue --policy no_queue
thermaldebt run --data data/session --profile data/profile.json --output outputs/no_monitor --policy no_monitor --thermal-multiplier 1.2
```

## Outputs

`metrics.json`, `requests.json` and `history.json` retain service metrics, offered requests and modeled state/actions.

```bash
thermaldebt plot --runs outputs/full outputs/high outputs/energy outputs/temperature --output outputs/figures
```

`reproduce.py` combines the finite preparation and full/energy_mpc/fixed_low comparison in a fresh `--output`; `--plan` only prints commands.
