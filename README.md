# ThermalDebt

Implementation of **Thermal-Debt Scheduling for Stable On-Device Visual Inference**.

The controller selects a classifier/resolution/rate configuration using interval thermal prediction, finite-session energy reserves, a latest-frame queue, switching cost and discounted thermal exposure. It executes the first action of a bounded candidate plan and replans from observations every second.

## Install

Use Python 3.11 or later in an activated environment:

```bash
python -m pip install -e .
```

## Prepare real visual predictions

Use permitted ImageNet images or another labeled classification dataset. Keep development and evaluation image IDs disjoint. Export pretrained or trained classifiers as TorchScript modules accepting normalized RGB tensors and returning `[batch,classes]` logits. A common class-index mapping is required for all models.

An image CSV contains `id,path,label`. Paths are relative to that CSV. A model JSON specifies the executable configurations:

```json
[
  {"name": "TD-L", "checkpoint": "data/models/low.pt", "resolution": 160},
  {"name": "TD-M", "checkpoint": "data/models/medium.pt", "resolution": 192},
  {"name": "TD-H", "checkpoint": "data/models/high.pt", "resolution": 224}
]
```

```bash
thermaldebt collect --images data/development_images.csv --models data/models.json --output data/development_predictions.csv
thermaldebt collect --images data/evaluation_images.csv --models data/models.json --output data/evaluation_predictions.csv
```

Collection performs actual classifier inference with RGB resizing and ImageNet mean/std normalization. It records image ID, action, true/predicted class, execution duration and initial loading/warmup duration. Use `--device cuda` when appropriate. Existing real prediction logs may use the same CSV schema directly. A classifier requiring another preprocessing convention should be exported with matching input behavior or collected through its native pipeline.

The scheduling method learns development profiles rather than classifier weights. Its expected quality is estimated from development correctness, not filled from paper result tables.

## Calibrate action and thermal profiles

```bash
thermaldebt calibrate --predictions data/development_predictions.csv --output data/profile.json
thermaldebt prepare --predictions data/evaluation_predictions.csv --output data/session --phase-duration 300 --seed 13
```

The default action library uses the paper's declared model inputs: 160/192/224-pixel configurations at 15/10/10 admitted fps; active power 2.0/2.8/4.0 W; service duration 25/45/75 ms; and resident memory 96/160/256 MiB. These are simulation characterization parameters. Development predictions supply actual per-action accuracy. The default 15-ms loading time, switch energy 0.05/0.08/0.12 J, 48-MiB workspace and bounded observation error are explicit implementation conventions. Configurable profiles let you replace them with measured device quantities.

`--measured-latency` uses development p95 execution duration and measured cold loading. `--config` supplies a custom full JSON profile. `--thermal-data` optionally fits nonnegative two-state A/B coefficients from development CSV columns `die_c,body_c,energy_j`, where each row's energy drives the next temperature row at one-second intervals. The fit must be stable. The maximum observed residual per channel supplies the development disturbance envelope. Specify the matching `--ambient` reference.

Default thermal coefficients are the paper's two-state matrix, energy input vector and 0.03/0.01-K disturbance bounds. Power and energy remain distinct: active power multiplies actual execution time, baseline/background power multiplies the one-second slot, and switch energy is charged once. The modeled temperature is the internal die/body state relative to ambient.

Preparation creates six phases with offered rates 15, 20, 30, 30, 20 and 15 fps. Default phases last five minutes. A seeded image-bank replay supplies the requested content. `workload.csv` exposes only the current slot's capture rate/background load to the policy. `requests.npz` stores the corresponding real per-action predictions for the execution/evaluation layer. Development/evaluation ID overlap is rejected at execution.

## Run the scheduler

```bash
thermaldebt run --data data/session --profile data/profile.json --output outputs/full --policy full
```

The controller knows present temperature observations, current requests/rate/background demand, completed work and development profiles. It does not read evaluation labels, future background slots or future image correctness. Prediction logs are accessed by the replay executor only after an action is selected.

The candidate family contains the shifted previous plan, constant actions and single-switch alternatives, capped at 24 plans over five slots. Hard checks include overlapping old/new model residency, loading, deadline, interval thermal propagation and electrical energy. The terminal check propagates the pause continuation to the fixed session end and reserves its baseline/background energy. It does not impose an infinite battery reserve. Pause retains the previously resident classifier for later hot reuse.

Discounted debt uses `Z_next = exp(-1/tau) * Z + max(T-soft_limit,0)` in degree-Celsius seconds. With the linear objective, inherited debt is common to candidate rankings; predicted new exceedances and their persistence affect selection. The implementation does not add a hard debt constraint or treat debt as stored heat.

The executor processes capture and admission events in time order. Captures arriving exactly at the current decision boundary enter the latest-frame queue before the controller reads pending age. A busy processor has one pending latest frame; a new capture supersedes that pending frame. The first admission follows actual planning, loading and preceding-job completion; subsequent admission opportunities respect the selected rate. Admission uses expected capture-to-result delay, and realized late/interrupted requests remain in the denominator. Requested actions appear in `history.json`; actually executed classifier choices appear per request. Cold transitions block execution and charge their memory/energy even when they reduce delivered work. Characterized action latency plus loading must fit the one-second control slot. `--measured-execution` uses real per-image timing logs instead of fixed profile service times, subject to that subsecond characterization. Pause keeps the cached model resident and advances its dwell interval, so hot resumption does not reset model residency or incur another cold load.

| Option | Default | Meaning |
|---|---:|---|
| `--ambient`, `--soft-temperature`, `--hard-temperature` | 32, 38, 45 | Ambient, soft internal budget and hard internal threshold in °C |
| `--power-budget` | 3 | Average-power allocation defining initial finite-session Wh |
| `--memory-budget` | 512 | Modeled resident/workspace MiB cap |
| `--deadline` | 0.1 | Capture-to-result deadline in seconds |
| `--horizon`, `--candidates` | 5, 24 | Remaining-horizon cap and plan-count cap |
| `--minimum-dwell` | 5 | Model-switch dwell slots |
| `--planning-allowance` | 0.01 | Predicted planning duration for admission; actual duration is also charged |
| `--debt-tau`, `--debt-weight` | 120, 0.2 | Exposure decay and normalized penalty |
| `--energy-weight`, `--switch-weight` | 0.05, 0.02 | Dimensionless planning penalties; frame age enters deadline admission |
| `--quality-floor` | 0 | Minimum development accuracy for admitted classifiers |
| `--residual-count`, `--recovery-count` | 2, 5 | Out/in-envelope observations for recovery transitions |
| `--thermal-multiplier` | 1 | Executor heat-input mismatch with the unchanged controller predictor |

Repeated out-of-envelope temperature observations restrict the planner to low demand or pause, widen the declared disturbance envelope and invalidate the previous plan. Recovery ends only after consecutive compatible observations and an admissible pause continuation under the expanded recovery envelope, including the finite-session thermal/energy reserve. When no action is modeled feasible, the executor pauses and records infeasibility rather than labeling the state safe.

## Baselines, ablations and stress

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

Fixed policies expose the low/high resource extremes and continue their fixed configuration even when the predictive checks fail; violations remain reported. Greedy/threshold controls consider the current slot. Energy MPC removes debt pricing, temperature MPC additionally removes energy pricing, and `no_debt` is the corresponding full-controller debt ablation. `no_switching` removes dwell and switching penalties while the executor still charges actual transitions. `no_queue` removes deadline admission in both planning and execution. `no_monitor` removes recovery activation. Apply the same `--thermal-multiplier` to the full policy for the mismatch pair.

Sweep `--power-budget`, ambient 28/32/36, phase duration, disturbance bounds and classifier profiles while sharing the same offered request bank. Choose energy-matching penalty settings on separate development sessions before comparing session service at matched consumption.

## Outputs

`metrics.json` reports timely correct outputs/s, the tenth percentile over nonoverlapping 30-second output windows, dropped/expired fraction over all offered requests, completed-output accuracy, p95 delay, seconds above soft/hard thresholds, total Wh, peak modeled resident memory, switches, infeasible slots and planning duration/overruns. A final partial 30-second window uses its actual duration. Accuracy and delay are null when nothing completes.

`requests.json` preserves every offered request, including superseded, expired and interrupted ones. `history.json` contains modeled thermal state, debt, energy, requested actions and recovery state. Thermal/memory/energy quantities follow the declared replay model; classifier correctness comes from real image predictions. No synthetic accuracy samples or prefilled experiment scores are used.

```bash
thermaldebt plot --runs outputs/full outputs/high outputs/energy outputs/temperature --output outputs/figures
```

The plots show temperature, debt, remaining energy and the service–energy frontier computed from completed replay files.
