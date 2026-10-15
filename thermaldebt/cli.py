import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .control import Scheduler


def collect(args):
    import torch
    from PIL import Image

    images = pd.read_csv(args.images)
    models = json.loads(Path(args.models).read_text(encoding="utf-8"))
    rows = []
    for specification in models:
        start = time.perf_counter()
        model = torch.jit.load(specification["checkpoint"], map_location=args.device).eval()
        size = int(specification["resolution"])
        with torch.no_grad():
            model(torch.zeros(1, 3, size, size, device=args.device))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        loading = time.perf_counter() - start
        for row in images.itertuples(index=False):
            image = Image.open(Path(args.images).parent / row.path).convert("RGB").resize((size, size))
            array = np.asarray(image, dtype=np.float32).copy() / 255
            tensor = torch.as_tensor(array, device=args.device).permute(2, 0, 1)[None]
            mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            scale = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            tensor = (tensor - mean) / scale
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            rows.append(dict(id=str(row.id), action=specification["name"], label=int(row.label), prediction=int(logits.argmax(1)),
                             latency_s=elapsed, load_s=loading))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


def calibrate(args):
    source = Path(args.config) if args.config else Path(__file__).with_name("defaults.json")
    config = json.loads(source.read_text(encoding="utf-8"))
    data = pd.read_csv(args.predictions, dtype={"id": str})
    if data.duplicated(["id", "action"]).any():
        raise ValueError("Development image/action pairs must be unique")
    for action in config["actions"][:-1]:
        rows = data[data["action"] == action["name"]]
        if rows.empty:
            raise ValueError(f"Missing development predictions for {action['name']}")
        action["accuracy"] = float((rows["prediction"] == rows["label"]).mean())
        if args.measured_latency:
            if (rows["latency_s"] <= 0).any():
                raise ValueError("Measured latency must be positive")
            action["latency_s"] = float(rows["latency_s"].quantile(0.95))
            if "load_s" in rows:
                action["load_s"] = float(rows["load_s"].max())
    if args.thermal_data:
        thermal = pd.read_csv(args.thermal_data)
        state = thermal[["die_c", "body_c"]].to_numpy() - args.ambient
        power = thermal["energy_j"].to_numpy()[:-1]
        design = np.column_stack([state[:-1], power])
        coefficients = np.stack([nnls(design, state[1:, channel])[0] for channel in range(2)])
        matrix, vector = coefficients[:, :2], coefficients[:, 2]
        if max(abs(np.linalg.eigvals(matrix))) >= 1:
            raise ValueError("Development fit is not stable; inspect sensor reference and calibration trace")
        residual = state[1:] - design @ coefficients.T
        config.update(A=matrix.tolist(), B=vector.tolist(), disturbance=np.maximum(np.max(np.abs(residual), 0), 1e-6).tolist())
    config["development_ids"] = sorted(data["id"].unique().tolist())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2), encoding="utf-8")


def prepare(args):
    table = pd.read_csv(args.predictions, dtype={"id": str})
    if table.duplicated(["id", "action"]).any() or (table.groupby("id")["label"].nunique() != 1).any():
        raise ValueError("Unique image/action pairs and one consistent label per image required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actions = sorted(table["action"].unique())
    ids = sorted(table["id"].unique())
    prediction = table.pivot(index="id", columns="action", values="prediction").reindex(index=ids, columns=actions)
    latency = table.pivot(index="id", columns="action", values="latency_s").reindex(index=ids, columns=actions)
    if not np.isfinite(prediction.to_numpy()).all() or not np.isfinite(latency.to_numpy()).all() or (latency <= 0).any().any():
        raise ValueError("Each evaluation image needs a positive-latency prediction from every action")
    labels = table.groupby("id")["label"].first().reindex(ids).to_numpy()
    rates = [15, 20, 30, 30, 20, 15]
    backgrounds = [0, 0.2, 0.5, 1.5, 0.2, 0]
    rng = np.random.default_rng(args.seed)
    capture, indices, slots = [], [], []
    for phase, (rate, background) in enumerate(zip(rates, backgrounds)):
        for offset in range(args.phase_duration):
            timestamp = phase * args.phase_duration + offset
            capture.extend(timestamp + np.arange(rate) / rate)
            indices.extend(rng.integers(len(ids), size=rate))
            slots.append(dict(time=timestamp, offered_rate=rate, background_w=background, phase=phase))
    np.savez_compressed(output / "requests.npz", capture=np.asarray(capture), image=np.asarray(indices),
                        ids=np.asarray(ids), labels=labels, predictions=prediction.to_numpy(dtype=int),
                        latency=latency.to_numpy(dtype=float), actions=np.asarray(actions))
    pd.DataFrame(slots).to_csv(output / "workload.csv", index=False)


def run(args):
    config = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    root = Path(args.data)
    with np.load(root / "requests.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if set(config["development_ids"]) & set(data["ids"].tolist()):
        raise ValueError("Development and evaluation images overlap")
    workload = pd.read_csv(root / "workload.csv")
    duration = len(workload)
    if not np.array_equal(workload["time"], np.arange(duration)):
        raise ValueError("Workload uses consecutive one-second slots starting at zero")
    lookup = {name: index for index, name in enumerate(data["actions"].tolist())}
    for action in config["actions"][:-1]:
        if action["name"] not in lookup:
            raise ValueError("Action library and evaluation predictions do not match")
    controller = Scheduler(config, duration, args)
    if args.measured_execution and (data["latency"] >= 1).any():
        raise ValueError("Measured execution durations must fit the characterized one-second slot")
    true_state = np.zeros(2)
    rng = np.random.default_rng(args.seed)
    captures, image_ids = data["capture"], data["image"]
    requests = [dict(capture=float(capture), image_id=str(data["ids"][image]), action=None, output=False,
                     correct=False, completion=None, status="pending") for capture, image in zip(captures, image_ids)]
    pending, busy_until, ready, resident, peak_memory = None, 0.0, 0.0, controller.pause, 0.0
    jobs, history, infeasible, overruns, switches = [], [], 0, 0, 0
    total_energy, hot_seconds, hard_seconds = 0.0, 0, 0
    for slot in workload.itertuples(index=False):
        timestamp = int(slot.time)
        temperature = true_state + args.ambient
        noise = rng.uniform(-1, 1, 2) * np.asarray(config["observation_error"])
        controller.observe(temperature + noise)
        boundary_captures = np.flatnonzero(captures == timestamp)
        for index in boundary_captures:
            if pending is not None:
                requests[pending]["status"] = "dropped_superseded"
            pending = int(index)
        age = max(0, timestamp - captures[pending]) if pending is not None else 0
        action, diagnostic = controller.select(duration - timestamp, age, slot.offered_rate, slot.background_w)
        infeasible += int(not diagnostic["feasible"])
        overruns += int(diagnostic["planning_seconds"] > args.planning_allowance)
        specification = config["actions"][action]
        loading, switching = controller.transition(resident, action)
        slot_memory = controller.memory(resident, action)
        if action != controller.pause and action != resident:
            switches += 1
            ready = max(timestamp + diagnostic["planning_seconds"], busy_until) + loading
            peak_memory = max(peak_memory, slot_memory)
            resident = action
        else:
            ready = max(ready, timestamp + diagnostic["planning_seconds"])
            peak_memory = max(peak_memory, slot_memory)
        indexes = np.flatnonzero((captures > timestamp) & (captures < timestamp + 1))
        events = [(float(captures[index]), 0, int(index)) for index in indexes]
        if action != controller.pause and specification["fps"] > 0:
            first_admission = max(timestamp + diagnostic["planning_seconds"], ready, busy_until)
            events.extend((event_time, 1, -1) for event_time in np.arange(first_admission, timestamp + 1, 1 / specification["fps"]))
        for event_time, kind, index in sorted(events):
            if kind == 0:
                if pending is not None:
                    requests[pending]["status"] = "dropped_superseded"
                pending = index
            elif pending is not None and event_time >= max(busy_until, ready):
                predicted_delay = event_time - captures[pending] + specification["latency_s"]
                if args.policy != "no_queue" and predicted_delay > args.deadline:
                    requests[pending]["status"] = "dropped_expired"
                    pending = None
                    continue
                image = int(image_ids[pending])
                action_column = lookup[specification["name"]]
                elapsed = float(data["latency"][image, action_column]) if args.measured_execution else specification["latency_s"]
                busy_until = event_time + elapsed
                requests[pending].update(action=specification["name"], completion=busy_until,
                                         output=busy_until < duration,
                                         correct=bool(data["predictions"][image, action_column] == data["labels"][image]),
                                         status="completed" if busy_until < duration else "interrupted")
                jobs.append((event_time, busy_until, specification["active_w"]))
                pending = None
        active_energy = sum(max(0, min(end, timestamp + 1) - max(start, timestamp)) * power for start, end, power in jobs if end > timestamp)
        slot_energy = config["base_w"] + slot.background_w + active_energy + switching
        total_energy += slot_energy
        hot_seconds += int(temperature[0] > args.soft_temperature)
        hard_seconds += int(temperature[0] > args.hard_temperature)
        history.append(dict(time=timestamp, phase=int(slot.phase), action=specification["name"],
                            die_c=float(temperature[0]), body_c=float(temperature[1]), debt=controller.debt,
                            remaining_wh=controller.energy, slot_j=slot_energy, recovery=controller.recovery,
                            pending_age=age, slot_peak_resident_memory_mib=slot_memory, **diagnostic))
        controller.feedback(action, slot_energy, temperature, slot.background_w)
        disturbance = rng.uniform(-1, 1, 2) * np.asarray(config["disturbance"])
        true_state = controller.a @ true_state + controller.b * args.thermal_multiplier * slot_energy + disturbance
    for request in requests:
        if request["status"] == "pending":
            request["status"] = "dropped_session_end"
    delivered = [row for row in requests if row["output"]]
    useful = [row for row in delivered if row["correct"] and row["completion"] - row["capture"] <= args.deadline]
    delays = [row["completion"] - row["capture"] for row in delivered]
    windows = []
    for start in range(0, duration, 30):
        end = min(duration, start + 30)
        windows.append(sum(start <= row["completion"] < end for row in useful) / (end - start))
    missed = sum(not row["output"] or row["completion"] - row["capture"] > args.deadline for row in requests)
    result = dict(config=vars(args), useful_service=len(useful) / duration, p10_30s_useful_service=float(np.quantile(windows, 0.1)),
                  offered=len(requests), completed=len(delivered), dropped_or_expired_fraction=missed / len(requests),
                  completed_output_accuracy=float(np.mean([row["correct"] for row in delivered])) if delivered else None,
                  delay_p95_seconds=float(np.quantile(delays, 0.95)) if delays else None,
                  seconds_over_soft=hot_seconds, seconds_over_hard=hard_seconds, session_energy_wh=total_energy / 3600,
                  peak_resident_memory_mib=peak_memory, switches=switches, infeasible_slots=infeasible,
                  planning_overruns=overruns, mean_planning_ms=float(np.mean([row["planning_seconds"] for row in history]) * 1000),
                  remaining_wh=controller.energy, minimum_window_service=min(windows))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics", result), ("history", history), ("requests", requests)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collection = sub.add_parser("collect")
    collection.add_argument("--images", required=True)
    collection.add_argument("--models", required=True)
    collection.add_argument("--output", required=True)
    collection.add_argument("--device", default="cpu")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--predictions", required=True)
    calibration.add_argument("--config")
    calibration.add_argument("--thermal-data")
    calibration.add_argument("--ambient", type=float, default=32)
    calibration.add_argument("--measured-latency", action="store_true")
    calibration.add_argument("--output", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--predictions", required=True)
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--phase-duration", type=int, default=300)
    preparation.add_argument("--seed", type=int, default=13)
    execution = sub.add_parser("run")
    execution.add_argument("--data", required=True)
    execution.add_argument("--profile", required=True)
    execution.add_argument("--output", required=True)
    execution.add_argument("--policy", choices=["full", "fixed_low", "fixed_high", "greedy", "threshold", "energy_mpc", "temperature_mpc", "no_debt", "no_switching", "no_queue", "no_monitor"], default="full")
    for name, default in (("horizon", 5), ("candidates", 24), ("minimum-dwell", 5), ("residual-count", 2), ("recovery-count", 5), ("seed", 13)):
        execution.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("ambient", 32), ("soft-temperature", 38), ("hard-temperature", 45), ("power-budget", 3),
                          ("memory-budget", 512), ("deadline", 0.1), ("planning-allowance", 0.01), ("quality-floor", 0),
                          ("debt-tau", 120), ("debt-weight", 0.2), ("energy-weight", 0.05), ("switch-weight", 0.02),
                          ("terminal-weight", 0.01), ("recovery-multiplier", 2), ("thermal-multiplier", 1)):
        execution.add_argument(f"--{name}", type=float, default=default)
    execution.add_argument("--measured-execution", action="store_true")
    plotting = sub.add_parser("plot")
    plotting.add_argument("--runs", nargs="+", required=True)
    plotting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "prepare":
        if args.phase_duration < 1:
            parser.error("Positive phase duration required")
        prepare(args)
    elif args.command == "run":
        if min(args.horizon, args.candidates, args.debt_tau, args.deadline, args.power_budget, args.thermal_multiplier) <= 0 or args.soft_temperature >= args.hard_temperature:
            parser.error("Invalid horizon, resource or temperature settings")
        run(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for metric in ("die_c", "debt", "remaining_wh"):
            figure, axis = plt.subplots(figsize=(8, 4))
            for run_path in args.runs:
                root = Path(run_path)
                rows = json.loads((root / "history.json").read_text(encoding="utf-8"))
                axis.plot([row["time"] for row in rows], [row[metric] for row in rows], label=root.name)
            axis.set(xlabel="Session seconds", ylabel=metric)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
        figure, axis = plt.subplots(figsize=(6, 4))
        for run_path in args.runs:
            row = json.loads((Path(run_path) / "metrics.json").read_text(encoding="utf-8"))
            axis.scatter(row["session_energy_wh"], row["useful_service"], label=Path(run_path).name)
        axis.set(xlabel="Session energy (Wh)", ylabel="Timely correct outputs/s")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "service_energy.png", dpi=180)
        plt.close(figure)
