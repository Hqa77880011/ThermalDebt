import itertools
import time

import numpy as np


class Scheduler:
    def __init__(self, config, duration, args):
        self.config, self.args, self.duration = config, args, duration
        self.actions = config["actions"]
        if any(action["latency_s"] < 0 or action["load_s"] < 0 or action["latency_s"] + action["load_s"] >= 1 for action in self.actions):
            raise ValueError("Action latency and loading must fit the one-second slot characterization")
        if any(not 0 <= action["accuracy"] <= 1 or min(action["fps"], action["active_w"], action["memory_mib"], action["switch_j"]) < 0 for action in self.actions):
            raise ValueError("Action accuracy must be a probability and resource quantities nonnegative")
        self.pause = len(self.actions) - 1
        self.a = np.array(config["A"], dtype=float)
        self.b = np.array(config["B"], dtype=float)
        if (self.a < 0).any() or (self.b < 0).any() or np.max(np.abs(np.linalg.eigvals(self.a))) >= 1:
            raise ValueError("Nonnegative stable thermal A and nonnegative B required")
        self.disturbance = np.array(config["disturbance"], dtype=float)
        self.lower, self.upper = np.zeros(2), np.zeros(2)
        self.previous_prediction = None
        self.debt, self.energy = 0.0, args.power_budget * duration / 3600
        self.previous, self.dwell, self.previous_plan = self.pause, args.minimum_dwell, None
        self.resident = self.pause
        self.recovery, self.bad, self.good = False, 0, 0
        self.powers = np.stack([np.linalg.matrix_power(self.a, i) for i in range(duration + 1)])

    def observe(self, observation):
        error = np.asarray(self.config["observation_error"])
        low, high = observation - self.args.ambient - error, observation - self.args.ambient + error
        if self.previous_prediction is not None:
            predicted_low, predicted_high = self.previous_prediction
            inside = bool(np.all(high >= predicted_low) and np.all(low <= predicted_high))
            self.bad = self.bad + 1 if not inside else 0
            self.good = self.good + 1 if inside else 0
            if self.bad >= self.args.residual_count and self.args.policy != "no_monitor":
                self.recovery = True
                self.previous_plan = None
            intersection_low, intersection_high = np.maximum(low, predicted_low), np.minimum(high, predicted_high)
            if inside:
                low, high = intersection_low, intersection_high
        self.lower, self.upper = low, high

    def transition(self, previous, action):
        if action == self.pause or action == previous:
            return 0.0, 0.0
        return self.actions[action]["load_s"], self.actions[action]["switch_j"]

    def memory(self, previous, action):
        model = 0 if action == self.pause else self.actions[action]["memory_mib"]
        if action != self.pause and previous not in (self.pause, action):
            model += self.actions[previous]["memory_mib"]
        if action == self.pause and previous != self.pause:
            model = self.actions[previous]["memory_mib"]
        return model + self.config["workspace_mib"]

    def slot_energy(self, previous, action, rate, background):
        specification = self.actions[action]
        loading, switching = self.transition(previous, action)
        count = min(rate, specification["fps"], 1 / max(specification["latency_s"], 1e-8)) * max(0, 1 - loading)
        active = min(1, count * specification["latency_s"])
        return self.config["base_w"] + background + specification["active_w"] * active + switching

    def propagate(self, low, high, energy):
        uncertainty = self.disturbance * (self.args.recovery_multiplier if self.recovery else 1)
        return self.a @ low + self.b * energy - uncertainty, self.a @ high + self.b * energy + uncertainty

    def terminal(self, low, high, remaining, energy, background):
        if remaining <= 0:
            return True
        fallback = self.config["base_w"] + background
        reserve = (remaining * fallback + self.config["reserve_j"]) / 3600
        if energy < reserve:
            return False
        forcing = self.b * fallback + self.disturbance * (self.args.recovery_multiplier if self.recovery else 1)
        equilibrium = np.linalg.solve(np.eye(2) - self.a, forcing)
        upper = np.einsum("tij,j->ti", self.powers[1:remaining + 1], high - equilibrium) + equilibrium
        return bool(np.max(upper[:, 0] + self.args.ambient) <= self.args.hard_temperature)

    def admissible(self, previous, action, low, high, remaining, energy, age, rate, background, reserve=True):
        specification = self.actions[action]
        loading, _ = self.transition(previous, action)
        if self.memory(previous, action) > self.args.memory_budget:
            return None
        if action != self.pause and specification["accuracy"] < self.args.quality_floor:
            return None
        if action != self.pause and self.args.policy != "no_queue" and age + specification["latency_s"] + loading + self.args.planning_allowance > self.args.deadline:
            return None
        expenditure = self.slot_energy(previous, action, rate, background)
        next_low, next_high = self.propagate(low, high, expenditure)
        next_energy = energy - expenditure / 3600
        if next_energy < 0 or next_high[0] + self.args.ambient > self.args.hard_temperature:
            return None
        if reserve and not self.terminal(next_low, next_high, remaining - 1, next_energy, background):
            return None
        return next_low, next_high, next_energy, expenditure

    def candidates(self, remaining):
        horizon = min(self.args.horizon, remaining)
        plans = []
        if self.previous_plan is not None:
            plans.append(tuple((self.previous_plan[1:] + [self.pause])[:horizon]))
        plans.extend(tuple([action] * horizon) for action in range(len(self.actions)))
        for switch, action in itertools.product(range(1, horizon), range(len(self.actions))):
            plans.append(tuple([self.previous] * switch + [action] * (horizon - switch)))
        return list(dict.fromkeys(plans))[:self.args.candidates]

    def select(self, remaining, age, rate, background):
        started = time.perf_counter()
        policy = self.args.policy
        if self.recovery and self.good >= self.args.recovery_count:
            fallback = self.admissible(self.resident, self.pause, self.lower, self.upper, remaining,
                                       self.energy, age, rate, background)
            if fallback is not None:
                self.recovery = False
        permitted = [0, self.pause] if self.recovery else list(range(len(self.actions)))
        if policy in ("fixed_low", "fixed_high"):
            selected = 0 if policy == "fixed_low" else self.pause - 1
            feasible = self.admissible(self.resident, selected, self.lower, self.upper, remaining, self.energy, age, rate, background) is not None
            return selected, dict(planning_seconds=time.perf_counter() - started, feasible=feasible, candidate_count=1)
        if policy == "threshold":
            desired = self.pause - 1 if self.upper[0] + self.args.ambient < self.args.soft_temperature - 0.5 else 0
            if self.previous == 0 and self.upper[0] + self.args.ambient > self.args.soft_temperature - 1:
                desired = 0
            permitted = [desired, self.pause]
        plans = [(action,) for action in permitted] if policy in ("greedy", "threshold") else self.candidates(remaining)
        feasible_previous = self.admissible(self.resident, self.previous, self.lower, self.upper, remaining, self.energy, age, rate, background) is not None
        scored = []
        for plan in plans:
            low, high, energy, debt = self.lower.copy(), self.upper.copy(), self.energy, self.debt
            previous, dwell, score = self.resident, self.dwell, 0.0
            valid = True
            for step, action in enumerate(plan):
                if action not in permitted:
                    valid = False
                    break
                if policy != "no_switching" and action not in (previous, self.pause) and dwell < self.args.minimum_dwell and feasible_previous:
                    valid = False
                    break
                result = self.admissible(previous, action, low, high, remaining - step, energy, age if step == 0 else 0,
                                         rate, background, reserve=step == len(plan) - 1)
                if result is None:
                    valid = False
                    break
                next_low, next_high, next_energy, expenditure = result
                loading, _ = self.transition(previous, action)
                specification = self.actions[action]
                service = min(rate, specification["fps"], 1 / max(specification["latency_s"], 1e-8)) * specification["accuracy"] * max(0, 1 - loading) / self.config["service_reference"]
                debt_weight = 0 if policy in ("temperature_mpc", "energy_mpc", "greedy", "threshold", "no_debt") else self.args.debt_weight
                energy_weight = 0 if policy in ("temperature_mpc", "greedy", "threshold") else self.args.energy_weight
                switch_weight = 0 if policy == "no_switching" else self.args.switch_weight
                score += service - debt_weight * debt / self.config["debt_reference"] - energy_weight * expenditure / self.config["energy_reference"] - switch_weight * loading / self.args.deadline
                debt = np.exp(-1 / self.args.debt_tau) * debt + max(0, high[0] + self.args.ambient - self.args.soft_temperature)
                low, high, energy = next_low, next_high, next_energy
                dwell = dwell + 1 if action in (previous, self.pause) else 1
                previous = previous if action == self.pause else action
            if valid:
                score -= self.args.terminal_weight * max(0, high[0] + self.args.ambient - self.args.soft_temperature)
                scored.append((score, plan))
        if not scored:
            selected, plan = self.pause, [self.pause]
        else:
            _, plan = max(scored, key=lambda row: row[0])
            selected = plan[0]
        self.previous_plan = list(plan)
        return selected, dict(planning_seconds=time.perf_counter() - started, feasible=bool(scored), candidate_count=len(plans))

    def feedback(self, action, measured_energy, temperature, background):
        self.previous_prediction = self.propagate(self.lower, self.upper, measured_energy)
        self.debt = np.exp(-1 / self.args.debt_tau) * self.debt + max(0, temperature[0] - self.args.soft_temperature)
        self.energy -= measured_energy / 3600
        self.dwell = self.dwell + 1 if action in (self.resident, self.pause) else 1
        self.previous = action
        if action != self.pause:
            self.resident = action
