"""Cyclotron beam dynamics for measurement and instrumentation studies.

The model uses the following conventions:

* magnetic field along +z, dee gap parallel to the x axis
* electric field along y inside the gap ``|y| <= d/2``
* RF voltage either sinusoidal or square wave
* non-relativistic or relativistic Lorentz dynamics

A single ion is tracked.  Instrumentation (pickup electrode, Faraday cup,
RF monitor) is synthesized from the tracked trajectory in
``cyclotron_sim.instrumentation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import (
    elementary_charge,
    neutron_mass,
    proton_mass,
    speed_of_light,
)
from scipy.integrate import solve_ivp
from scipy.signal import square

C_LIGHT = speed_of_light


@dataclass(frozen=True)
class Particle:
    """Ion definition using the proton/neutron/charge convention of the UI."""

    charge_number: int = 1
    protons: int = 1
    neutrons: int = 0

    def __post_init__(self) -> None:
        if self.protons + self.neutrons <= 0:
            raise ValueError("A particle needs at least one nucleon.")
        if self.charge_number == 0:
            raise ValueError("The charge number must not be zero.")

    @property
    def charge(self) -> float:
        return self.charge_number * elementary_charge

    @property
    def mass(self) -> float:
        return self.protons * proton_mass + self.neutrons * neutron_mass

    @property
    def label(self) -> str:
        if (self.protons, self.neutrons, self.charge_number) == (1, 0, 1):
            return "proton"
        return (
            f"ion Z={self.charge_number} "
            f"(p={self.protons}, n={self.neutrons})"
        )


@dataclass
class CyclotronConfig:
    """Complete description of the cyclotron and the tracked ion."""

    particle: Particle = field(default_factory=Particle)
    magnetic_field: float = 0.185
    rf_frequency: float = 2.82e6
    acceleration_voltage: float = 1000.0
    gap_width: float = 0.02
    rf_phase_deg: float = 0.0
    waveform: str = "sine"
    chamber_radius: float = 0.07
    margin: float = 0.003
    x0: float = -0.01
    y0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0
    t_end: float = 4.0e-6
    time_step: float = 1.0e-9
    relativistic: bool = False
    voltage_reduction_pct: float = 0.0
    cup_enabled: bool = False
    cup_radius_mm: float = 32.0

    def __post_init__(self) -> None:
        self.waveform = self.waveform.lower().strip()
        if self.waveform not in {"sine", "square"}:
            raise ValueError("waveform must be 'sine' or 'square'.")
        if self.magnetic_field <= 0:
            raise ValueError("The magnetic field must be positive.")
        if self.gap_width <= 0:
            raise ValueError("The gap width must be positive.")
        if not 0.0 <= self.voltage_reduction_pct < 100.0:
            raise ValueError("voltage_reduction_pct must be in [0, 100).")
        if self.chamber_radius <= self.gap_width / 2:
            raise ValueError("The gap does not fit into the chamber.")
        if np.hypot(self.x0, self.y0) >= self.chamber_radius:
            raise ValueError("The start position must lie inside the chamber.")
        if self.cup_enabled and self.cup_radius_mm * 1.0e-3 >= self.chamber_radius:
            raise ValueError("The Faraday cup must lie inside the chamber.")
        if self.time_step <= 0 or self.t_end <= 0:
            raise ValueError("time_step and t_end must be positive.")

    @property
    def effective_voltage(self) -> float:
        return self.acceleration_voltage * (
            1.0 - self.voltage_reduction_pct / 100.0
        )

    @property
    def gap_field(self) -> float:
        return self.effective_voltage / self.gap_width

    @property
    def angular_frequency(self) -> float:
        return self.particle.charge * self.magnetic_field / self.particle.mass

    @property
    def cyclotron_frequency(self) -> float:
        return abs(self.angular_frequency) / (2.0 * np.pi)

    @property
    def detuning(self) -> float:
        return self.rf_frequency / self.cyclotron_frequency - 1.0

    @property
    def phase(self) -> float:
        return np.deg2rad(self.rf_phase_deg)

    @property
    def cup_radius(self) -> float:
        return self.cup_radius_mm * 1.0e-3

    def waveform_value(self, rf_phase: np.ndarray | float):
        if self.waveform == "square":
            return square(rf_phase)
        return np.cos(rf_phase)

    def rf_voltage(self, t: np.ndarray | float):
        return self.effective_voltage * self.waveform_value(
            2.0 * np.pi * self.rf_frequency * t - self.phase
        )


def _kinetic_energy(config: CyclotronConfig, speed: np.ndarray) -> np.ndarray:
    mass = config.particle.mass
    if config.relativistic:
        beta2 = np.clip((speed / C_LIGHT) ** 2, 0.0, 1.0 - 1.0e-12)
        gamma = 1.0 / np.sqrt(1.0 - beta2)
        return (gamma - 1.0) * mass * C_LIGHT**2
    return 0.5 * mass * speed**2


@dataclass
class SimulationResult:
    """Sampled trajectory plus derived beam and turn-by-turn quantities."""

    config: CyclotronConfig
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray
    radius: np.ndarray
    energy_j: np.ndarray
    path_length: np.ndarray
    gap_crossings_t: np.ndarray
    gap_crossings_phase: np.ndarray
    gap_crossings_energy_j: np.ndarray
    turn_times: np.ndarray
    turn_radii: np.ndarray
    turn_energies_j: np.ndarray
    cup_hit_times: np.ndarray
    exit_reason: str

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def final_speed(self) -> float:
        return float(self.speed[-1]) if self.speed.size else 0.0

    @property
    def final_radius(self) -> float:
        return float(self.radius[-1]) if self.radius.size else 0.0

    @property
    def final_energy_j(self) -> float:
        return float(self.energy_j[-1]) if self.energy_j.size else 0.0

    @property
    def final_energy_kev(self) -> float:
        return self.final_energy_j / (elementary_charge * 1.0e3)

    @property
    def final_radius_mm(self) -> float:
        return self.final_radius * 1.0e3

    @property
    def total_path_length(self) -> float:
        return float(self.path_length[-1]) if self.path_length.size else 0.0

    @property
    def n_turns(self) -> int:
        return int(self.turn_times.size)

    @property
    def initial_index(self) -> int:
        after_start = np.where(self.gap_crossings_t > 1.0e-12)[0]
        if after_start.size:
            crossing_time = self.gap_crossings_t[after_start[0]]
            return int(np.searchsorted(self.t, crossing_time))
        return 0

    @property
    def initial_speed(self) -> float:
        index = min(self.initial_index, self.speed.size - 1)
        return float(self.speed[index]) if self.speed.size else 0.0

    @property
    def initial_energy_j(self) -> float:
        index = min(self.initial_index, self.energy_j.size - 1)
        return float(self.energy_j[index]) if self.energy_j.size else 0.0

    @property
    def initial_radius_mm(self) -> float:
        if not self.speed.size:
            return 0.0
        radius = (
            self.config.particle.mass
            * self.initial_speed
            / (abs(self.config.particle.charge) * self.config.magnetic_field)
        )
        return radius * 1.0e3

    @property
    def revolution_frequency(self) -> float:
        if self.turn_times.size < 2:
            return 0.0
        intervals = np.diff(self.turn_times)
        return float(1.0 / np.mean(intervals))

    @property
    def energy_gain_per_turn_kev(self) -> float:
        if self.turn_energies_j.size < 2:
            return 0.0
        return float(
            np.mean(np.diff(self.turn_energies_j))
            / (elementary_charge * 1.0e3)
        )

    @property
    def mean_crossing_phase(self) -> float:
        if not self.gap_crossings_phase.size:
            return 0.0
        return float(np.mean(np.mod(self.gap_crossings_phase, np.pi)))


class CyclotronSimulation:
    """Integrate the single-particle equations of motion."""

    def __init__(self, config: CyclotronConfig | None = None) -> None:
        self.config = config or CyclotronConfig()
        self.result: SimulationResult | None = None

    def electric_field(self, t: float) -> float:
        voltage = np.asarray(self.config.rf_voltage(t)).ravel()[0]
        return float(voltage) / self.config.gap_width

    def _rhs(self, t: float, state: np.ndarray) -> list[float]:
        _, y, vx, vy = state
        if abs(y) <= self.config.gap_width / 2.0:
            field = self.electric_field(t)
        else:
            field = 0.0

        mass = self.config.particle.mass
        charge = self.config.particle.charge
        omega = charge * self.config.magnetic_field / mass

        if self.config.relativistic:
            beta2 = (vx * vx + vy * vy) / C_LIGHT**2
            gamma = 1.0 / np.sqrt(max(1.0 - beta2, 1.0e-12))
        else:
            gamma = 1.0

        ax = omega * vy / gamma
        ay = (charge * field) / (gamma * mass) - omega * vx / gamma
        return [vx, vy, ax, ay]

    def _first_cup_hit(
        self, t: np.ndarray, states: np.ndarray
    ) -> tuple[int, float, np.ndarray] | None:
        if not self.config.cup_enabled:
            return None
        radial = np.hypot(states[:, 0], states[:, 1])
        target = self.config.cup_radius
        for index in range(1, len(t)):
            previous = radial[index - 1]
            current = radial[index]
            if previous < target <= current:
                fraction = (target - previous) / (current - previous)
                hit_time = t[index - 1] + fraction * (t[index] - t[index - 1])
                hit_state = states[index - 1] + fraction * (
                    states[index] - states[index - 1]
                )
                position_radius = np.hypot(hit_state[0], hit_state[1])
                if position_radius > 0:
                    hit_state[0] *= target / position_radius
                    hit_state[1] *= target / position_radius
                return index, float(hit_time), hit_state
        return None

    def _cup_hits(self, t: np.ndarray, states: np.ndarray) -> np.ndarray:
        if not self.config.cup_enabled:
            return np.zeros(0)
        if np.hypot(states[-1, 0], states[-1, 1]) >= self.config.cup_radius - 1.0e-9:
            return np.asarray([t[-1]])
        return np.zeros(0)

    def _gap_crossings(
        self, t: np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(t) < 2:
            empty = np.zeros(0)
            return empty, empty, empty
        times: list[float] = []
        phases: list[float] = []
        gains: list[float] = []
        for index in range(1, len(t)):
            y_previous = states[index - 1, 1]
            y_current = states[index, 1]
            if not (y_previous < 0.0 <= y_current or y_previous > 0.0 >= y_current):
                continue
            fraction = -y_previous / (y_current - y_previous)
            crossing_time = t[index - 1] + fraction * (t[index] - t[index - 1])
            vy_crossing = states[index - 1, 3] + fraction * (
                states[index, 3] - states[index - 1, 3]
            )
            rf_phase = float(
                np.mod(
                    2.0 * np.pi * self.config.rf_frequency * crossing_time
                    - self.config.phase,
                    2.0 * np.pi,
                )
            )
            field = self.config.gap_field * self.config.waveform_value(rf_phase)
            gain = (
                self.config.particle.charge
                * field
                * np.sign(vy_crossing)
                * self.config.gap_width
            )
            times.append(crossing_time)
            phases.append(rf_phase)
            gains.append(gain)
        return (
            np.asarray(times),
            np.asarray(phases),
            np.asarray(gains),
        )

    def _turns(
        self, t: np.ndarray, states: np.ndarray, energy_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(t) < 2:
            empty = np.zeros(0)
            return empty, empty, empty
        angle = np.unwrap(np.arctan2(states[:, 1], states[:, 0]))
        travelled = np.abs(angle - angle[0])
        turn_number = np.floor(travelled / (2.0 * np.pi)).astype(int)
        boundaries = np.where(np.diff(turn_number) > 0)[0] + 1
        if not boundaries.size:
            empty = np.zeros(0)
            return empty, empty, empty
        return (
            t[boundaries],
            np.hypot(states[boundaries, 0], states[boundaries, 1]),
            energy_j[boundaries],
        )

    def run(self) -> SimulationResult:
        config = self.config
        t_eval = np.arange(0.0, config.t_end, config.time_step)
        if t_eval.size == 0 or t_eval[-1] < config.t_end:
            t_eval = np.append(t_eval, config.t_end)

        def extraction(_t: float, state: np.ndarray) -> float:
            return float(np.hypot(state[0], state[1]) - config.chamber_radius)

        extraction.terminal = True
        extraction.direction = 1

        initial_state = np.array(
            [config.x0, config.y0, config.vx0, config.vy0], dtype=float
        )
        solution = solve_ivp(
            self._rhs,
            (0.0, config.t_end),
            initial_state,
            method="DOP853",
            t_eval=t_eval,
            events=extraction,
            rtol=1.0e-9,
            atol=1.0e-11,
            max_step=config.time_step,
        )
        if not solution.success and solution.status < 0:
            raise RuntimeError(f"Integration failed: {solution.message}")

        t = np.asarray(solution.t, dtype=float)
        states = np.asarray(solution.y.T, dtype=float)
        if t.size == 0:
            t = np.array([0.0])
            states = initial_state.reshape(1, 4)

        if solution.t_events and len(solution.t_events[0]):
            event_time = float(solution.t_events[0][0])
            event_state = np.asarray(solution.y_events[0][0], dtype=float)
            if event_time > t[-1] + 1.0e-15:
                t = np.append(t, event_time)
                states = np.vstack((states, event_state))

        cup_hit = self._first_cup_hit(t, states)
        if cup_hit is not None:
            cut_index, hit_time, hit_state = cup_hit
            t = np.append(t[:cut_index], hit_time)
            states = np.vstack((states[:cut_index], hit_state))

        x = states[:, 0]
        y = states[:, 1]
        vx = states[:, 2]
        vy = states[:, 3]
        speed = np.hypot(vx, vy)
        radius = np.hypot(x, y)
        energy_j = _kinetic_energy(config, speed)
        steps = np.hypot(np.diff(x), np.diff(y))
        path_length = np.concatenate(([0.0], np.cumsum(steps)))

        crossings_t, crossings_phase, crossings_energy = self._gap_crossings(
            t, states
        )
        turn_times, turn_radii, turn_energies = self._turns(t, states, energy_j)
        cup_hits = self._cup_hits(t, states)

        if config.cup_enabled and cup_hits.size and cup_hits[-1] >= t[-1] - 1.0e-15:
            exit_reason = "faraday cup"
        elif radius[-1] >= config.chamber_radius - 1.0e-9:
            exit_reason = "extracted"
        else:
            exit_reason = "time limit"

        result = SimulationResult(
            config=config,
            t=t,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=speed,
            radius=radius,
            energy_j=energy_j,
            path_length=path_length,
            gap_crossings_t=crossings_t,
            gap_crossings_phase=crossings_phase,
            gap_crossings_energy_j=crossings_energy,
            turn_times=turn_times,
            turn_radii=turn_radii,
            turn_energies_j=turn_energies,
            cup_hit_times=cup_hits,
            exit_reason=exit_reason,
        )
        self.result = result
        return result
