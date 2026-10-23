"""Synthetic detector signals for the cyclotron simulation.

The tracked ion is treated as a bunch of ``beam_particles`` ions.  Three
instruments are modelled:

* a capacitive pickup electrode that produces one pulse per revolution,
  from which the cyclotron frequency is estimated (time domain and FFT)
* a Faraday cup at ``cup_radius``, producing a current
  pulse whenever the beam crosses it
* an RF monitor sampling the dee voltage

The signal chain includes sampling, Gaussian noise, timing jitter and a
finite detector resolution, so that measured values differ slightly from
the simulated truth in the same way as in a real measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import elementary_charge, speed_of_light
from scipy.signal import find_peaks

from .model import SimulationResult


@dataclass
class InstrumentConfig:
    sample_rate_hz: float = 20.0e9
    pickup_amplitude_v: float = 1.0
    pickup_sigma_s: float = 0.5e-9
    pickup_noise_rms_v: float = 0.01
    timing_jitter_s: float = 20.0e-12
    cup_pulse_sigma_s: float = 0.8e-9
    cup_noise_rms_a: float = 1.0e-6
    beam_particles: float = 1.0e7
    radius_resolution_mm: float = 0.05
    field_relative_error: float = 1.0e-3
    rf_noise_rms_v: float = 0.5
    seed: int = 20261023

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("The sample rate must be positive.")
        if self.beam_particles <= 0:
            raise ValueError("beam_particles must be positive.")


@dataclass
class MeasurementResult:
    instrument: InstrumentConfig
    t: np.ndarray
    rf_voltage_v: np.ndarray
    pickup_voltage_v: np.ndarray
    pickup_peak_times: np.ndarray
    pickup_frequency_hz: float
    pickup_frequency_uncertainty_hz: float
    theoretical_frequency_hz: float
    theoretical_final_energy_kev: float
    fft_frequency_hz: float
    fft_amplitude: float
    fft_frequencies_hz: np.ndarray
    fft_spectrum: np.ndarray
    cup_current_a: np.ndarray
    cup_charge_c: float
    cup_peak_current_a: float
    measured_final_radius_mm: float
    measured_final_energy_kev: float
    measured_energy_uncertainty_kev: float

    @property
    def frequency_error_hz(self) -> float:
        return self.pickup_frequency_hz - self.theoretical_frequency_hz

    @property
    def frequency_error_rel(self) -> float:
        if self.theoretical_frequency_hz == 0:
            return 0.0
        return self.frequency_error_hz / self.theoretical_frequency_hz


def _sample_times(result: SimulationResult, sample_rate: float) -> np.ndarray:
    duration = max(result.duration, 0.0)
    count = int(np.floor(duration * sample_rate)) + 1
    return np.arange(count, dtype=float) / sample_rate


def _pickup_signal(
    result: SimulationResult,
    times: np.ndarray,
    instrument: InstrumentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.zeros_like(times)
    if not result.turn_times.size:
        return signal, np.zeros(0)

    radius_scale = np.clip(result.turn_radii / result.config.chamber_radius, 0.1, 1.0)
    amplitudes = instrument.pickup_amplitude_v * np.sqrt(radius_scale)
    centers = result.turn_times + rng.normal(
        0.0, instrument.timing_jitter_s, result.turn_times.size
    )
    sigma = instrument.pickup_sigma_s
    for center, amplitude in zip(centers, amplitudes):
        window = np.abs(times - center) <= 6.0 * sigma
        if np.any(window):
            signal[window] += amplitude * np.exp(
                -0.5 * ((times[window] - center) / sigma) ** 2
            )
    signal += rng.normal(0.0, instrument.pickup_noise_rms_v, times.size)
    return signal, centers


def _estimate_frequency(
    times: np.ndarray,
    signal: np.ndarray,
    rough_frequency: float,
    instrument: InstrumentConfig,
) -> tuple[float, float, np.ndarray]:
    if signal.size < 3 or rough_frequency <= 0:
        return np.nan, np.nan, np.zeros(0)
    minimum_distance = max(
        1, int(0.6 * instrument.sample_rate_hz / rough_frequency)
    )
    peaks, _ = find_peaks(
        signal,
        height=0.35 * float(np.max(signal)),
        distance=minimum_distance,
    )
    peak_times = times[peaks]
    if peak_times.size < 2:
        return np.nan, np.nan, peak_times

    cycles = np.arange(peak_times.size, dtype=float)
    slope, intercept = np.polyfit(peak_times, cycles, 1.0)
    fitted = slope * peak_times + intercept
    residuals = cycles - fitted
    if peak_times.size >= 3:
        residual_std = np.sqrt(np.sum(residuals**2) / (peak_times.size - 2))
        sxx = np.sum((peak_times - peak_times.mean()) ** 2)
        uncertainty = residual_std / np.sqrt(sxx) if sxx > 0 else np.nan
    else:
        uncertainty = abs(slope) * 1.0e-3
    return float(slope), float(uncertainty), peak_times


def _fft_estimate(
    times: np.ndarray,
    signal: np.ndarray,
    sample_rate: float,
    expected_frequency: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if signal.size < 8:
        empty = np.zeros(0)
        return np.nan, 0.0, empty, empty
    centered = signal - np.mean(signal)
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(centered * window)) * 2.0 / np.sum(window)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    if expected_frequency > 0:
        band = (frequencies >= 0.25 * expected_frequency) & (
            frequencies <= 1.75 * expected_frequency
        )
        if not np.any(band):
            band = np.ones_like(frequencies, dtype=bool)
        indices = np.where(band)[0]
    else:
        indices = np.arange(frequencies.size)
    index = int(indices[np.argmax(spectrum[indices])])
    frequency = float(frequencies[index])
    if 0 < index < frequencies.size - 1:
        left = spectrum[index - 1]
        middle = spectrum[index]
        right = spectrum[index + 1]
        denominator = left - 2.0 * middle + right
        if denominator != 0:
            offset = 0.5 * (left - right) / denominator
            frequency += offset * (frequencies[1] - frequencies[0])
    amplitude = float(spectrum[index])
    return frequency, amplitude, frequencies, spectrum


def _cup_signal(
    result: SimulationResult,
    times: np.ndarray,
    instrument: InstrumentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    signal = np.zeros_like(times)
    if not result.cup_hit_times.size:
        return signal, 0.0, 0.0
    charge_per_hit = (
        abs(result.config.particle.charge) * instrument.beam_particles
    )
    sigma = instrument.cup_pulse_sigma_s
    for hit in result.cup_hit_times:
        window = np.abs(times - hit) <= 6.0 * sigma
        if np.any(window):
            amplitude = charge_per_hit / (sigma * np.sqrt(2.0 * np.pi))
            signal[window] += amplitude * np.exp(
                -0.5 * ((times[window] - hit) / sigma) ** 2
            )
    if signal.size:
        signal += rng.normal(0.0, instrument.cup_noise_rms_a, signal.size)
    total_charge = charge_per_hit * result.cup_hit_times.size
    peak_current = float(np.max(signal)) if signal.size else 0.0
    return signal, total_charge, peak_current


def _fit_orbit_radius(result: SimulationResult) -> float:
    frequency = result.config.cyclotron_frequency
    if frequency <= 0 or result.t.size < 10:
        return result.final_radius
    period = 1.0 / frequency
    window = max(0.25 * period, 5.0 * result.config.time_step)
    mask = result.t >= result.t[-1] - window
    if np.count_nonzero(mask) < 10:
        mask = np.zeros_like(result.t, dtype=bool)
        mask[-min(result.t.size, 64) :] = True
    x = result.x[mask]
    y = result.y[mask]
    if x.size < 3 or (np.ptp(x) == 0.0 and np.ptp(y) == 0.0):
        return result.final_radius
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    right_hand_side = x * x + y * y
    if np.linalg.matrix_rank(matrix) < 3:
        return result.final_radius
    try:
        solution, *_ = np.linalg.lstsq(matrix, right_hand_side, rcond=None)
    except np.linalg.LinAlgError:
        return result.final_radius
    center_x, center_y, constant = solution
    radius_squared = constant + center_x**2 + center_y**2
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return result.final_radius
    return float(np.sqrt(radius_squared))


def _energy_from_radius(
    result: SimulationResult, radius_m: float, magnetic_field: float
) -> float:
    config = result.config
    charge = abs(config.particle.charge)
    mass = config.particle.mass
    rigidity = charge * magnetic_field * radius_m
    if config.relativistic:
        total_energy = np.sqrt(
            (rigidity * speed_of_light) ** 2 + (mass * speed_of_light**2) ** 2
        )
        return float(total_energy - mass * speed_of_light**2)
    return float(rigidity**2 / (2.0 * mass))


def measure_beam(
    result: SimulationResult, instrument: InstrumentConfig | None = None
) -> MeasurementResult:
    instrument = instrument or InstrumentConfig()
    rng = np.random.default_rng(instrument.seed)
    times = _sample_times(result, instrument.sample_rate_hz)

    rf_voltage = np.asarray(result.config.rf_voltage(times), dtype=float)
    rf_voltage = rf_voltage + rng.normal(
        0.0, instrument.rf_noise_rms_v, rf_voltage.size
    )

    pickup, pulse_centers = _pickup_signal(result, times, instrument, rng)
    rough_frequency = result.config.cyclotron_frequency
    pickup_frequency, frequency_uncertainty, peak_times = _estimate_frequency(
        times, pickup, rough_frequency, instrument
    )
    if result.turn_times.size:
        fft_frequency, fft_amplitude, fft_frequencies, fft_spectrum = _fft_estimate(
            times, pickup, instrument.sample_rate_hz, rough_frequency
        )
    else:
        fft_frequency, fft_amplitude = np.nan, 0.0
        fft_frequencies = np.zeros(0)
        fft_spectrum = np.zeros(0)

    cup_current, cup_charge, cup_peak = _cup_signal(
        result, times, instrument, rng
    )

    orbit_radius = _fit_orbit_radius(result)
    measured_radius_mm = orbit_radius * 1.0e3 + rng.normal(
        0.0, instrument.radius_resolution_mm
    )
    sigma_r_m = instrument.radius_resolution_mm * 1.0e-3
    energy_high = _energy_from_radius(
        result, orbit_radius + sigma_r_m, result.config.magnetic_field
    )
    energy_low = _energy_from_radius(
        result, max(orbit_radius - sigma_r_m, 0.0), result.config.magnetic_field
    )
    measured_energy = _energy_from_radius(
        result,
        max(measured_radius_mm, 0.0) * 1.0e-3,
        result.config.magnetic_field,
    )
    radius_uncertainty = abs(energy_high - energy_low) / 2.0
    field_uncertainty = measured_energy * instrument.field_relative_error
    total_uncertainty = float(
        np.hypot(radius_uncertainty, field_uncertainty)
        / (elementary_charge * 1.0e3)
    )

    measurement = MeasurementResult(
        instrument=instrument,
        t=times,
        rf_voltage_v=rf_voltage,
        pickup_voltage_v=pickup,
        pickup_peak_times=peak_times if peak_times.size else pulse_centers,
        pickup_frequency_hz=pickup_frequency,
        pickup_frequency_uncertainty_hz=frequency_uncertainty,
        theoretical_frequency_hz=result.config.cyclotron_frequency,
        theoretical_final_energy_kev=result.final_energy_kev,
        fft_frequency_hz=fft_frequency,
        fft_amplitude=fft_amplitude,
        fft_frequencies_hz=fft_frequencies,
        fft_spectrum=fft_spectrum,
        cup_current_a=cup_current,
        cup_charge_c=cup_charge,
        cup_peak_current_a=cup_peak,
        measured_final_radius_mm=measured_radius_mm,
        measured_final_energy_kev=measured_energy / (elementary_charge * 1.0e3),
        measured_energy_uncertainty_kev=total_uncertainty,
    )
    return measurement
