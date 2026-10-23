"""Matplotlib visualisations for the cyclotron simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .instrumentation import MeasurementResult
from .model import SimulationResult


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(style="plain", axis="both")


def chamber_geometry(ax, result: SimulationResult) -> None:
    config = result.config
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(
        config.chamber_radius * np.cos(theta),
        config.chamber_radius * np.sin(theta),
        color="#1b8a3a",
        linewidth=2.0,
        label="chamber",
    )
    half_gap = config.gap_width / 2.0
    dee_radius = max(config.chamber_radius - config.margin, half_gap * 1.01)
    edge = np.sqrt(max(dee_radius**2 - half_gap**2, 0.0))
    for sign in (-1.0, 1.0):
        ax.plot(
            [-edge, edge],
            [sign * half_gap, sign * half_gap],
            color="#222222",
            linewidth=1.5,
        )
    alpha = np.arcsin(min(half_gap / dee_radius, 1.0))
    upper = np.linspace(alpha, np.pi - alpha, 200)
    lower = np.linspace(np.pi + alpha, 2.0 * np.pi - alpha, 200)
    ax.plot(
        dee_radius * np.cos(upper),
        dee_radius * np.sin(upper),
        color="#444444",
        linewidth=1.0,
    )
    ax.plot(
        dee_radius * np.cos(lower),
        dee_radius * np.sin(lower),
        color="#444444",
        linewidth=1.0,
    )
    if config.cup_enabled:
        ax.plot(
            config.cup_radius * np.cos(theta),
            config.cup_radius * np.sin(theta),
            color="#d97706",
            linestyle="--",
            linewidth=2.0,
            label="Faraday cup",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_trajectory(
    result: SimulationResult,
    ax,
    color: str | None = None,
    linestyle: str | tuple = "-",
    color_by_energy: bool = False,
) -> None:
    chamber_geometry(ax, result)
    if color_by_energy and result.x.size > 1:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        points = np.column_stack((result.x, result.y)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        energy = result.energy_j[:-1] / (1.602176634e-19 * 1.0e3)
        norm = Normalize(
            vmin=float(np.min(energy)), vmax=float(np.max(energy) + 1.0e-9)
        )
        collection = LineCollection(
            segments, cmap="plasma", norm=norm, linewidths=1.6
        )
        collection.set_array(energy)
        ax.add_collection(collection)
        colorbar = ax.figure.colorbar(collection, ax=ax, pad=0.02)
        colorbar.set_label("kinetic energy (keV)")
    else:
        ax.plot(
            result.x,
            result.y,
            color=color or "#111111",
            linestyle=linestyle,
            linewidth=1.8,
            label="ion trajectory",
        )
    if result.turn_times.size:
        radii = result.turn_radii
        angles = np.arctan2(
            np.interp(result.turn_times, result.t, result.y),
            np.interp(result.turn_times, result.t, result.x),
        )
        ax.plot(
            radii * np.cos(angles),
            radii * np.sin(angles),
            "o",
            color="#2563eb",
            markersize=3,
            label="turn markers",
        )
    ax.plot(result.x[0], result.y[0], "o", color="#16a34a", markersize=7)
    ax.plot(result.x[-1], result.y[-1], "s", color="#dc2626", markersize=6)
    ax.set_title("Beam trajectory")
    ax.legend(loc="upper right", fontsize=8)


def trajectory_figure(result: SimulationResult, figsize=(7.0, 7.0)):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=figsize)
    draw_trajectory(result, ax, color_by_energy=True)
    figure.tight_layout()
    return figure


def diagnostics_figure(result: SimulationResult, figsize=(11.0, 8.0)):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    time_us = result.t * 1.0e6

    axes[0, 0].plot(time_us, result.speed / 1.0e3, color="#2563eb")
    axes[0, 0].set_xlabel("time (us)")
    axes[0, 0].set_ylabel("speed (km/s)")
    axes[0, 0].set_title("Beam velocity")
    _style_axes(axes[0, 0])

    axes[0, 1].plot(time_us, result.energy_j / (1.602176634e-16), color="#dc2626")
    axes[0, 1].set_xlabel("time (us)")
    axes[0, 1].set_ylabel("kinetic energy (keV)")
    axes[0, 1].set_title("Beam energy")
    _style_axes(axes[0, 1])

    if result.turn_times.size:
        turns = np.arange(1, result.turn_times.size + 1)
        axes[1, 0].plot(
            turns,
            result.turn_radii * 1.0e3,
            "o-",
            color="#7c3aed",
            label="turn radius",
        )
        axes[1, 0].set_xlabel("turn number")
        axes[1, 0].set_ylabel("radius (mm)")
        axes[1, 0].set_title("Orbit growth")
        axes[1, 0].legend(fontsize=8)
        _style_axes(axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "no complete turn", ha="center", va="center")
        axes[1, 0].set_axis_off()

    if result.turn_energies_j.size:
        turns = np.arange(1, result.turn_energies_j.size + 1)
        axes[1, 1].plot(
            turns,
            result.turn_energies_j / (1.602176634e-16),
            "s-",
            color="#ea580c",
            label="turn energy",
        )
        axes[1, 1].set_xlabel("turn number")
        axes[1, 1].set_ylabel("kinetic energy (keV)")
        axes[1, 1].set_title("Energy gain per turn")
        axes[1, 1].legend(fontsize=8)
        _style_axes(axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "no complete turn", ha="center", va="center")
        axes[1, 1].set_axis_off()

    figure.tight_layout()
    return figure


def instrumentation_figure(
    result: SimulationResult,
    measurement: MeasurementResult,
    figsize=(11.0, 8.0),
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    time_us = measurement.t * 1.0e6

    axes[0, 0].plot(time_us, measurement.rf_voltage_v, color="#0f766e", linewidth=0.8)
    axes[0, 0].set_xlabel("time (us)")
    axes[0, 0].set_ylabel("dee voltage (V)")
    axes[0, 0].set_title("RF monitor")
    _style_axes(axes[0, 0])

    axes[0, 1].plot(
        time_us, measurement.pickup_voltage_v * 1.0e3, color="#1d4ed8", linewidth=0.8
    )
    if measurement.pickup_peak_times.size:
        peak_values = np.interp(
            measurement.pickup_peak_times,
            measurement.t,
            measurement.pickup_voltage_v,
        )
        axes[0, 1].plot(
            measurement.pickup_peak_times * 1.0e6,
            peak_values * 1.0e3,
            "rx",
            markersize=5,
            label="detected pulses",
        )
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_xlabel("time (us)")
    axes[0, 1].set_ylabel("pickup voltage (mV)")
    axes[0, 1].set_title("Pickup electrode signal")
    _style_axes(axes[0, 1])

    if measurement.fft_frequencies_hz.size:
        scale = 1.0e-6
        axes[1, 0].plot(
            measurement.fft_frequencies_hz * scale,
            measurement.fft_spectrum * 1.0e3,
            color="#9333ea",
            linewidth=1.0,
        )
        axes[1, 0].axvline(
            result.config.cyclotron_frequency * scale,
            color="#16a34a",
            linestyle="--",
            linewidth=1.0,
            label="theory",
        )
        if np.isfinite(measurement.fft_frequency_hz):
            axes[1, 0].axvline(
                measurement.fft_frequency_hz * scale,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
                label="FFT estimate",
            )
        axes[1, 0].set_xlim(0.0, 2.0 * result.config.cyclotron_frequency * scale)
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "no complete revolution",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_axis_off()
    axes[1, 0].set_xlabel("frequency (MHz)")
    axes[1, 0].set_ylabel("amplitude (mV)")
    axes[1, 0].set_title("Pickup spectrum")
    _style_axes(axes[1, 0])

    if result.config.cup_enabled:
        axes[1, 1].plot(
            time_us,
            measurement.cup_current_a * 1.0e6,
            color="#b45309",
            linewidth=0.9,
        )
        axes[1, 1].set_xlabel("time (us)")
        axes[1, 1].set_ylabel("cup current (uA)")
        axes[1, 1].set_title("Faraday cup current")
        _style_axes(axes[1, 1])
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Faraday cup not inserted",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_axis_off()

    figure.tight_layout()
    return figure


def save_figures(
    result: SimulationResult,
    measurement: MeasurementResult,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    trajectory = trajectory_figure(result)
    path = output / "trajectory.png"
    trajectory.savefig(path, dpi=160)
    paths.append(path)

    diagnostics = diagnostics_figure(result)
    path = output / "beam_diagnostics.png"
    diagnostics.savefig(path, dpi=160)
    paths.append(path)

    instrumentation = instrumentation_figure(result, measurement)
    path = output / "instrumentation.png"
    instrumentation.savefig(path, dpi=160)
    paths.append(path)

    import matplotlib.pyplot as plt

    plt.close(trajectory)
    plt.close(diagnostics)
    plt.close(instrumentation)
    return paths
