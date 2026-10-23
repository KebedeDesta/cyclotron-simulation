"""Command-line runner for the cyclotron measurement simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cyclotron_sim import (
    CyclotronConfig,
    CyclotronSimulation,
    InstrumentConfig,
    Particle,
    measure_beam,
)
from cyclotron_sim.plots import save_figures
from cyclotron_sim.report import (
    format_report,
    write_csv_files,
    write_excel,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = CyclotronConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Simulate cyclotron beam dynamics and the response of a pickup "
            "electrode, Faraday cup and RF monitor."
        )
    )
    parser.add_argument("--protons", type=int, default=defaults.particle.protons)
    parser.add_argument("--neutrons", type=int, default=defaults.particle.neutrons)
    parser.add_argument(
        "--charge", type=int, default=defaults.particle.charge_number
    )
    parser.add_argument(
        "--b", type=float, default=defaults.magnetic_field, help="magnetic field (T)"
    )
    parser.add_argument(
        "--f-mhz", type=float, default=defaults.rf_frequency / 1.0e6,
        help="RF frequency (MHz)",
    )
    parser.add_argument(
        "--voltage", type=float, default=defaults.acceleration_voltage,
        help="acceleration voltage (V)",
    )
    parser.add_argument(
        "--gap-mm", type=float, default=defaults.gap_width * 1.0e3,
        help="dee gap width (mm)",
    )
    parser.add_argument(
        "--phase-deg", type=float, default=defaults.rf_phase_deg,
        help="RF phase (deg)",
    )
    parser.add_argument(
        "--waveform", choices=("sine", "square"), default=defaults.waveform
    )
    parser.add_argument(
        "--reduction-pct", type=float, default=defaults.voltage_reduction_pct,
        help="reduction of the nominal gap voltage (percent)",
    )
    parser.add_argument(
        "--x0-mm", type=float, default=defaults.x0 * 1.0e3,
        help="initial x position (mm)",
    )
    parser.add_argument(
        "--y0-mm", type=float, default=defaults.y0 * 1.0e3,
        help="initial y position (mm)",
    )
    parser.add_argument("--vx0", type=float, default=defaults.vx0)
    parser.add_argument("--vy0", type=float, default=defaults.vy0)
    parser.add_argument(
        "--t-us", type=float, default=defaults.t_end * 1.0e6,
        help="maximum simulated time (us)",
    )
    parser.add_argument(
        "--dt-ps", type=float, default=defaults.time_step * 1.0e12,
        help="maximum integration step (ps)",
    )
    parser.add_argument("--relativistic", action="store_true")
    parser.add_argument(
        "--cup", action="store_true", help="insert the Faraday cup"
    )
    parser.add_argument(
        "--cup-radius-mm", type=float, default=defaults.cup_radius_mm,
        help="Faraday cup radial position (mm)",
    )
    parser.add_argument(
        "--beam-particles", type=float, default=1.0e7,
        help="number of ions represented by the tracked particle",
    )
    parser.add_argument(
        "--noise", type=float, default=0.01, help="pickup noise (V rms)"
    )
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="directory for plots, CSV and Excel output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    particle = Particle(
        charge_number=args.charge,
        protons=args.protons,
        neutrons=args.neutrons,
    )
    config = CyclotronConfig(
        particle=particle,
        magnetic_field=args.b,
        rf_frequency=args.f_mhz * 1.0e6,
        acceleration_voltage=args.voltage,
        gap_width=args.gap_mm * 1.0e-3,
        rf_phase_deg=args.phase_deg,
        waveform=args.waveform,
        x0=args.x0_mm * 1.0e-3,
        y0=args.y0_mm * 1.0e-3,
        vx0=args.vx0,
        vy0=args.vy0,
        t_end=args.t_us * 1.0e-6,
        time_step=args.dt_ps * 1.0e-12,
        relativistic=args.relativistic,
        voltage_reduction_pct=args.reduction_pct,
        cup_enabled=args.cup,
        cup_radius_mm=args.cup_radius_mm,
    )
    instrument = InstrumentConfig(
        pickup_noise_rms_v=args.noise,
        beam_particles=args.beam_particles,
        seed=args.seed,
    )

    simulation = CyclotronSimulation(config)
    result = simulation.run()
    measurement = measure_beam(result, instrument)
    report = format_report(result, measurement)
    print(report)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.txt").write_text(report, encoding="utf-8")
    plots = save_figures(result, measurement, output)
    tables = write_csv_files(result, measurement, output)
    workbook = write_excel(result, measurement, output / "results.xlsx")

    print("\nSaved output:")
    for path in [*plots, *tables, workbook]:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
