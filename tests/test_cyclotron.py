"""Unit tests for the cyclotron physics and instrumentation models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyclotron_sim import (
    CyclotronConfig,
    CyclotronSimulation,
    InstrumentConfig,
    Particle,
    measure_beam,
)


class CyclotronFrequencyTests(unittest.TestCase):
    def test_proton_frequency_matches_ui_default(self) -> None:
        config = CyclotronConfig()
        self.assertAlmostEqual(
            config.cyclotron_frequency / 1.0e6, 2.82036, places=4
        )

    def test_resonance_detuning_is_small(self) -> None:
        config = CyclotronConfig()
        config.rf_frequency = config.cyclotron_frequency
        self.assertAlmostEqual(config.detuning, 0.0, places=12)

    def test_invalid_magnetic_field(self) -> None:
        with self.assertRaises(ValueError):
            CyclotronConfig(magnetic_field=0.0)

    def test_cup_outside_chamber(self) -> None:
        with self.assertRaises(ValueError):
            CyclotronConfig(cup_enabled=True, cup_radius_mm=80.0)

    def test_custom_particle_mass(self) -> None:
        alpha = Particle(charge_number=2, protons=2, neutrons=2)
        proton = Particle()
        self.assertAlmostEqual(alpha.mass / proton.mass, 4.0028, places=3)
        self.assertAlmostEqual(alpha.charge / proton.charge, 2.0)


class BeamDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = CyclotronSimulation(CyclotronConfig()).run()

    def test_extraction_at_chamber_boundary(self) -> None:
        self.assertEqual(self.result.exit_reason, "extracted")
        self.assertAlmostEqual(
            self.result.final_radius_mm, 70.0, delta=0.05
        )

    def test_energy_is_physically_reasonable(self) -> None:
        self.assertGreater(self.result.final_energy_kev, 5.0)
        self.assertLess(self.result.final_energy_kev, 10.0)
        self.assertLess(self.result.final_speed, 2.99792458e8)

    def test_turns_and_crossings(self) -> None:
        self.assertGreaterEqual(self.result.n_turns, 3)
        self.assertGreaterEqual(self.result.gap_crossings_t.size, 6)
        self.assertAlmostEqual(
            self.result.energy_gain_per_turn_kev, 1.85, delta=0.2
        )

    def test_path_length_exceeds_orbit_circumference(self) -> None:
        self.assertGreater(self.result.total_path_length, 0.5)

    def test_crossing_phase_is_folded(self) -> None:
        self.assertGreaterEqual(self.result.mean_crossing_phase, 0.0)
        self.assertLessEqual(self.result.mean_crossing_phase, 3.141592653589793)

    def test_square_wave_runs(self) -> None:
        config = CyclotronConfig(waveform="square")
        result = CyclotronSimulation(config).run()
        self.assertGreater(result.final_energy_kev, 1.0)

    def test_relativistic_run_is_finite(self) -> None:
        config = CyclotronConfig(relativistic=True)
        result = CyclotronSimulation(config).run()
        self.assertTrue((result.energy_j >= 0).all())
        self.assertLess(result.final_speed, 2.99792458e8)


class FaradayCupTests(unittest.TestCase):
    def test_cup_stops_beam_at_requested_radius(self) -> None:
        config = CyclotronConfig(cup_enabled=True, cup_radius_mm=32.0)
        result = CyclotronSimulation(config).run()
        measurement = measure_beam(result, InstrumentConfig())
        self.assertEqual(result.exit_reason, "faraday cup")
        self.assertAlmostEqual(result.final_radius_mm, 32.0, delta=0.01)
        self.assertEqual(result.cup_hit_times.size, 1)
        self.assertGreater(measurement.cup_charge_c, 0.0)
        self.assertGreater(measurement.cup_peak_current_a, 0.0)

    def test_no_cup_means_no_cup_signal(self) -> None:
        result = CyclotronSimulation(CyclotronConfig()).run()
        measurement = measure_beam(result, InstrumentConfig())
        self.assertEqual(measurement.cup_charge_c, 0.0)
        self.assertEqual(measurement.cup_peak_current_a, 0.0)
        self.assertFalse(measurement.cup_current_a.any())


class InstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = CyclotronSimulation(CyclotronConfig()).run()
        cls.measurement = measure_beam(cls.result, InstrumentConfig())

    def test_pickup_frequency_within_one_percent(self) -> None:
        measured = self.measurement.pickup_frequency_hz
        expected = self.result.config.cyclotron_frequency
        self.assertLess(abs(measured - expected) / expected, 0.01)

    def test_fft_frequency_within_five_percent(self) -> None:
        measured = self.measurement.fft_frequency_hz
        expected = self.result.config.cyclotron_frequency
        self.assertLess(abs(measured - expected) / expected, 0.05)

    def test_energy_from_orbit_fit_within_two_percent(self) -> None:
        measured = self.measurement.measured_final_energy_kev
        expected = self.result.final_energy_kev
        self.assertLess(abs(measured - expected) / expected, 0.02)

    def test_measurement_uncertainty_is_reported(self) -> None:
        self.assertGreater(
            self.measurement.measured_energy_uncertainty_kev, 0.0
        )

    def test_sampling_axis_matches_duration(self) -> None:
        sample_period = 1.0 / self.measurement.instrument.sample_rate_hz
        self.assertLessEqual(self.measurement.t[-1], self.result.duration)
        self.assertLess(
            self.result.duration - self.measurement.t[-1], sample_period
        )

    def test_noise_seed_is_reproducible(self) -> None:
        first = measure_beam(self.result, InstrumentConfig(seed=7))
        second = measure_beam(self.result, InstrumentConfig(seed=7))
        self.assertTrue(
            (first.pickup_voltage_v == second.pickup_voltage_v).all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
