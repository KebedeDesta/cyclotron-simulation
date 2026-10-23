# Physics and Instrumentation Notes

This document describes the model implemented in `cyclotron_sim` and the
assumptions behind the synthetic measurements.

## Coordinate system and fields

- Magnetic field: `B = (0, 0, Bz)` with `Bz > 0`.
- Dee gap: parallel to the `x` axis; the particle crosses it along `y`.
- Electric field: `E = (0, Ey(t), 0)` inside `|y| <= d/2`, zero outside.
- Chamber: circle of radius `R` in the `x-y` plane.
- State vector: `(x, y, vx, vy)`.

The RF voltage is

```text
U(t) = U0 * (1 - reduction/100) * waveform(2*pi*f*t - phi)
```

where `waveform` is `sin` for the sinusoidal mode and `sign(sin(...))` for the
square-wave mode. The gap field is `Ey = U(t) / d`.

## Equations of motion

Non-relativistic mode:

```text
dx/dt  = vx
dy/dt  = vy
dvx/dt =  omega * vy
dvy/dt =  q * Ey / m - omega * vx
```

Relativistic mode:

```text
dx/dt  = vx
dy/dt  = vy
dvx/dt =  omega * vy / gamma
dvy/dt =  q * Ey / (gamma * m) - omega * vx / gamma
gamma  =  1 / sqrt(1 - (vx^2 + vy^2)/c^2)
```

with `omega = q * Bz / m`. The particle mass is computed from the proton and
neutron counts; the charge is `Z * e`.

## Numerical integration

The trajectory is integrated with `scipy.integrate.solve_ivp` using the
`DOP853` explicit Runge-Kutta method. The default tolerances are
`rtol = 1e-9` and `atol = 1e-11`; the maximum step is the configured
`time_step` (default `1 ns`). The output is sampled on a uniform grid up to
`t_end`.

Two stopping conditions are applied:

1. **Extraction** — an event fires when `sqrt(x^2 + y^2) - R = 0`, with a
   positive direction. The event state is appended to the trajectory.
2. **Faraday cup** — if the cup is enabled, the first upward crossing of
   `sqrt(x^2 + y^2) = r_cup` truncates the trajectory at the interpolated hit
   point.

The integration is single-particle. The bunch charge used by the
instrumentation model is `beam_particles * q`, not a space-charge force.

## Derived beam quantities

- Speed: `v = sqrt(vx^2 + vy^2)`.
- Radius: `r = sqrt(x^2 + y^2)`.
- Kinetic energy: `0.5 * m * v^2` (non-relativistic) or
  `(gamma - 1) * m * c^2` (relativistic).
- Path length: cumulative sum of `sqrt(dx^2 + dy^2)`.
- Turn boundaries: unwrapped polar angle; a new turn starts at each multiple of
  `2*pi`.
- Gap crossings: sign changes of `y`; the crossing time is linearly
  interpolated and the RF phase and energy gain are evaluated there.

## Pickup electrode

One Gaussian pulse is generated per revolution:

```text
V_pickup(t) = A * exp(-(t - t_i)^2 / (2*sigma^2))
```

The pulse times include Gaussian timing jitter. The signal is sampled at
`sample_rate_hz`, Gaussian voltage noise is added, and the frequency is
estimated in two ways:

- **Peak-time estimate** — detect pulses with `scipy.signal.find_peaks`, fit a
  line to pulse index versus time, and convert the slope to a frequency.
- **FFT estimate** — compute the spectrum of the sampled pickup signal and
  locate the dominant peak in the expected band.

The peak-time estimate is the primary reported pickup frequency; the FFT value
is retained as a cross-check and for the diagnostic figure.

## Faraday cup

The cup is a radial beam stop at `r_cup`. When the beam reaches it, the model
reports the collected bunch charge, a Gaussian current pulse and the number of
hits. The cup can be used to study low-energy beams that do not reach the
chamber wall.

## Bρ energy reconstruction

For a particle of charge `q` and mass `m` in a uniform field, the momentum is
`p = q * B * r`. The reconstructed kinetic energy is obtained from the
relativistic or non-relativistic energy-momentum relation, using the measured
radius and a field value perturbed by `field_relative_error`. This provides a
second, independent energy estimate that can be compared with the simulated
final energy.

## Validation

The default operating point is a proton with `B = 0.185 T`,
`f_RF = 2.82 MHz`, `U = 1000 V` and `d = 20 mm`. The simulation extracts at
`R = 70 mm` after three complete turns with `E = 6.8098 keV`. The pickup
frequency is `2.82047 MHz`, compared with the theoretical cyclotron frequency
`2.82036 MHz` (relative error `+0.004 %`). The Bρ estimate is `6.8338 keV`
(`+0.352 %` relative to the simulated energy), which is consistent with the
configured radius resolution and field uncertainty.

## Limitations

- Single particle, two transverse dimensions.
- No space charge, image charges, fringe fields or magnet errors.
- Idealized gap and chamber geometry.
- Synthetic detector responses; no calibration to a specific instrument.
- The square-wave RF model is an ideal sign function, not a band-limited
  waveform.
