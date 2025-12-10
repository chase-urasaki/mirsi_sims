# MIRSI Simulator

A Python-based simulator for the Mid-Infrared Spectrometer and Imager (MIRSI) instrument, designed to model infrared detector performance, sky background, and chop-nod observing techniques.

## Overview

This simulator provides tools for:
- Sky background noise modeling with slow atmospheric drift
- Detector noise simulation (read noise, dark current, Poisson noise)
- Point source injection with chop-nod patterns
- Frame subtraction for background removal
- Exposure sequence generation and analysis

## Features

### Sky Background Simulation (`sky_sim.py`)
- **Planck function-based sky background calculation** for N-band (10 µm)
- **Detector noise modeling**: read noise, dark current, quantum efficiency
- Configurable detector parameters (240×320 pixel array by default)
- Single exposure and sequence generation

### Exposure Sequences (`exposure_sequences.py`)
- **Generate sequences of exposures** with realistic noise properties
- **Slow atmospheric drift simulation** using Ornstein-Uhlenbeck process
  - Configurable correlation timescale (τ) and RMS amplitude
  - Models correlated sky background variations
- **Frame coadding** with variance tracking
- Statistical analysis tools (mean, variance per frame)

### Chop-Nod Simulation (`chop_and_nod_sim.py`)
- **Point source injection** with magnitude-to-flux conversion
- **True chop-nod pattern implementation**:
  - 4-frame cycle: (nod−, chop−), (nod−, chop+), (nod+, chop−), (nod+, chop+)
  - Configurable chop and nod throws
- **Pairwise frame subtraction** for sky background removal
- Gaussian PSF modeling with configurable FWHM

## Installation

### Requirements
```bash
python >= 3.8
numpy
matplotlib
```

### Setup
```bash
git clone https://github.com/chase-urasaki/mirsi_sims.git
cd mirsi_sims
pip install numpy matplotlib
```

## Quick Start

### Generate a Simple Exposure Sequence
```python
from exposure_sequences import make_exposure_sequence

# Generate 10 frames with 20ms exposure time
exposures = make_exposure_sequence(
    n_exposures=10,
    exposure_time=0.02,
    self_similar=True
)
```

### Add Atmospheric Drift
```python
import numpy as np
from exposure_sequences import make_exposure_sequence

rng = np.random.default_rng(42)

# Generate sequence with correlated atmospheric drift
exposures = make_exposure_sequence(
    n_exposures=20,
    exposure_time=0.02,
    drift={"tau": 0.5, "amp_frac": 0.03, "rng": rng},
    self_similar=True
)
```

**Drift parameters:**
- `tau`: Correlation timescale in seconds (e.g., 0.5s = persistent for ~0.5s)
- `amp_frac`: RMS fractional amplitude (e.g., 0.03 = 3% variation)

### Chop-Nod with Point Source
```python
from exposure_sequences import make_exposure_sequence
from chop_and_nod_sim import inject_point_source_chop_nod, subtract_frames

# Create exposure sequence (must be multiple of 4 for full chop-nod cycle)
exposures = make_exposure_sequence(8, 0.02, self_similar=False)

# Inject a point source with chop-nod pattern
exposures = inject_point_source_chop_nod(
    exposures,
    position=(120, 160),  # (y, x) pixel coordinates
    mag=2.0,              # N-band magnitude
    extent=3.0,           # FWHM in pixels
    exposure_time=0.02,
    chop_throw=20,        # pixels
    nod_throw=40          # pixels
)

# Subtract chopped frame pairs
subtracted = subtract_frames(exposures)
```

## Project Structure

```
mirsi_sims/
├── sky_sim.py              # Sky background and detector noise
├── exposure_sequences.py   # Exposure generation with drift
├── chop_and_nod_sim.py    # Point source injection and subtraction
└── README.md              # This file
```

## Physics & Methods

### Sky Background Model
The simulator uses the Planck function to compute thermal emission from a 273K atmosphere at 10 µm:

```
B(λ, T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
```

Sky background rate accounts for:
- Telescope collecting area (3m diameter)
- Pixel solid angle (50 µm pixels at f/37)
- N-band filter (5 µm bandpass)
- System transmission (30%)

### Atmospheric Drift
Modeled as an Ornstein-Uhlenbeck (AR(1)) process:

```
x[i] = ρ·x[i-1] + √(1-ρ²)·ε[i]
```

where `ρ = exp(-Δt/τ)` controls correlation strength.

### Chop-Nod Pattern
Standard 4-position chop-nod cycle eliminates sky background and telescope emission:

| Frame | Nod | Chop | Description |
|-------|-----|------|-------------|
| 0     | −   | −    | Nod offset -, Chop offset - |
| 1     | −   | +    | Nod offset -, Chop offset + |
| 2     | +   | −    | Nod offset +, Chop offset - |
| 3     | +   | +    | Nod offset +, Chop offset + |

Pairwise subtraction (0-1), (2-3) removes chopped sky background.
Final nod subtraction removes residual systematic effects.

## Default Parameters

### Detector
- **Read noise**: 800 e⁻ rms
- **Dark current**: 50 e⁻/s/pixel
- **Quantum efficiency**: 0.5 (50%)
- **Array size**: 240 × 320 pixels
- **Pixel size**: 50 µm

### Telescope & Optics
- **Aperture**: 3.0 m diameter
- **f-number**: f/37
- **System throughput**: 30%

### N-band Filter
- **Central wavelength**: 10 µm
- **Bandpass**: 5 µm (FWHM)
- **Zero-point flux**: 36 Jy (Vega system)

## Examples

Each module contains test code that runs when executed directly:

```bash
# Test sky background simulation
python sky_sim.py

# Test exposure sequences with drift
python exposure_sequences.py

# Test chop-nod pattern
python chop_and_nod_sim.py
```

## API Reference

### `make_exposure_sequence(n_exposures, exposure_time, drift=None, self_similar=False)`

Generate a sequence of detector exposures with realistic noise.

**Parameters:**
- `n_exposures` (int): Number of exposures to simulate
- `exposure_time` (float): Exposure time in seconds for each frame
- `drift` (None, dict, or array): Atmospheric drift parameters or custom drift array
- `self_similar` (bool): If True, reuses one simulator instance (faster)

**Returns:**
- `exposures` (ndarray): 3D array of shape (n_exposures, 240, 320)

### `inject_point_source_chop_nod(exposure_sequence, position, mag, extent, ...)`

Inject a point source with chop-nod pattern into exposure sequence.

**Parameters:**
- `exposure_sequence` (ndarray): 3D array of exposures
- `position` (tuple): (y, x) pixel coordinates of source
- `mag` (float): N-band magnitude
- `extent` (float): FWHM of PSF in pixels
- `exposure_time` (float): Exposure time in seconds
- `chop_throw` (int): Chop offset in pixels
- `nod_throw` (int): Nod offset in pixels

**Returns:**
- `exposure_sequence` (ndarray): Modified exposure sequence with injected source

### `subtract_frames(exposure_sequence)`

Perform pairwise frame subtraction for chop subtraction.

**Parameters:**
- `exposure_sequence` (ndarray): 3D array of shape (n_exposures, height, width)

**Returns:**
- `subtracted_sequence` (ndarray): 3D array of shape (n_exposures//2, height, width)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

Chase M. Urasaki
- GitHub: [@chase-urasaki](https://github.com/chase-urasaki)
- Repository: [mirsi_sims](https://github.com/chase-urasaki/mirsi_sims)

## Acknowledgments

- MIRSI instrument team at NASA Ames
- IRTF (Infrared Telescope Facility) staff

## References

1. Planck, M. (1901). "On the Law of Distribution of Energy in the Normal Spectrum"
2. Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the Theory of the Brownian Motion"
3. Rayner, J. T., et al. (2003). "SpeX: A Medium-Resolution 0.8-5.5 Micron Spectrograph and Imager for the NASA Infrared Telescope Facility"
