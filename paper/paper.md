---
title: 'triples-sigfast: A GIL-Free, Numba JIT-Compiled Data Analysis Engine for Simulation-Based Physics Research'
tags:
  - Python
  - nuclear physics
  - Monte Carlo simulation
  - Geant4
  - FLUKA
  - MCNP
  - SERPENT
  - radiation shielding
  - signal processing
  - high-performance computing
authors:
  - name: Samdani Sayam
    affiliation: 1
affiliations:
  - name: TripleS Studio, Independent Researcher
    index: 1
date: 2026-04-08
bibliography: paper.bib
---

# Summary

Python's Global Interpreter Lock (GIL) fundamentally limits scientific
data analysis to a single CPU core, regardless of hardware capability.
For simulation-based physics research — where Geant4, FLUKA, MCNP, and
SERPENT routinely produce datasets of 100 million or more particle tracks —
this bottleneck transforms minutes of simulation time into hours of
analysis time.

`triples-sigfast` eliminates this bottleneck by applying Numba
Just-In-Time (JIT) compilation with `@njit(parallel=True, nogil=True)`
decorators to transform pure Python analysis functions into native machine
code that executes across all available CPU cores simultaneously. The
library processes 100 million floating-point data points in 1.225 seconds
— a 30–40× speedup over equivalent Pandas implementations — without
requiring researchers to write a single line of C or C++ code.

# Statement of Need

Simulation-based physics research produces large Monte Carlo datasets
that require a reproducible, standards-compliant analysis pipeline between
raw simulation output and publication-ready results. Existing tools address
parts of this problem: `uproot` [@uproot] reads ROOT files, `scipy`
[@scipy] provides general signal processing, and `numpy` [@numpy] enables
array operations. However, no existing library simultaneously provides:

1. GIL-free parallel execution on 100M+ row Monte Carlo datasets
2. Built-in nuclear physics domain knowledge using international standards
   (ICRP 74, ANSI/ANS-6.4.3, NIST XCOM, NUBASE2020)
3. Native readers for all major simulation code output formats
   (Geant4, FLUKA, MCNP, SERPENT)
4. Monte Carlo statistical convergence validation
5. A beginner-accessible API and command-line interface

`triples-sigfast` fills this gap, providing a unified, standards-compliant
data analysis layer that bridges simulation output and publication-ready
results.

# Core Innovation: GIL-Free Parallel Execution

The Python GIL prevents multiple threads from executing Python bytecode
simultaneously. `triples-sigfast` releases the GIL by compiling analysis
functions to native machine code via Numba:

```pythonfrom numba import njit, prange@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _numba_rolling_avg(data: np.ndarray, window_size: int):
n = len(data)
result = np.empty(n - window_size + 1, dtype=np.float64)
for i in prange(n - window_size + 1):  # true parallel loop
window_sum = 0.0
for j in range(window_size):
window_sum += data[i + j]
result[i] = window_sum / window_size
return result

The `nogil=True` flag releases the GIL before execution. The `prange`
function distributes loop iterations across all CPU cores. The compiled
binary is cached after the first call, eliminating recompilation overhead
on subsequent invocations.

# Features

## Signal Processing

The core module provides GIL-free implementations of:

- **Rolling average** — JIT-compiled sliding window mean
- **Exponential Moving Average (EMA)** — GIL-free recursive filter
- **Z-score anomaly detection** — parallelised outlier detection
- **Savitzky-Golay filter** — peak-preserving polynomial smoothing [@savitzky1964]
- **Peak detection** — automatic gamma ray line identification
- **Flux-to-dose conversion** — ICRP 74 standard [@icrp74]
- **Shielding attenuation** — Beer-Lambert law, 9 NIST XCOM materials

A critical design decision distinguishes `triples-sigfast` from
general-purpose signal processing libraries: the Savitzky-Golay filter
is preferred over rolling average for nuclear energy spectra. Rolling
average shifts peak positions by fitting a rectangular window, while
Savitzky-Golay preserves peak position and amplitude through polynomial
least-squares fitting — a property of fundamental importance where peak
energy encodes the physical identity of isotopes and gamma ray lines.

## Monte Carlo Statistics

The statistics module implements the standard MCNP convergence criterion
[@mcnp], computing per-bin relative error R = 1/√N, Figure of Merit
FOM = 1/R²T, automatic convergence classification (R < 0.05), and
GUM-compliant uncertainty propagation [@gum] through detector efficiency
corrections.

## Nuclear Physics

Radiation shielding calculations implement the Geometric Progression (GP)
buildup factor method per ANSI/ANS-6.4.3-1991 [@ansans643], correcting
the systematic underestimation of dose in Beer-Lambert attenuation by
20–50% in thick shields. Buildup factors are tabulated for six materials
across 0.1–10.0 MeV.

The Watt fission spectrum [@watt1952]:

$$N(E) = C \cdot e^{-E/a} \cdot \sinh\left(\sqrt{bE}\right)$$

is implemented for Cf-252, U-235, U-238, and Pu-239, enabling direct
validation of Geant4 source definitions against analytical predictions.

Biological dose calculations follow ICRP Publication 74 (1996) [@icrp74]
tabulated flux-to-ambient-dose-equivalent H*(10) conversion coefficients
with 48-point neutron and 26-point gamma energy tables, interpolated via
log-log regression.

The isotope database is compiled from NUBASE2020 [@nubase2020], providing
half-lives, decay modes, thermal neutron cross sections, resonance
integrals, gamma energies, spontaneous fission neutron yields, and
activity calculations.

## Simulation File Readers

The universal `SimReader` class auto-detects and delegates to
format-specific backends:

- `RootReader` — Geant4 ROOT via uproot [@uproot]
- `FlukaReader` — FLUKA USRBIN/USRBDX/USRTRACK
- `MCNPReader` — MCNP6 MCTAL tally format
- `SerpentReader` — SERPENT2 detector output, k-effective, burnup

All readers expose a unified API: `get_spectrum()`, `get_tally()`,
`summary()`, and `export_csv()`.

## Visualization and Reporting

The `PhysicsPlot` class generates publication-quality figures with
journal-specific style presets (Physical Review, Nature, thesis,
presentation) in PDF, PNG, and SVG formats with LaTeX caption generation.

The `AutoReport` class generates complete multi-page PDF analysis reports
from simulation files in three lines of user code:

```pythonreport = AutoReport(title="Shielding Analysis")
report.add_simulation("pb.root", label="Lead")
report.generate("report.pdf")

## Command-Line Interface

The `sigfast` CLI provides seven commands accessible to researchers
without programming experience: `analyze`, `compare`, `dose`, `shield`,
`guide`, `report`, and `info`.

# Performance

| Dataset Size | Execution Time | Peak RAM |
|---|---|---|
| 1,000,000 rows | 0.339 s | 192 MB |
| 10,000,000 rows | 0.281 s | 404 MB |
| 50,000,000 rows | 0.747 s | 940 MB |
| 100,000,000 rows | 1.225 s | 1,596 MB |

# Testing

The library comprises 1,053 statements across 23 Python modules,
validated by 385 unit and integration tests achieving 100% code coverage,
tested on Ubuntu, macOS, and Windows across Python 3.10, 3.11, and 3.12
via a 9-job GitHub Actions CI matrix.

# Acknowledgements

The author thanks the Numba [@numba], NumPy [@numpy], and uproot [@uproot]
development teams whose foundational work made this library possible.

# References
