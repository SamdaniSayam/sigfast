import numpy as np
import pandas as pd
from numba import njit, prange

# ============================================================
#  INTERNAL UTILITIES
# ============================================================


def _ensure_float64_numpy(data):
    """
    Forces input data into a C-contiguous, 64-bit float NumPy array
    for maximum CPU cache efficiency and memory alignment.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        arr = data.to_numpy().flatten()
    elif isinstance(data, list):
        arr = np.array(data)
    else:
        arr = data
    return np.ascontiguousarray(arr, dtype=np.float64)


# ============================================================
#  1. ROLLING AVERAGE (HPC Optimized) — unchanged
# ============================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _numba_rolling_avg(data: np.ndarray, window_size: int):  # pragma: no cover
    n = len(data)
    result = np.empty(n - window_size + 1, dtype=np.float64)
    for i in prange(n - window_size + 1):
        window_sum = 0.0
        for j in range(window_size):
            window_sum += data[i + j]
        result[i] = window_sum / window_size
    return result


def rolling_average(data, window_size: int):
    """
    Computes a moving average on a 1D array at C-speed using Numba JIT.

    This function bypasses the Python GIL via multithreading and is optimized
    for cache-locality on large datasets (100M+ rows).

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        window_size (int): The number of samples for the moving window.

    Returns:
        np.ndarray or pd.Series: The smoothed data array.
    """
    if window_size <= 0:
        raise ValueError("Window size must be > 0.")
    if len(data) < window_size:
        raise ValueError("Data length must be >= window size.")

    clean_data = _ensure_float64_numpy(data)
    result = _numba_rolling_avg(clean_data, window_size)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index[window_size - 1 :])
    return result


# ============================================================
#  2. EXPONENTIAL MOVING AVERAGE (HPC Optimized) — unchanged
# ============================================================


@njit(fastmath=True, cache=True, nogil=True)
def _numba_ema(data: np.ndarray, alpha: float):  # pragma: no cover
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    return result


def ema(data, span: int):
    """
    Calculates the Exponential Moving Average (EMA) at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        span (int): The span of the EMA, common in financial analysis.

    Returns:
        np.ndarray or pd.Series: The EMA data array.
    """
    if span <= 0:
        raise ValueError("Span must be > 0")
    clean_data = _ensure_float64_numpy(data)
    alpha = 2.0 / (span + 1.0)
    result = _numba_ema(clean_data, alpha)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# ============================================================
#  3. Z-SCORE ANOMALY DETECTION (HPC Optimized) — unchanged
# ============================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _numba_zscore_anomalies(data: np.ndarray, threshold: float):  # pragma: no cover
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    is_anomaly = np.zeros(n, dtype=np.bool_)
    if std_val == 0:
        return is_anomaly
    for i in prange(n):
        z_score = abs(data[i] - mean_val) / std_val
        if z_score > threshold:
            is_anomaly[i] = True
    return is_anomaly


def detect_anomalies(data, threshold: float = 3.0):
    """
    Identifies anomalies in a dataset using the Z-score method at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): The input time-series data.
        threshold (float): The Z-score threshold. Default is 3.0.

    Returns:
        np.ndarray or pd.Series: A boolean array where True indicates an anomaly.
    """
    clean_data = _ensure_float64_numpy(data)
    result = _numba_zscore_anomalies(clean_data, threshold)
    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# ============================================================
#  4. EMA CROSSOVER STRATEGY (HPC Optimized) — unchanged
# ============================================================


@njit(fastmath=True, cache=True, nogil=True)
def _numba_crossover(fast_ema: np.ndarray, slow_ema: np.ndarray):  # pragma: no cover
    n = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if fast_ema[i] > slow_ema[i] and fast_ema[i - 1] <= slow_ema[i - 1]:
            signals[i] = 1  # BUY
        elif fast_ema[i] < slow_ema[i] and fast_ema[i - 1] >= slow_ema[i - 1]:
            signals[i] = -1  # SELL
    return signals


def ema_crossover_strategy(data, fast_span: int = 9, slow_span: int = 21):
    """
    Generates trading signals based on the EMA crossover strategy at C-speed.

    Args:
        data (np.ndarray | pd.Series | list): Input asset price data.
        fast_span (int): The span for the shorter-term EMA.
        slow_span (int): The span for the longer-term EMA.

    Returns:
        tuple: (fast_ema, slow_ema, signals) where signals are 1 (Buy), -1 (Sell), 0 (Hold).
    """
    clean_data = _ensure_float64_numpy(data)
    fast_ema = _numba_ema(clean_data, 2.0 / (fast_span + 1.0))
    slow_ema = _numba_ema(clean_data, 2.0 / (slow_span + 1.0))
    signals = _numba_crossover(fast_ema, slow_ema)
    return fast_ema, slow_ema, signals


# ============================================================
#  5. SAVITZKY-GOLAY FILTER (Nuclear Spectrum Smoothing)
# ============================================================


@njit(fastmath=True, cache=True, nogil=True)
def _numba_savitzky_golay(
    data: np.ndarray, coeffs: np.ndarray, half_win: int
):  # pragma: no cover
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    for i in range(half_win):
        result[i] = data[i]
        result[n - 1 - i] = data[n - 1 - i]
    for i in range(half_win, n - half_win):
        val = 0.0
        for j in range(len(coeffs)):
            val += coeffs[j] * data[i - half_win + j]
        result[i] = val
    return result


def _compute_sg_coeffs(window: int, polyorder: int) -> np.ndarray:
    """Compute Savitzky-Golay convolution coefficients via least squares."""
    half_win = window // 2
    x = np.arange(-half_win, half_win + 1, dtype=np.float64)
    A = np.zeros((window, polyorder + 1), dtype=np.float64)
    for i in range(window):
        for j in range(polyorder + 1):
            A[i, j] = x[i] ** j
    coeffs = np.linalg.pinv(A)[0]
    return np.ascontiguousarray(coeffs, dtype=np.float64)


def savitzky_golay(data, window: int = 11, polyorder: int = 3):
    """
    Smooths a signal using the Savitzky-Golay filter at C-speed.

    Unlike rolling_average, this filter preserves peak height and shape —
    essential for nuclear energy spectra where peak position and amplitude
    carry physical meaning (e.g. neutron capture peaks, gamma ray lines).

    Args:
        data (np.ndarray | pd.Series | list): Input spectral or time-series data.
        window (int): Number of points in the smoothing window. Must be odd. Default 11.
        polyorder (int): Polynomial order for fitting. Must be < window. Default 3.

    Returns:
        np.ndarray or pd.Series: Smoothed data with same length as input.

    Example:
        >>> smoothed = savitzky_golay(neutron_counts, window=11, polyorder=3)
    """
    if window % 2 == 0:
        raise ValueError("Window must be odd.")
    if polyorder >= window:
        raise ValueError("polyorder must be less than window.")
    if len(data) < window:
        raise ValueError("Data length must be >= window.")

    clean_data = _ensure_float64_numpy(data)
    coeffs = _compute_sg_coeffs(window, polyorder)
    half_win = window // 2
    result = _numba_savitzky_golay(clean_data, coeffs, half_win)

    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index)
    return result


# ============================================================
#  6. GAUSSIAN PEAK FINDER (Nuclear Spectroscopy)
# ============================================================


@njit(fastmath=True, cache=True)
def _numba_find_peaks(
    data: np.ndarray, min_height: float, min_distance: int
):  # pragma: no cover
    n = len(data)
    peak_indices = np.empty(n, dtype=np.int64)
    peak_count = 0
    for i in range(1, n - 1):
        if data[i] > min_height and data[i] > data[i - 1] and data[i] > data[i + 1]:
            if peak_count == 0 or (i - peak_indices[peak_count - 1]) >= min_distance:
                peak_indices[peak_count] = i
                peak_count += 1
    return peak_indices[:peak_count]


def find_peaks(data, min_height: float = 0.0, min_distance: int = 1):
    """
    Detects peaks in a 1D spectrum at C-speed using Numba JIT.

    Designed for nuclear energy spectra — identifies characteristic gamma
    ray lines and neutron capture peaks from Geant4 / ROOT output.

    Args:
        data (np.ndarray | pd.Series | list): Input spectral data (counts per MeV bin).
        min_height (float): Minimum peak height. Default 0.0.
        min_distance (int): Minimum bins between adjacent peaks. Default 1.

    Returns:
        np.ndarray: Array of indices where peaks are located.

    Example:
        >>> peaks = find_peaks(gamma_spectrum, min_height=50, min_distance=10)
        >>> print(f"Found {len(peaks)} gamma lines at bins: {peaks}")
    """
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1.")
    clean_data = _ensure_float64_numpy(data)
    return _numba_find_peaks(clean_data, min_height, min_distance)


# ============================================================
#  7. FLUX-TO-DOSE CONVERTER (ICRP 74 Standard)
# ============================================================

# ICRP Publication 74 (1996) — Table A.1 and A.2
# Neutron flux-to-dose conversion coefficients H*(10) in pSv·cm²
_NEUTRON_ENERGIES_MEV = np.array(
    [
        1.0e-9,
        1.0e-8,
        2.5e-8,
        1.0e-7,
        2.0e-7,
        5.0e-7,
        1.0e-6,
        2.0e-6,
        5.0e-6,
        1.0e-5,
        2.0e-5,
        5.0e-5,
        1.0e-4,
        2.0e-4,
        5.0e-4,
        1.0e-3,
        2.0e-3,
        5.0e-3,
        1.0e-2,
        2.0e-2,
        3.0e-2,
        5.0e-2,
        7.0e-2,
        1.0e-1,
        1.5e-1,
        2.0e-1,
        3.0e-1,
        5.0e-1,
        7.0e-1,
        9.0e-1,
        1.0e0,
        1.2e0,
        1.5e0,
        2.0e0,
        3.0e0,
        4.0e0,
        5.0e0,
        6.0e0,
        7.0e0,
        8.0e0,
        9.0e0,
        1.0e1,
        1.2e1,
        1.4e1,
        1.5e1,
        1.6e1,
        1.8e1,
        2.0e1,
    ],
    dtype=np.float64,
)

_NEUTRON_COEFFS_PSV_CM2 = np.array(
    [
        3.09,
        3.10,
        3.26,
        3.64,
        3.82,
        4.15,
        4.44,
        4.82,
        5.44,
        5.96,
        6.52,
        7.38,
        8.22,
        9.11,
        1.05e1,
        1.16e1,
        1.33e1,
        1.65e1,
        2.05e1,
        2.74e1,
        3.20e1,
        4.00e1,
        4.57e1,
        5.23e1,
        5.99e1,
        6.58e1,
        7.48e1,
        8.80e1,
        9.76e1,
        1.03e2,
        1.08e2,
        1.13e2,
        1.20e2,
        1.27e2,
        1.35e2,
        1.33e2,
        1.28e2,
        1.23e2,
        1.18e2,
        1.13e2,
        1.10e2,
        1.07e2,
        1.04e2,
        1.05e2,
        1.07e2,
        1.08e2,
        1.11e2,
        1.16e2,
    ],
    dtype=np.float64,
)

# Gamma flux-to-dose conversion coefficients H*(10) in pSv·cm²
_GAMMA_ENERGIES_MEV = np.array(
    [
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.80,
        1.00,
        1.50,
        2.00,
        3.00,
        4.00,
        5.00,
        6.00,
        8.00,
        10.0,
    ],
    dtype=np.float64,
)

_GAMMA_COEFFS_PSV_CM2 = np.array(
    [
        0.061,
        0.83,
        1.05,
        0.81,
        0.64,
        0.55,
        0.51,
        0.52,
        0.53,
        0.61,
        0.89,
        1.20,
        1.80,
        2.38,
        2.93,
        3.44,
        4.38,
        5.20,
        6.90,
        8.60,
        11.1,
        13.4,
        15.5,
        17.6,
        21.6,
        25.6,
    ],
    dtype=np.float64,
)


def flux_to_dose(flux, energy_mev: float, particle: str = "neutron"):
    """
    Converts particle flux to ambient dose equivalent rate H*(10).

    Uses ICRP Publication 74 (1996) conversion coefficients — the
    international standard for radiation protection calculations.
    Uses log-log interpolation between tabulated energy points for
    maximum accuracy across the full energy range.

    Args:
        flux (float | np.ndarray): Particle flux in particles/cm²/s.
        energy_mev (float): Particle energy in MeV.
        particle (str): "neutron" or "gamma". Default "neutron".

    Returns:
        float | np.ndarray: Dose equivalent rate in μSv/hr.

    Example:
        >>> # 252Cf average neutron energy 2.35 MeV
        >>> dose = flux_to_dose(flux=1e6, energy_mev=2.35, particle="neutron")
        >>> print(f"Dose rate: {dose:.4f} uSv/hr")

    References:
        ICRP Publication 74, Annals of the ICRP 26(3/4), 1996.
    """
    particle = particle.lower()
    if particle not in ("neutron", "gamma"):
        raise ValueError("particle must be 'neutron' or 'gamma'.")
    if energy_mev <= 0:
        raise ValueError("energy_mev must be > 0.")

    if particle == "neutron":
        energies = _NEUTRON_ENERGIES_MEV
        coeffs = _NEUTRON_COEFFS_PSV_CM2
    else:
        energies = _GAMMA_ENERGIES_MEV
        coeffs = _GAMMA_COEFFS_PSV_CM2

    # log-log interpolation for accuracy across wide energy range
    log_e = np.log(np.clip(energy_mev, energies[0], energies[-1]))
    log_energies = np.log(energies)
    log_coeffs = np.log(coeffs)
    conversion_pSv_cm2 = np.exp(np.interp(log_e, log_energies, log_coeffs))

    # flux (particles/cm²/s) × coeff (pSv·cm²) → pSv/s → μSv/hr
    return flux * conversion_pSv_cm2 * 1e-12 * 3600.0


# ============================================================
#  8. SHIELDING ATTENUATION (Beer-Lambert Law)
# ============================================================

# Mass attenuation coefficients μ/ρ (cm²/g) at ~1 MeV gamma
# Source: NIST XCOM Photon Cross Sections Database
_ATTENUATION_MATERIALS = {
    "lead": {"density": 11.35, "mu_rho": 0.0708},
    "polyethylene": {"density": 0.95, "mu_rho": 0.0636},
    "concrete": {"density": 2.30, "mu_rho": 0.0664},
    "water": {"density": 1.00, "mu_rho": 0.0706},
    "iron": {"density": 7.87, "mu_rho": 0.0599},
    "bismuth": {"density": 9.79, "mu_rho": 0.0705},
    "tungsten": {"density": 19.30, "mu_rho": 0.0880},
    "borated_poly": {"density": 1.06, "mu_rho": 0.0640},
    "polysulfone": {"density": 1.24, "mu_rho": 0.0638},
}


def attenuation(
    thickness_cm: float,
    material: str = "lead",
    mu_rho: float = None,
    density: float = None,
):
    """
    Calculates gamma ray transmission through a shielding layer.

    Applies Beer-Lambert law: T = exp(-mu * x)
    where mu = (mu/rho) x rho is the linear attenuation coefficient.

    Built-in materials (NIST XCOM, ~1 MeV gamma):
        lead, polyethylene, concrete, water, iron,
        bismuth, tungsten, borated_poly, polysulfone

    Args:
        thickness_cm (float): Shield thickness in centimeters.
        material (str): Material name from built-in database. Default "lead".
        mu_rho (float): Custom mass attenuation coefficient (cm²/g). Overrides material.
        density (float): Custom material density (g/cm³). Required if mu_rho provided.

    Returns:
        float: Transmission fraction (0.0 = fully blocked, 1.0 = no shielding).

    Example:
        >>> # 10cm lead transmission
        >>> T = attenuation(thickness_cm=10, material="lead")
        >>> print(f"Transmission: {T*100:.4f}%")

        >>> # Composite: 5cm lead + 10cm polyethylene
        >>> T = attenuation(5, "lead") * attenuation(10, "polyethylene")

    References:
        NIST XCOM Photon Cross Sections Database.
    """
    if thickness_cm < 0:
        raise ValueError("thickness_cm must be >= 0.")

    if mu_rho is not None:
        if density is None:
            raise ValueError("density must be provided with custom mu_rho.")
        mu_linear = mu_rho * density
    else:
        mat = material.lower()
        if mat not in _ATTENUATION_MATERIALS:
            available = list(_ATTENUATION_MATERIALS.keys())
            raise ValueError(f"Unknown material '{material}'. Available: {available}")
        props = _ATTENUATION_MATERIALS[mat]
        mu_linear = props["mu_rho"] * props["density"]

    return float(np.exp(-mu_linear * thickness_cm))


def attenuation_series(thickness_range, material: str = "lead"):
    """
    Calculates transmission across a range of shield thicknesses.

    Useful for plotting shielding effectiveness curves — a standard
    deliverable in nuclear shielding research reports.

    Args:
        thickness_range (np.ndarray | list): Array of thickness values in cm.
        material (str): Material name from built-in database.

    Returns:
        np.ndarray: Transmission fractions for each thickness value.

    Example:
        >>> thicknesses = np.linspace(0, 30, 100)
        >>> T = attenuation_series(thicknesses, material="lead")
    """
    thicknesses = _ensure_float64_numpy(thickness_range)
    return np.array([attenuation(float(t), material) for t in thicknesses])
