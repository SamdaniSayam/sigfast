import numpy as np
import pandas as pd
from numba import njit, prange

# --- THE SMART WRAPPER (Pandas Bridge) ---
def ensure_numpy(data):
    """Safely converts inputs (Lists, Pandas Series) into flat NumPy arrays."""
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.to_numpy().flatten()
    elif isinstance(data, list):
        return np.array(data, dtype=np.float64)
    return data.astype(np.float64)

# --- 1. ROLLING AVERAGE ---
@njit(parallel=True, fastmath=True)
def _numba_rolling_avg(data, window_size):
    n = len(data)
    result = np.empty(n - window_size + 1, dtype=np.float64)
    for i in prange(n - window_size + 1):
        window_sum = 0.0
        for j in range(window_size):
            window_sum += data[i + j]
        result[i] = window_sum / window_size
    return result

def rolling_average(data, window_size: int):
    if window_size <= 0: raise ValueError("Window size must be > 0.")
    clean_data = ensure_numpy(data)
    result = _numba_rolling_avg(clean_data, window_size)
    if isinstance(data, pd.Series): return pd.Series(result, index=data.index[window_size - 1:])
    return result

# --- 2. EXPONENTIAL MOVING AVERAGE (EMA) ---
@njit(fastmath=True)
def _numba_ema(data, alpha):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

def ema(data, span: int):
    if span <= 0: raise ValueError("Span must be > 0")
    clean_data = ensure_numpy(data)
    alpha = 2.0 / (span + 1.0)
    result = _numba_ema(clean_data, alpha)
    if isinstance(data, pd.Series): return pd.Series(result, index=data.index)
    return result

# --- 3. Z-SCORE ANOMALY DETECTION ---
@njit(parallel=True, fastmath=True)
def _numba_zscore_anomalies(data, threshold):
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    is_anomaly = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        z_score = abs(data[i] - mean_val) / std_val
        if z_score > threshold: is_anomaly[i] = True
    return is_anomaly

def detect_anomalies(data, threshold: float = 3.0):
    clean_data = ensure_numpy(data)
    result = _numba_zscore_anomalies(clean_data, threshold)
    if isinstance(data, pd.Series): return pd.Series(result, index=data.index)
    return result

# --- 4. QUANT TRADING: EMA CROSSOVER ---
@njit(fastmath=True)
def _numba_crossover(fast_ema, slow_ema):
    n = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8) 
    for i in range(1, n):
        if fast_ema[i] > slow_ema[i] and fast_ema[i-1] <= slow_ema[i-1]: signals[i] = 1 # BUY
        elif fast_ema[i] < slow_ema[i] and fast_ema[i-1] >= slow_ema[i-1]: signals[i] = -1 # SELL
    return signals

def ema_crossover_strategy(data, fast_span: int = 9, slow_span: int = 21):
    clean_data = ensure_numpy(data)
    fast_ema = _numba_ema(clean_data, 2.0 / (fast_span + 1.0))
    slow_ema = _numba_ema(clean_data, 2.0 / (slow_span + 1.0))
    signals = _numba_crossover(fast_ema, slow_ema)
    return fast_ema, slow_ema, signals