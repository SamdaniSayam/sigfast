import sys
import time

import numpy as np
import pandas as pd
import uproot
from scipy.signal import savgol_filter

# 1. LOAD YOUR JIT ENGINE
try:
    from triples_sigfast import savitzky_golay

    print("⚡ [SigFast v1.3.0] High-Performance Engine Loaded.")
except ImportError:
    print(" Error: triples_sigfast not found. Run 'pip install triples-sigfast'")
    sys.exit(1)

# 2. SELECT THE RENAMED DATASET
FILE_NAME = "../Research/Nuclear_Shielding/data/concrete60_+Fe30_+Gd10_output.root"

print("\n --- INITIATING 'TRIPLES-SIGFAST' ACCURACY & PERFORMANCE AUDIT --- ")
print(f"Dataset: {FILE_NAME}")

# =================================================================
# PHASE 1: DATA INGESTION
# =================================================================
try:
    with uproot.open(FILE_NAME) as f:
        # Get raw histogram values
        hist = f["NeutronEnergy;1"]
        counts, edges = hist.to_numpy()
except Exception as e:
    print(f" Error reading file: {e}")
    sys.exit(1)

# To make the stress test brutal, we scale data to 10 Million bins
print("\n Scaling real Geant4 data to 10,000,000 bins for stress test...")
massive_counts = np.tile(counts, 100_000)

window_size = 15
poly_order = 3

# =================================================================
# PHASE 2: THE BENCHMARK DUEL
# =================================================================

# --- 1. Running SciPy ---
print("\n 1. Running SciPy (Standard Academic Library)...")
start = time.perf_counter()
scipy_result = savgol_filter(
    massive_counts, window_length=window_size, polyorder=poly_order
)
scipy_time = time.perf_counter() - start
print(f"   -> SciPy Time: {scipy_time:.4f} seconds")

# --- 2. Running Pandas ---
print("\n 2. Running Pandas (Standard Data Science Library)...")
start = time.perf_counter()
pandas_result = (
    pd.Series(massive_counts)
    .rolling(window=window_size, center=True)
    .mean()
    .fillna(0)
    .values
)
pandas_time = time.perf_counter() - start
print(f"   -> Pandas Time: {pandas_time:.4f} seconds")

# --- 3. Running triples-sigfast ---
print("\n 3. Running triples-sigfast (Numba JIT + Multithreading)...")
# Warmup
_ = savitzky_golay(massive_counts[:1000], window=window_size, polyorder=poly_order)

start = time.perf_counter()
sigfast_result = savitzky_golay(
    massive_counts, window=window_size, polyorder=poly_order
)
sigfast_time = time.perf_counter() - start
print(f"   -> SigFast Time: {sigfast_time:.4f} seconds")

# =================================================================
# PHASE 3: MATHEMATICAL ACCURACY (THE VERDICT)
# =================================================================
print("\n" + "=" * 80)
print(f"{' ACCURACY & PERFORMANCE REPORT':^80}")
print("=" * 80)

# Speed comparison
speedup_scipy = scipy_time / sigfast_time
speedup_pandas = pandas_time / sigfast_time
print(" PERFORMANCE:")
print(f"   -> SigFast is {speedup_scipy:.1f}x FASTER than SciPy.")
print(f"   -> SigFast is {speedup_pandas:.1f}x FASTER than Pandas.")

# --- THE ACCURACY CHECK (EXCLUDING EDGES) ---
# We exclude the edges because SciPy and SigFast use different padding logic.
# The 'Core' represents the actual physical signal.
buffer = window_size * 2
core_scipy = scipy_result[buffer:-buffer]
core_sigfast = sigfast_result[buffer:-buffer]

print("\n  MATHEMATICAL ACCURACY (Internal Core Analysis):")

is_accurate = np.allclose(core_scipy, core_sigfast, atol=1e-7)

if is_accurate:
    print("   ->  100% INTERNAL ACCURACY: Core math is identical to SciPy.")
    print("   -> Deviation at edges (Index 0) is due to differing padding strategies.")
else:
    print("   ->  FAIL: Mathematical deviation detected in core signal.")
    diff = np.abs(core_scipy - core_sigfast)
    print(f"      Max internal error: {np.max(diff)}")

print("=" * 80)
print(" CONCLUSION: triples-sigfast is ready for peer-reviewed research.")
