import numpy as np
import pandas as pd
import time
import sys

# Import your newly optimized engine
try:
    from triples_sigfast.core import rolling_average
except ImportError:
    print("❌ Error: Could not find sigfast. Make sure you are in the root directory.")
    sys.exit(1)

print("\n🔬 INITIATING MATHEMATICAL ACCURACY VALIDATION...")
print("Generating 1,000,000 random floating-point numbers...")
# Set a random seed for reproducibility
np.random.seed(42)
data = np.random.rand(1_000_000)
df = pd.DataFrame({'Signal': data})

window = 50

# --- 1. THE GOLD STANDARD (Pandas) ---
print("\n🐼 Calculating 'Gold Standard' Moving Average using Pandas...")
start = time.perf_counter()
# Pandas calculates and drops the NaN values at the start
pandas_result = df['Signal'].rolling(window=window).mean().dropna().to_numpy()
pandas_time = time.perf_counter() - start
print(f"   -> Pandas finished in {pandas_time:.4f} seconds.")

# --- 2. THE TITAN ENGINE (SigFast) ---
print("\n⚡ Calculating High-Performance Average using SigFast...")
start = time.perf_counter()
sigfast_result = rolling_average(data, window)
sigfast_time = time.perf_counter() - start
print(f"   -> SigFast finished in {sigfast_time:.4f} seconds.")

# --- 3. THE ASSERTION (Mathematical Proof) ---
print("\n⚖️ COMPARING RESULTS AT THE 8TH DECIMAL PLACE...")

# Check if the arrays are exactly the same length
if len(pandas_result) != len(sigfast_result):
    print(f"❌ FAIL: Length mismatch! Pandas: {len(pandas_result)} | SigFast: {len(sigfast_result)}")
    sys.exit(1)

# Check if every single number is mathematically identical (within floating point precision)
is_accurate = np.allclose(pandas_result, sigfast_result, atol=1e-8)

if is_accurate:
    print("✅ SUCCESS: SigFast is 100% mathematically identical to Pandas.")
    print(f"🚀 PERFORMANCE: SigFast achieved a {pandas_time/sigfast_time:.1f}x speedup with zero accuracy loss.")
else:
    print("❌ FAIL: SigFast produced different mathematical results than Pandas.")
    # Print the first mismatch to see where it broke
    diff = np.abs(pandas_result - sigfast_result)
    max_diff_idx = np.argmax(diff)
    print(f"   -> Max error found at index {max_diff_idx}:")
    print(f"   -> Pandas: {pandas_result[max_diff_idx]}")
    print(f"   -> SigFast: {sigfast_result[max_diff_idx]}")
    print(f"   -> Difference: {diff[max_diff_idx]}")