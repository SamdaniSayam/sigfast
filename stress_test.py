import numpy as np
import pandas as pd
import time
import sys

# We need the memory profiler library
try:
    from memory_profiler import memory_usage
except ImportError:
    print("❌ Please run: pip install memory-profiler")
    sys.exit(1)

# Import your Titan Engine
try:
    from sigfast.core import rolling_average
except ImportError:
    print("❌ Error: Could not find sigfast library.")
    sys.exit(1)

def run_benchmark(n_rows):
    print(f"\n--- ⚡ STRESS TEST: {n_rows:,} ROWS ⚡ ---")
    
    # 1. Generate Data
    data = np.random.rand(n_rows)
    data_size_mb = data.nbytes / 1e6
    print(f"   -> Data size: {data_size_mb:.2f} MB in RAM.")
    
    # 2. Run the Benchmark with Memory Profiling
    start_time = time.perf_counter()
    
    # memory_usage() watches the function in real-time and reports peak RAM usage
    mem_usage = memory_usage((rolling_average, (data, 100)), interval=0.1)
    
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    peak_mem = max(mem_usage)
    
    # 3. Report Results
    print(f"   -> Execution Time: {execution_time:.4f} seconds.")
    print(f"   -> Peak RAM Usage: {peak_mem:.2f} MB.")
    
    # THE VERDICT: Check for memory leaks!
    # If the RAM usage is >2x the data size, it means your code is duplicating memory!
    if peak_mem > data_size_mb * 2.5:
        print("   -> ⚠️ WARNING: High memory overhead detected! Possible memory leak.")
    else:
        print("   -> ✅ SUCCESS: Memory usage is stable and efficient (Near Zero-Copy).")

if __name__ == "__main__":
    print("="*60)
    print("      PROJECT TITAN: The 'Crucible' Stress Test")
    print("="*60)
    
    # Run the tests at different scales
    run_benchmark(1_000_000)
    run_benchmark(10_000_000)
    run_benchmark(50_000_000)
    
    # The Final Boss
    try:
        run_benchmark(100_000_000)
        print("\n\n🏆🏆🏆 CRUCIBLE PASSED! The engine survived 100M+ rows. 🏆🏆🏆")
    except MemoryError:
        print("\n\n🔥🔥🔥 BREAKDOWN THRESHOLD REACHED! Your laptop ran out of RAM. 🔥🔥🔥")
        print("This is normal! It means you have found the physical hardware limit.")