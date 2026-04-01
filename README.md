![Python Package CI](https://github.com/SamdaniSayam/triples-sigfast/actions/workflows/main.yml/badge.svg)

#  SigFast

![PyPI](https://img.shields.io/badge/PyPI-v0.3.1-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Performance triples-sigfast has been stress-tested on 100-Million-row NumPy arrays. It scales linearly and maintains efficient memory usage, making it suitable for enterprise-level financial data and scientific research.

A high-performance time-series processing library built for Data Scientists and Physicists. Uses **Numba JIT** and **C-level multithreading** to bypass the Python GIL.

### Why SigFast?
Pandas is great, but it runs on a single thread. When analyzing millions of data points (IoT sensors, high-frequency trading, astrophysics), Pandas becomes a bottleneck. SigFast distributes the math across all your CPU cores.

**Benchmark (10 Million Data Points - Rolling Window):**
*    Pandas `.rolling().mean()`: **~1.20 seconds**
*    SigFast Engine: **~0.03 seconds (40x Faster)**

### Installation
```bash
pip install triples-sigfast
