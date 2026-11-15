# numox

**English** | [简体中文](./README_zh.md)

[![CI](https://github.com/oxoxox-oxox/numox/actions/workflows/ci.yml/badge.svg)](https://github.com/oxoxox-oxox/numox/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/numox.svg)](https://pypi.org/project/numox/)
[![Python versions](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://pypi.org/project/numox/)

A high-performance `float64` matrix C-extension for Python, accelerated with OpenMP, AVX2, and FMA instructions (with scalar fallback on non-x86 architectures).

[Features](#features) · [Installation](#installation) · [Quick Start](#quick-start) · [API Overview](#api-overview) · [Benchmarks](#benchmarks) · [Development & Testing](#development--testing) · [中文文档](./README_zh.md)

---

## Features

- **SIMD & Multi-threading**: Compute kernels written in C99 with OpenMP parallelization and AVX2/FMA intrinsics on x86_64.
- **Zero-Copy NumPy Interop**: Implements the Python Buffer Protocol, allowing `np.asarray(mat)` and `memoryview(mat)` to share underlying `float64` memory without copying (including strided slice views).
- **Zero-Copy Slice Views**: Supports 1D/2D slicing (`mat[r0:r1, c0:c1]`), nested slice views, and in-place slice assignment.
- **Built-in Type Stubs**: Ships with [__init__.pyi](./src/numox/__init__.pyi) and `py.typed` (PEP 561) for IDE autocompletion and static type checking.

---

## Installation

### From PyPI (Recommended)

Prebuilt wheels are published for 64-bit **Linux** and **Windows** (`Python >= 3.8`):

```bash
pip install numox
```

### From Source

Requires a C99 compiler with OpenMP support:
- **Linux**: GCC or Clang with OpenMP (`-fopenmp`)
- **Windows**: MSVC Build Tools (`/openmp`)
- **macOS**: Apple Clang + Homebrew `libomp` (`-Xpreprocessor -fopenmp -lomp`)

```bash
pip install git+https://github.com/oxoxox-oxox/numox.git
```

### Docker Development Environment

An Ubuntu 22.04 `Dockerfile` is included for compiling and running both Python and CUnit tests in an isolated container:

```bash
# 1. Build the image (note: Dockerfile uses Tsinghua APT/PyPI mirrors by default)
docker build -t numox-env .

# 2. Start the container with the repository mounted
# Linux / macOS / PowerShell:
docker run -it --rm -v "${PWD}:/workspace" numox-env bash
# Windows CMD:
# docker run -it --rm -v "%cd%:/workspace" numox-env bash

# 3. Inside the container: build/install and run tests
pip install -e . --no-build-isolation
make test
pytest -v tests/test_numox.py
```

---

## Quick Start

```python
import numpy as np
import numox as nc

# Create reproducible 3x3 random matrices in [0.0, 1.0)
mat1 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0, seed=42)
mat2 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0, seed=99)

# Matrix addition and matrix multiplication
sum_mat = mat1 + mat2
prod_mat = mat1 @ mat2  # Note: mat1 * mat2 is also matrix multiplication
print(prod_mat.to_list())

# Zero-copy slice view and in-place update
sub = mat1[0:2, 1:3]
sub[0, 0] = 10.0

# Zero-copy sharing with NumPy via Buffer Protocol
arr = np.asarray(mat1)
print(arr.shape, arr.dtype)  # (3, 3) float64
```

---

## API Overview

| Category | Usage | Notes |
|---|---|---|
| **Constructors** | `Matrix(rows, cols, val=0.0)` | Zero or constant fill |
| | `Matrix(rows, cols, rand=True, low=0.0, high=1.0, seed=0)` | Uniform random values in `[low, high)` |
| | `Matrix(rows, cols, flat_list)` | `1xN` / `Nx1` automatically gets 1D `shape == (N,)` |
| | `Matrix([[...], [...]])` | From 2D nested Python lists |
| **Operators** | `A + B`, `A - B`, `-A`, `abs(A)` | Element-wise operations |
| | `A @ B`, `A * B` | Matrix multiplication |
| | `A ** k` | Non-negative integer matrix power (`k >= 0`) |
| **Indexing & Views** | `m[i]`, `m[i, j]`, `m[r0:r1, c0:c1]` | Unit-stride slice views (`step=1`), supports nested views & assignment |
| **Attributes & Utils** | `m.shape`, `m.get(r, c)`, `m.set(r, c, val)`, `m.to_list()` | `shape` is read-only; also exposes module-level `numox.to_list(m)` |
| **Buffer Protocol** | `np.asarray(m)`, `memoryview(m)` | Zero-copy `float64` buffer export |

> **Important (`*` vs `@`):** Unlike NumPy where `*` performs element-wise multiplication, in `numox` **both `A * B` and `A @ B` perform matrix multiplication**.

---

## Benchmarks

> [查看中文版性能说明与结果](./README_zh.md#性能基准测试)

Compared with NumPy, `numox` has clear strengths and trade-offs across matrix sizes:
- **Where `numox` leads**: Fast random generation (`rand`, ~`1.7x–2.5x`), zero-copy slice arithmetic (`slice_add`, up to `3.23x` at `128x128` and `1.30x` at `1024x1024`), low-overhead small-matrix ops (`128x128`: `1.15x–1.53x`), and competitive or faster `1024x1024` element-wise ops and `matmul` (`347.58 GFLOPS`).
- **Where NumPy leads**: Medium sizes around `512x512` for simple element-wise kernels (`~0.49x–0.60x` due to OpenMP thread-dispatch/cache characteristics) and repeated matrix power `pow10` (`0.62x–0.78x`).

### Summary (`Python 3.13.15` | `NumPy 2.5.3` | `numox 1.0.2` | `Windows AMD64` | `32 cores`)

| Op | 128x128 (NumPy / numox) | 512x512 (NumPy / numox) | 1024x1024 (NumPy / numox) |
|---|---:|---:|---:|
| `fill` | 0.0019 / 0.0020 ms (**0.97x**) | 0.3398 / 0.6833 ms (**0.50x**) | 1.3990 / 1.3835 ms (**1.01x**) |
| `rand` | 0.0627 / 0.0251 ms (**2.50x**) | 0.9299 / 0.5423 ms (**1.71x**) | 4.2733 / 1.9486 ms (**2.19x**) |
| `add` | 0.0057 / 0.0037 ms (**1.53x**) | 0.3384 / 0.6936 ms (**0.49x**) | 1.5250 / 1.4339 ms (**1.06x**) |
| `sub` | 0.0041 / 0.0029 ms (**1.41x**) | 0.3440 / 0.6760 ms (**0.51x**) | 1.5518 / 1.4106 ms (**1.10x**) |
| `neg` | 0.0029 / 0.0020 ms (**1.49x**) | 0.3377 / 0.6884 ms (**0.49x**) | 1.3764 / 1.3688 ms (**1.01x**) |
| `abs` | 0.0023 / 0.0020 ms (**1.15x**) | 0.3461 / 0.6625 ms (**0.52x**) | 1.4087 / 1.4292 ms (**0.99x**) |
| `matmul` | 0.0669 / 0.0446 ms (**1.50x**) | 0.9995 / 1.0658 ms (**0.94x**) | 6.2893 / 6.1784 ms (**1.02x**) |
| `pow10` | 0.2181 / 0.2789 ms (**0.78x**) | 4.1690 / 6.7224 ms (**0.62x**) | 23.9454 / 31.3925 ms (**0.76x**) |
| `slice_add` | 0.0106 / 0.0033 ms (**3.23x**) | 0.4012 / 0.6647 ms (**0.60x**) | 1.8339 / 1.4138 ms (**1.30x**) |

Run [benchmark.py](./benchmark.py) to measure on your own machine:

```bash
python benchmark.py
# Useful options:
# python benchmark.py --installed                  # Force importing pip-installed numox
# python benchmark.py --sizes 128 512 1024 2048    # Custom matrix sizes
# python benchmark.py --ops add matmul --json out.json
```

<details>
<summary>Full benchmark.py output (includes throughput in GB/s &amp; GFLOPS)</summary>

```text
==================================================================================================
numox vs NumPy Performance Benchmark
Python 3.13.15 | NumPy 2.5.3 | numox 1.0.2 | OS Windows AMD64 | CPU cores: 32 | OMP_NUM_THREADS=auto
==================================================================================================

--- 128 x 128 (float64) ---
     Size  Op          Iters  NumPy med(ms)  numox med(ms)   Speedup      NumPy rate      numox rate   OK
--------------------------------------------------------------------------------------------------
  128x128  fill         2000         0.0019         0.0020     0.97x      68.02 GB/s      66.09 GB/s  yes
  128x128  rand           59         0.0627         0.0251     2.50x       2.09 GB/s       5.23 GB/s    -
  128x128  add          2000         0.0057         0.0037     1.53x      68.69 GB/s     105.35 GB/s  yes
  128x128  sub          2000         0.0041         0.0029     1.41x      96.80 GB/s     136.38 GB/s  yes
  128x128  neg          2000         0.0029         0.0020     1.49x      89.29 GB/s     132.77 GB/s  yes
  128x128  abs          2000         0.0023         0.0020     1.15x     112.76 GB/s     129.83 GB/s  yes
  128x128  matmul        600         0.0669         0.0446     1.50x    62.71 GFLOPS    94.14 GFLOPS  yes
  128x128  pow10         463         0.2181         0.2789     0.78x               -               -  yes
  128x128  slice_add    2000         0.0106         0.0033     3.23x      35.91 GB/s     115.92 GB/s  yes

--- 512 x 512 (float64) ---
     Size  Op          Iters  NumPy med(ms)  numox med(ms)   Speedup      NumPy rate      numox rate   OK
--------------------------------------------------------------------------------------------------
  512x512  fill          259         0.3398         0.6833     0.50x       6.17 GB/s       3.07 GB/s  yes
  512x512  rand          108         0.9299         0.5423     1.71x       2.26 GB/s       3.87 GB/s    -
  512x512  add           239         0.3384         0.6936     0.49x      18.59 GB/s       9.07 GB/s  yes
  512x512  sub           354         0.3440         0.6760     0.51x      18.29 GB/s       9.31 GB/s  yes
  512x512  neg           255         0.3377         0.6884     0.49x      12.42 GB/s       6.09 GB/s  yes
  512x512  abs           274         0.3461         0.6625     0.52x      12.12 GB/s       6.33 GB/s  yes
  512x512  matmul         97         0.9995         1.0658     0.94x   268.57 GFLOPS   251.86 GFLOPS  yes
  512x512  pow10          25         4.1690         6.7224     0.62x               -               -  yes
  512x512  slice_add     241         0.4012         0.6647     0.60x      15.56 GB/s       9.39 GB/s  yes

--- 1024 x 1024 (float64) ---
     Size  Op          Iters  NumPy med(ms)  numox med(ms)   Speedup      NumPy rate      numox rate   OK
--------------------------------------------------------------------------------------------------
1024x1024  fill          116         1.3990         1.3835     1.01x       6.00 GB/s       6.06 GB/s  yes
1024x1024  rand           24         4.2733         1.9486     2.19x       1.96 GB/s       4.30 GB/s    -
1024x1024  add            82         1.5250         1.4339     1.06x      16.50 GB/s      17.55 GB/s  yes
1024x1024  sub            86         1.5518         1.4106     1.10x      16.22 GB/s      17.84 GB/s  yes
1024x1024  neg           100         1.3764         1.3688     1.01x      12.19 GB/s      12.26 GB/s  yes
1024x1024  abs            90         1.4087         1.4292     0.99x      11.91 GB/s      11.74 GB/s  yes
1024x1024  matmul         19         6.2893         6.1784     1.02x   341.45 GFLOPS   347.58 GFLOPS  yes
1024x1024  pow10           4        23.9454        31.3925     0.76x               -               -  yes
1024x1024  slice_add      54         1.8339         1.4138     1.30x      13.67 GB/s      17.73 GB/s  yes
```

</details>

---

## Development & Testing

```text
src/
├── c/         # matrix.c, matrix.h, numox.c, numox.h (C99 + CPython C-API -> numox._core)
└── numox/     # __init__.py, __init__.pyi, py.typed (public Python package)
tests/         # CUnit tests (mat_test.c) and Python test suite (test_numox.py)
```

```bash
# Install build & test dependencies
pip install setuptools wheel -r requirements.txt

# Build and install package from source
pip install --no-build-isolation --no-deps .
# Or build C extension in-place inside src/numox/:
# python setup.py build_ext --inplace

# Run Python test suite (tests/test_numox.py)
pytest -v tests/test_numox.py
# Or using standard library unittest:
python -m unittest discover -s tests -v
```

On Linux with `libcunit1-dev` installed, you can also run the C-level unit tests via [Makefile](./Makefile):

```bash
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
make test
```

---

[Back to top](#numox) | [切换到中文文档 (README_zh.md)](./README_zh.md)
