# numox

[English](./README.md) | **简体中文**

[![CI](https://github.com/oxoxox-oxox/numox/actions/workflows/ci.yml/badge.svg)](https://github.com/oxoxox-oxox/numox/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/numox.svg)](https://pypi.org/project/numox/)
[![Python versions](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://pypi.org/project/numox/)

面向 Python 的高性能 `float64` 矩阵 C 扩展库，底层采用 OpenMP 多线程并行以及 AVX2 + FMA 指令集加速（在非 x86 架构下自动回退至标量实现）。

[核心特性](#核心特性) · [安装](#安装) · [快速开始](#快速开始) · [API 概览](#api-概览) · [性能基准测试](#性能基准测试) · [开发与测试](#开发与测试) · [English Docs](./README.md)

---

## 核心特性

- **SIMD 与多线程加速**：核心计算采用 C99 编写，结合 OpenMP 并行调度与 x86_64 AVX2/FMA 向量指令优化。
- **零拷贝 NumPy 互操作**：完整实现 Python Buffer Protocol，可通过 `np.asarray(mat)` 或 `memoryview(mat)` 零拷贝共享底层 `float64` 内存（包括切片视图）。
- **零拷贝切片视图**：支持一维/二维切片（`mat[r0:r1, c0:c1]`）、嵌套切片视图以及原位切片赋值。
- **类型存根支持**：内置 [__init__.pyi](./src/numox/__init__.pyi) 与 `py.typed`（PEP 561），提供完整的 IDE 自动补全与静态类型检查支持。

---

## 安装

### 通过 PyPI 安装（推荐）

官方 CI 已为 64 位 **Linux** 和 **Windows**（`Python >= 3.8`）构建并发布预编译 Wheel 包，无需本地编译器：

```bash
pip install numox
```

### 从源码编译安装

需要支持 OpenMP 的 C99 编译器：
- **Linux**：GCC 或 Clang（`-fopenmp`）
- **Windows**：MSVC Build Tools（`/openmp`）
- **macOS**：Apple Clang + Homebrew `libomp`（`-Xpreprocessor -fopenmp -lomp`）

```bash
pip install git+https://github.com/oxoxox-oxox/numox.git
```

### Docker 开发环境

仓库提供基于 Ubuntu 22.04 的 `Dockerfile`，无需配置本机环境即可编译并运行 CUnit 与 Python 测试：

```bash
# 1. 构建镜像（注：Dockerfile 默认配置了清华 APT / PyPI 镜像源）
docker build -t numox-env .

# 2. 启动容器并挂载当前代码目录
# Linux / macOS / PowerShell：
docker run -it --rm -v "${PWD}:/workspace" numox-env bash
# Windows CMD：
# docker run -it --rm -v "%cd%:/workspace" numox-env bash

# 3. 在容器内编译安装并运行测试
pip install -e . --no-build-isolation
make test
pytest -v tests/test_numox.py
```

---

## 快速开始

```python
import numpy as np
import numox as nc

# 创建可复现的 3x3 随机矩阵（均匀分布于 [0.0, 1.0)）
mat1 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0, seed=42)
mat2 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0, seed=99)

# 矩阵加法与矩阵乘法
sum_mat = mat1 + mat2
prod_mat = mat1 @ mat2  # 注意：在 numox 中 mat1 * mat2 同样是矩阵乘法
print(prod_mat.to_list())

# 零拷贝切片视图与原位修改
sub = mat1[0:2, 1:3]
sub[0, 0] = 10.0

# 通过 Buffer Protocol 与 NumPy 零拷贝共享内存
arr = np.asarray(mat1)
print(arr.shape, arr.dtype)  # (3, 3) float64
```

---

## API 概览

| 类别 | 用法 | 说明 |
|---|---|---|
| **构造方法** | `Matrix(rows, cols, val=0.0)` | 全零或指定常量填充 |
| | `Matrix(rows, cols, rand=True, low=0.0, high=1.0, seed=0)` | `[low, high)` 区间均匀分布随机矩阵 |
| | `Matrix(rows, cols, flat_list)` | `1xN` / `Nx1` 会自动识别为一维 `shape == (N,)` |
| | `Matrix([[...], [...]])` | 从二维嵌套列表构造 |
| **运算符** | `A + B`、`A - B`、`-A`、`abs(A)` | 逐元素运算 |
| | `A @ B`、`A * B` | 矩阵乘法 |
| | `A ** k` | 非负整数矩阵幂（`k >= 0`） |
| **索引与视图** | `m[i]`、`m[i, j]`、`m[r0:r1, c0:c1]` | 步长为 1 的零拷贝切片视图，支持嵌套切片与原位赋值 |
| **属性与工具** | `m.shape`、`m.get(r, c)`、`m.set(r, c, val)`、`m.to_list()` | `shape` 为只读属性；亦提供模块函数 `numox.to_list(m)` |
| **缓冲区协议** | `np.asarray(m)`、`memoryview(m)` | 零拷贝导出 `float64` 缓冲区（支持跨步切片视图） |

> **重要提示（`*` 与 `@` 的语义区别）：** 与 NumPy 中 `*` 表示逐元素乘法不同，在 `numox` 中 **`A * B` 和 `A @ B` 均表示矩阵乘法**，从 NumPy 迁移时请特别留意。

---

## 性能基准测试

> [Read Benchmarks in English](./README.md#benchmarks)

相比 NumPy，`numox` 在不同规模与算子上各有优劣：
- **优势场景**：随机数生成（`rand`，约 `1.7x–2.5x`）、切片视图加法（`slice_add`，`128x128` 达 `3.23x`、`1024x1024` 达 `1.30x`）、低 Python 调用开销的小矩阵运算（`128x128` 多数算子 `1.15x–1.53x`），以及 `1024x1024` 下的逐元素运算与矩阵乘法（`matmul` 达 `347.58 GFLOPS`）。
- **劣势场景**：在 `512x512` 附近受 OpenMP 线程调度与缓存特征影响，简单逐元素算子慢于 NumPy（约 `0.49x–0.60x`）；高次矩阵幂 `pow10` 在各尺寸下均慢于 NumPy（`0.62x–0.78x`）。

### 结果汇总（`Python 3.13.15` | `NumPy 2.5.3` | `numox 1.0.2` | `Windows AMD64` | `32 cores`）

| 算子 | 128x128 (NumPy / numox) | 512x512 (NumPy / numox) | 1024x1024 (NumPy / numox) |
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

可在本地运行 [benchmark.py](./benchmark.py) 复现测试：

```bash
python benchmark.py
# 常用参数示例：
# python benchmark.py --installed                  # 强制测试 site-packages 中 pip 安装的版本
# python benchmark.py --sizes 128 512 1024 2048    # 自定义测试矩阵尺寸
# python benchmark.py --ops add matmul --json out.json
```

<details>
<summary>点击展开 benchmark.py 完整输出（含 GB/s 与 GFLOPS 吞吐率）</summary>

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

## 开发与测试

```text
src/
├── c/         # matrix.c、matrix.h、numox.c、numox.h（C99 + CPython C-API -> numox._core）
└── numox/     # __init__.py、__init__.pyi、py.typed（对外 Python 包）
tests/         # CUnit 测试（mat_test.c）与 Python 单元测试（test_numox.py）
```

```bash
# 安装构建与测试依赖
pip install setuptools wheel -r requirements.txt

# 从源码编译安装
pip install --no-build-isolation --no-deps .
# 或原地构建扩展模块至 src/numox/ 目录：
# python setup.py build_ext --inplace

# 运行 Python 单元测试（tests/test_numox.py）
pytest -v tests/test_numox.py
# 或使用标准库 unittest：
python -m unittest discover -s tests -v
```

在已安装 `libcunit1-dev` 的 Linux 环境下，可通过 [Makefile](./Makefile) 运行 C 层单元测试：

```bash
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
make test
```

---

[返回顶部](#numox) | [Switch to English Documentation (README.md)](./README.md)
