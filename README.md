# numox

A high-performance Python C-extension for matrix mathematics, accelerated with OpenMP, AVX, and FMA instructions.

## Installation

### From Source (Requires C compiler)

You can install `numox` directly from this GitHub repository:

```bash
# Note: Ensure you have a C compiler (GCC/Clang) with OpenMP support installed.
pip install git https://github.com/YourUsername/numox.git
```

### Docker Development Environment

If you want a clean, "painless" environment to compile and test the C-extension without configuring your local machine, use the provided Ubuntu 22.04 Dockerfile:

```bash
# 1. Build the docker image
docker build -t numox-env .

# 2. Run the container, mounting your current code directory
docker run -it --rm -v "${PWD}:/workspace" numox-env bash

# 3. Inside the container, install the library or run tests
pip install .
make test
```

## Quick Start

```python
import numox as nc

# Create a 3x3 matrix initialized with random values between 0.0 and 1.0
mat1 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0)
mat2 = nc.Matrix(3, 3, rand=True, low=0.0, high=1.0)

# Perform fast matrix operations
result = mat1 + mat2
print(result.to_list())
```

## Test

```python
python setup.py build_ext --inplace
python test.py
```
