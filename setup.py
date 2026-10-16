from setuptools import setup, Extension
import sys

def main():
    # 根据操作系统动态分配编译参数
    if sys.platform == 'win32':
        CFLAGS = ['/openmp:llvm', '/O2', '/arch:AVX2', '/fp:fast', '/GL']
        LDFLAGS = ['/LTCG']
    elif sys.platform == 'darwin':
        # Mac OS
        CFLAGS = ['-O3', '-Wall', '-std=c99', '-Xpreprocessor', '-fopenmp', '-mavx2', '-mfma', '-ffast-math', '-flto']
        LDFLAGS = ['-lomp', '-flto']
    else:
        # Linux
        CFLAGS = ['-Wall', '-std=c99', '-fopenmp', '-mavx2', '-mfma', '-ffast-math', '-flto', '-pthread', '-O3']
        LDFLAGS = ['-fopenmp', '-flto']

    numox_module = Extension('numox',
                            sources=['numox.c', 'matrix.c'],
                            extra_compile_args=CFLAGS,
                            extra_link_args = LDFLAGS)

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name='numox',
        version='1.0.3',
        author="oxoxox-oxox",
        description='A high-performance C Extension for matrix math',
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=[numox_module],
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()
