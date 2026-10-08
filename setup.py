from setuptools import setup, Extension
import sys

def main():
    # 根据操作系统动态分配编译参数
    if sys.platform == 'win32':
        CFLAGS = ['/openmp', '/O2', '/arch:AVX']
        LDFLAGS = []
    elif sys.platform == 'darwin':
        # Mac OS
        CFLAGS = ['-g', '-Wall', '-std=c99', '-Xpreprocessor', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
        LDFLAGS = ['-lomp']
    else:
        # Linux
        CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
        LDFLAGS = ['-fopenmp']

    numox_module = Extension('numox',
                            sources=['numox.c', 'matrix.c'],
                            extra_compile_args=CFLAGS,
                            extra_link_args = LDFLAGS)

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name='numox',
        version='1.0.2',
        author="oxoxox-oxox",
        description='A high-performance C Extension for matrix math',
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=[numox_module],
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()
