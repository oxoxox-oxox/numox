from setuptools import setup, Extension
import platform
import sys

def main():
    machine = platform.machine().lower()
    is_x86 = machine in ('x86_64', 'amd64', 'x86', 'i386', 'i686')

    # 根据操作系统和 CPU 架构动态分配编译参数
    if sys.platform == 'win32':
        CFLAGS = ['/openmp', '/O2', '/fp:fast', '/GL']
        if is_x86:
            CFLAGS.append('/arch:AVX2')
        LDFLAGS = ['/LTCG']
    elif sys.platform == 'darwin':
        # Mac OS
        CFLAGS = ['-O3', '-Wall', '-std=c99', '-Xpreprocessor', '-fopenmp', '-ffast-math', '-flto']
        if is_x86:
            CFLAGS.extend(['-mavx2', '-mfma'])
        LDFLAGS = ['-lomp', '-flto']
    else:
        # Linux
        CFLAGS = ['-Wall', '-std=c99', '-fopenmp', '-ffast-math', '-flto', '-pthread', '-O3']
        if is_x86:
            CFLAGS.extend(['-mavx2', '-mfma'])
        LDFLAGS = ['-fopenmp', '-flto']

    numox_module = Extension('numox',
                            sources=['numox.c', 'matrix.c'],
                            extra_compile_args=CFLAGS,
                            extra_link_args = LDFLAGS)

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name='numox',
        version='1.0.4',
        author="oxoxox-oxox",
        description='A high-performance C Extension for matrix math',
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=[numox_module],
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()
