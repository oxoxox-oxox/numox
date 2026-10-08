from setuptools import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # Use the setup function we imported and set up the modules.
    # You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
    numox_module = Extension('numox',
                            sources=['numox.c', 'matrix.c'],
                            extra_compile_args=CFLAGS,
                            extra_link_args = LDFLAGS)

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name='numox',
        version='1.0.0',
        author="oxoxox-oxox",
        description='A high-performance C Extension for matrix math',
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=[numox_module],
        python_requires=">=3.6",
    )

if __name__ == "__main__":
    main()
