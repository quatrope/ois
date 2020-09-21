from setuptools import setup, Extension
import numpy

# Get the version from astroalign file itself (not imported)
with open("ois.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, ois_version = line.replace('"', "").split()
            break

with open("README.md", "r") as f:
    long_description = f.read()

varconv = Extension(
    "varconv",
    sources=["src/varconv.c", "src/oistools.c"],
    include_dirs=["src", numpy.get_include()],
    libraries=["m"],
    extra_compile_args=["-std=c99"],
)

setup(
    name="ois",
    version=ois_version,
    description="Optimal Image Subtraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Martin Beroiz",
    author_email="martinberoiz@gmail.com",
    url="https://github.com/toros-astro/ois",
    py_modules=[
        "ois",
    ],
    ext_modules=[varconv],
    install_requires=["numpy>=1.6", "scipy>=0.16"],
    test_suite="tests",
)
