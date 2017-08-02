from setuptools import setup, Extension
import numpy

# Get the version from astroalign file itself (not imported)
with open('ois.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            _, _, ois_version = line.replace("'", '').split()
            break

varconv = Extension('varconv', sources=['src/varconv.c'],
                    extra_compile_args=["-std=c99"])

setup(name='ois',
      version=ois_version,
      description='Optimal Image Subtraction',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/ois',
      py_modules=['ois', ],
      ext_modules=[varconv],
      include_dirs=[numpy.get_include()],
      install_requires=["numpy>=1.6",
                        "scipy>=0.16"],
      test_suite='tests',
      )
