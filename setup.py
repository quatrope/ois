from setuptools import setup, Extension
import numpy

varconv = Extension('varconv', sources=['src/varconv.c'],
                    extra_compile_args=["-std=c99"])

setup(name='ois',
      version='0.1.2',
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
