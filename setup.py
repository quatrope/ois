from setuptools import setup

setup(name='ois',
      version='0.1a1',
      description='Optimal Image Subtraction',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/ois',
      py_modules=['ois', ],
      install_requires=["numpy>=1.6",
                        "scipy>=0.16"],
      test_suite='tests',
      )
