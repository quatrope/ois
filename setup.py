from setuptools import setup

setup(name='ois',
      version='0.1a0',
      description='Optimal Image Subtraction',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/ois',
      packages=['ois', ],
      install_requires=["numpy>=1.6.2",
                        ],
      test_suite='tests',
      )
