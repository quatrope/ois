# Optimal Image Subtraction (OIS)

[![Build Status](https://travis-ci.org/toros-astro/ois.svg?branch=master)](https://travis-ci.org/toros-astro/ois.svg?branch=master)
[![codecov.io](https://codecov.io/github/toros-astro/ois/coverage.svg?branch=master)](https://codecov.io/github/toros-astro/ois?branch=master)

***OIS*** is a package to perform optimal image subtraction on astronomical images.

It uses the subtraction algorithm described in [Alard&Lupton (1998)][1] as well as [Bramich (2010)][2] improvement.

***

Usage example

    >>> from ois import subtract
    >>> subtraction = subtract.optimal_subtract(image, image_ref)

More information available on docstrings

***

Author: Martin Beroiz

<martinberoiz@gmail.com>

[1]: http://arxiv.org/abs/astro-ph/9712287 "A method for optimal image subtraction"
[2]: http://arxiv.org/abs/0802.1273 "A New Algorithm For Difference Image Analysis"
