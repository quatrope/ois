# Optimal Image Subtraction (OIS)

[![Build Status](https://travis-ci.org/toros-astro/ois.svg?branch=master)](https://travis-ci.org/toros-astro/ois.svg?branch=master)
[![codecov.io](https://codecov.io/github/toros-astro/ois/coverage.svg?branch=master)](https://codecov.io/github/toros-astro/ois?branch=master)
[![Documentation Status](https://readthedocs.org/projects/optimal-image-subtraction/badge/?version=latest)](http://optimal-image-subtraction.readthedocs.io/en/latest/?badge=latest)

***OIS*** is a package to perform optimal image subtraction on astronomical images.

It offers different methods to subtract images:

 * Modulated multi-Gaussian kernel (as described in [Alard&Lupton (1998)][1])
 * Delta basis kernel (as described in [Bramich (2010)][2])
 * Adaptive Delta Basis kernel (as described in [Miller (2008)][3])

***

Usage example

    >>> import ois
    >>> diff, _, _, _ =  ois.optimal_system(image, image_ref)

More information available on docstrings

***

Author: Martin Beroiz

<martinberoiz@gmail.com>

[1]: http://arxiv.org/abs/astro-ph/9712287 "A method for optimal image subtraction"
[2]: http://arxiv.org/abs/0802.1273 "A New Algorithm For Difference Image Analysis"
[3]: http://adswww.harvard.edu