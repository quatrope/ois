# Optimal Image Subtraction (OIS)

[![Build Status](https://travis-ci.org/toros-astro/ois.svg?branch=master)](https://travis-ci.org/toros-astro/ois.svg?branch=master)
[![codecov.io](https://codecov.io/github/toros-astro/ois/coverage.svg?branch=master)](https://codecov.io/github/toros-astro/ois?branch=master)
[![Documentation Status](https://readthedocs.org/projects/optimal-image-subtraction/badge/?version=latest)](http://optimal-image-subtraction.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/49916188.svg)](https://zenodo.org/badge/latestdoi/49916188)
[![Updates](https://pyup.io/repos/github/toros-astro/ois/shield.svg)](https://pyup.io/repos/github/toros-astro/ois/)
[![Python 3](https://pyup.io/repos/github/toros-astro/ois/python-3-shield.svg)](https://pyup.io/repos/github/toros-astro/ois/)

***OIS*** is a package to perform optimal image subtraction on astronomical images.

It offers different methods to subtract images:

 * Modulated multi-Gaussian kernel (as described in [Alard&Lupton (1998)][1])
 * Delta basis kernel (as described in [Bramich (2010)][2])
 * Adaptive Delta Basis kernel (as described in [Miller (2008)][3])

Each method can (optionally) simultaneously fit and remove common background.

You can find a Jupyter notebook example with the main features at [http://toros-astro.github.io/ois/](http://toros-astro.github.io/ois/)

***

## Minimal usage example

    >>> import ois
    >>> diff = ois.optimal_system(image, image_ref)[0]

More information available on docstrings

***
## Other Parameters:

**kernelshape**: shape of the kernel to use. Must be of odd size.

**bkgdegree**: degree of the polynomial to fit the background.
    To turn off background fitting set this to None.

**method**: One of the following strings

  * `Bramich`: A Delta basis for the kernel (all pixels fit
      independently). Default method.

 * `AdaptiveBramich`: Same as Bramich, but with a polynomial variation across the image. It needs the parameter **poly_degree**, which is the polynomial degree of the variation.

  * `Alard-Lupton`: A modulated multi-Gaussian kernel.
      It needs the **gausslist** keyword. **gausslist** is a list of dictionaries containing data of the gaussians used in the decomposition of the kernel. Dictionary keywords are: center, sx, sy, modPolyDeg

Extra parameters are passed to the individual methods.

**poly_degree**: needed only for `AdaptiveBramich`. It is the degree
    of the polynomial for the kernel spatial variation.

**gausslist**: needed only for `Alard-Lupton`. A list of dictionaries with info for the modulated multi-Gaussian. Dictionary keys are:

* **center**: a (row, column) tuple for the center of the Gaussian. Default: kernel center.
* **modPolyDeg**: the degree of the modulating polynomial. Default: 2
* **sx**: sigma in x direction. Default: 2.
* **sy**: sigma in y direction. Deafult: 2.

***

**Author**: Martin Beroiz

<martinberoiz@gmail.com>

[1]: http://arxiv.org/abs/astro-ph/9712287 "A method for optimal image subtraction"
[2]: http://arxiv.org/abs/0802.1273 "A New Algorithm For Difference Image Analysis"
[3]: http://adswww.harvard.edu
