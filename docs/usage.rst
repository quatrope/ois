Usage
=====

If your image has a relatively narrow field of view, and your PSF doesn't change significatively across the field,
you can use the method optimal_system::

    >>> diff_image, optimal_image, kernel, background = ois.optimal_system(test_image, refimage)

Here test_image is the image we want to analize and refimage is an archive or reference image from the same location in the sky, previously aligned with test_image.
The subtraction works best when refimage is of better quality than test_image.

This will return the difference image and the optimal_image, which is the convolution of refimage with a proper convolution kernel and
background fit so that optimal_image be as close to test_image as possible.
The kernel and the background that minimized the difference between test_image and optimal_image are also returned for reference.

The default method for kernel fit is Bramich (2008), which uses the information of all pixels in the image and fits for every pixel in the convolution kernel independently.
For more information on other fitting methods, see section :ref:`methods`.


optimal_system Parameters
-------------------------

In addition to the image to be subtracted and the reference image, you can specify several other parameters.

* gridshape
    When the PSF can't be approximated as constant across the CCD, you may want to divide the image into a grid and perform image subtraction on each grid element separately.
    This takes care of border issues among the grid sections, using some padding at the section boundaries for the convolution, wherever possible.
    The shape of the grid as a rows, colums tuple. Default is None (no grid).

 * image
    A numpy array with 2 dimensions. It will be the image to be subtracted from.

 * refimage
    A numpy array with 2 dimensions. It will be the reference image to make it match *image*.

 * kernelshape
    The shape of the kernel as a row, colums tuple. Default is (11, 11).

 * bkgdegree
    The degree of the polynomial fit to the background. Default is None (no background fit).
    Note that bkgdegree=0 corresponds to a constant-background fit.

 * method
    A string containing one of the following: "Alard-Lupton", "Bramich" or "AdaptiveBramich".
    Default is "Bramich"

 * poly_degree
    *This has to be supplied only for the Adaptive Bramich method*.

    It is the degree of the polynomial variation of the kernel across the image.

 * gausslist
    *This has to be supplied only for the Alard-Lupton method*.

    A list of dictionaries containing info about the Gaussians and modulating polynomial used in the multi-Gaussian fit.
    For each gaussian we want to use, we need the following keys (Not all the keys need to be present):

    **center:**
        the pixel x, y coordinates of the center relative to the kernel origin (column=0, row=0).
        For a 11 rows by 13 columns kernel, the center of the kernel would be at x, y = (6, 5)
        If not provided, ois will assume the kernel center calculated from its shape.

    **sx:**
        sigma in the x direction

    **sy:**
        sigma in the y direction

    **modPolyDeg:**
        The degree of the modulating polynomial.
        If not provided, it will default to 2.

Below is an example of a gausslist::

    gausslist = [{center: (5, 5),
                  sx: 2.
                  sy: 2.
                  modPolyDeg: 3},
                 {sx: 1.0
                  sy: 2.5
                  modPolyDeg: 1},
                 {sx: 3.0
                  sy: 1.0},
                ]


Working with bad pixels (masks)
-------------------------------

If your reference image or test image have bad sections of pixels, it can distort the kernel estimation.
This is especially true when the fitting algorithm uses all the pixels in the image.
Saturated stars can also confuse the fit of the convolution kernel.

To let *ois* know which pixels are good, you can create a numpy masked array, with True on bad pixels.
The ois subtraction methods will ignore completely the information on those bad pixels.

The returned image, will have a combined OR mask from the mask in test_image and the mask on refimage expanded to exclude pixels that would have used defective pixels in the convolution.

If no mask is provided in both test_image and refimage, the returned image will be a numpy array (no mask).
