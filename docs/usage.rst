Usage
=====

If your image has a relatively narrow field of view, and your PSF doesn't change significatively across the field,
you can use the method optimalkernelandbkg::

    >>> optimal_image, kernel, background = optimalkernelandbkg(test_image, refimage)

Here test_image is the image we want to analize and refimage is an archive or reference image from the same location in the sky, previously aligned with test_image.

This will return the optimal_image, which is the convolution of refimage with a proper convolution kernel and
background fit so that optimal_image be as close to test_image as possible.
The default method for kernel fit is Bramich (2008), which uses the information of all pixels in the image and fits for every pixel in the convolution kernel independently.
For more information on other fitting methods, see section ??.

It's also provided the kernel and background that minimized the difference between test_image and optimal_image.

The next step is just subtracting optimal_image from test_image::

    >>> diff = test_image - optimal_image

Working with masks
------------------

If your reference image or test image have bad sections of pixels, it can distort the kernel estimation.
This is especially true when the fitting algorithm uses all the pixels in the image.

To let ois know which pixels are good, you can create a numpy masked array, with True on bad pixels.
The ois subtraction methods will ignore completely the information on those bad pixels.

The returned image, will have a combined OR mask from the mask in test_image and the mask on refimage expanded to exclude pixels that would have used defective pixels in the convolution.

If no mask is provided in both test_image and refimage, the returned image will be a numpy array (no mask).

Non-constant PSF
----------------

When the image has a large field of view or lens defects, the PSF won't be constant across the CCD.

In this case you may want to section your CCD into smaller regions on a grid:

    >>> diff = subtractongrid(test_image, refimage, gridshape=(3, 2))

This method will divide test_image into 3 rows and 2 columns and apply optimalkernelandbkg on each grid section.

It uses a little bit of padding at the section boundaries for the convolution, when possible.