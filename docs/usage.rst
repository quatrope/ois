Usage
=====

If your image has a relatively narrow field of view where your PSF doesn't change significatively across the field,
you can use ``optimal_system`` on the default settings::

    >>> from ois import optimal_system
    >>> diff_image, optimal_image, kernel, background = optimal_system(test_image, refimage)

Where (See :ref:`summary`):

  * ``test_image`` is the image we want to analize, :math:`I`
  * ``refimage`` is an archive or reference image from the same location in the sky, :math:`R`
  * ``diff_image`` is :math:`D = I - (R \otimes K + B_{kg})`
  * ``optimal_image`` is :math:`R \otimes K + B_{kg}`
  * ``kernel`` is :math:`K`
  * ``background`` is :math:`B_{kg}`

.. note::

    ``test_image`` must be previously aligned with ``refimage``

.. note::
    
    The subtraction works best when ``refimage`` is of better quality than ``test_image``.

The default method for kernel fit is `Bramich (2008) <https://ui.adsabs.harvard.edu/abs/2008MNRAS.386L..77B/abstract>`_, which uses the information of all pixels in the image and fits for every pixel in the convolution kernel independently.
The available methods are ``"Alard-Lupton"``, ``"Bramich"`` and ``"AdaptiveBramich"`` (see :ref:`methods`).

Refer to the :ref:`api` for a complete description of the method.

Working with bad pixels (masks)
-------------------------------

If your reference image or test image have sections of bad pixels, it can distort the kernel estimation.
This is especially true when the fitting algorithm uses all the pixels in the image.
Saturated stars can also confuse the fit of the convolution kernel.

To let *ois* know which pixels are good, you can create a numpy masked array, with ``True`` on bad pixels.
The ois subtraction methods will ignore completely the information on those bad pixels.

The returned image, will have a combined ``OR`` mask from the mask in ``test_image`` and the mask on ``refimage`` expanded to exclude pixels that would have used defective pixels in the convolution.

If no mask is provided in both ``test_image`` and ``refimage``, the returned image will be a plain numpy array (no mask).
