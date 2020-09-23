.. _theory:

Theory
======

General Assumptions
-------------------

All of the methods assume we have a reference image :math:`R` and a
science image :math:`I` that can be approximately modelled as:

.. math::
    I \approx R \otimes K + B_{kg}

for some background :math:`B_{kg}` and some kernel :math:`K`.

The optimal image subtraction :math:`D` is then:

.. math::
    D = I - (R \otimes K + B_{kg})

The methods differ in their modelling of :math:`K`.

.. warning::

    In the ideal case of perfect subtraction, :math:`D` should contain only noise and optical transients.
    In practice, tiny image misalignments, saturated stars and poor PSF fitting can leave subtraction artifacts near sources.

.. [alard1998] `"A Method for Optimal Image Subtraction" - C. Alard, R. H. Lupton, 1997. <https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A/abstract>`_
.. [bramich2008] `"A New Algorithm For Difference Image Analysis" - D.M. Bramich, 2008. <https://ui.adsabs.harvard.edu/abs/2008MNRAS.386L..77B/abstract>`_
.. [miller2008] `"Optimal Image Subtraction Method: Summary Derivations, Applications, and Publicly Shared Application Using IDL" - J. PATRICK MILLER et al., 2008. <https://ui.adsabs.harvard.edu/abs/2008PASP..120..449M/abstract>`_

.. _methods:

Methods
-------

.. _alard:

Alard & Lupton
^^^^^^^^^^^^^^

Alard and Lupton [alard1998]_ introduced a method to simultaneously fit a background :math:`B_{kg}` and a convolution kernel :math:`K` that will minimize the difference between a reference image :math:`R` and an another science image :math:`I`.

.. math::
    I \approx R \otimes K + B_{kg}

This method assumes that the convolution kernel can be approximated by a linear combination of fixed Gaussians, where the linear coefficients are left free to vary.
In addition, each Gaussian is modulated with a polynomial of an arbitrary degree.

.. math::
    K &= \sum_i a_i B_i \\
      &= \sum_n a_n \times \left[ \exp \left(- \frac{u^2 + v^2}{2 \sigma_n^2} \right) \sum_{d_n^x} \sum_{d_n^y} u^{d_n^x} v^{d_n^y} \right]

.. note::
    The centre, width and orientation of the Gaussians are fixed beforehand as well as the number of Gaussians to use in the expansion.

In OIS this is specified with a list of dictionaries, one for each gaussian we want to use.
Below is an example of a basis with 3 Gaussians:

.. code:: python

    gausslist=[{center: (5, 5), sx: 2., sy: 2., modPolyDeg: 3},
               {sx: 1.0, sy: 2.5, modPolyDeg: 1},
               {sx: 3.0, sy: 1.0},]

.. _bramich:

Bramich
^^^^^^^

The method developed in [bramich2008]_ modifies Alard-Lupton making each pixel of the kernel an independent parameter to fit.
This is equivalent as having a vector basis consisting of `Delta kernels`.
It can also simultaneously fit a polynomial background.

.. math::
    K = \sum_n a_n \times \delta_{nn'}


This method does not make assumptions on the kernel shape and can thus model completely arbitrary kernels.
It can also correct for small translations between the images.

While more accurate, this method is computationally more expensive than :ref:`alard`'s.

.. warning::

  Since each pixel is treated independently, a 11 by 11 kernel will have 121 free parameters just for the kernel.
  It grows quadratically with the kernel side. This needs to be taken in consideration for large kernels.

.. _adapt:

Adaptive Bramich
^^^^^^^^^^^^^^^^

Like Bramich, this method also treats each pixel independently,
but it will also multiply each pixel by a polynomial on the coordinates of the image.

This requires a special type of convolution where the kernel varies point to point in the image.

It is especially suited for situations where the PSF varies significatively across the image.

The method is described in more detail in [miller2008]_.

.. warning::

  Just like Bramich, the number of free parameters scales quadratically with the kernel side.
  Furthermore, the degree of the polynomial multiplies the number of parameters by (deg + 1) * (deg + 2) / 2.
  This needs to be taken in consideration for large kernels.
