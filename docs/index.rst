Optimal Image Subtraction (OIS)
===============================

**OIS** is a package to perform optimal image subtraction on astronomical images.

It offers different methods to subtract images:

 * Modulated multi-Gaussian kernel (as described in [alard1998]_)
 * Delta basis kernel (as described in [bramich2008]_)
 * Adaptive Delta Basis kernel (as described in [miller2008]_)

Each method can (optionally) simultaneously fit and remove common background.

.. _summary:

Theoretical Summary
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


Contents:
^^^^^^^^^

.. toctree::
   :maxdepth: 2

   installation
   usage
   methods
   api
