Optimal Image Subtraction (OIS)
===============================

**OIS** is a Python package and a C command-line program to perform optimal image subtraction on astronomical images.

It offers different methods to subtract images:

  * Modulated multi-Gaussian kernel (as described in [alard1998]_)
  * Delta basis kernel (as described in [bramich2008]_)
  * Adaptive Delta Basis kernel (as described in [miller2008]_)

Main features:

  * Each method can (optionally) simultaneously fit and remove common background.
  * Each method can resolve small translations on the image
  * Adaptive Bramich can resolve small relative rotations on the images

(See :ref:`methods`)

Installation
------------

Install it directly from PyPI using pip::

    pip install ois


Contents:
^^^^^^^^^

.. toctree::
   :maxdepth: 2

   usage
   methods
   cprog
   api
