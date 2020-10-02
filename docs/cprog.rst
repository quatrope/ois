.. _cprog:

Command-line program
====================

The C command-line program *ois* is somewhat limited compared to its Python counterpart.
It can only perform :ref:`adapt` method without simultanous background fit.
For that reason we suggest removing the background on the images to be used.

OIS on the command line will read the reference and science images from FITS files
on the file system. It will only output the difference image.
It will not output the kernel or optimal image.

.. note::

    To perform Bramich method instead of Adaptive Bramich, set the command line argument -kd to 0

Installation
------------

To compile and execute the C command-line program::

    $ git clone https://github.com/toros-astro/ois.git
    $ cd ois
    $ make ois
    $ ./ois --help

Usage
-----

.. code:: bash

    $ ois -ks, --kernel-side <int> -kd, --kernel-poly-deg <int> -ref <filename> -sci <filename> [-o <filename>] [-h, --help] [--version]

Command-line arguments:
^^^^^^^^^^^^^^^^^^^^^^^

    .. option:: -ks, --kernel-side

        The side in pixels of the kernel to calculate the optimal difference. Must be an odd number.

    .. option:: -kd, --kernel-poly-deg

        Degree of the interpolating polynomial for the variable kernel.

    .. option:: -ref

        The reference image path.

    .. option:: -sci

        The science image path.

    .. option:: -o

        [Optional] The path where the subtraction FITS file will be written.
        Default value is "diff_img.fits".

    .. option:: -h, --help

        Print usage help and exit.

    .. option:: --version

        Print version information and exit.
