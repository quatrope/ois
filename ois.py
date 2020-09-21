"""
    OIS is a package to perform optimal image subtraction on astronomical images.
    It offers different methods to subtract images:

      * Modulated multi-Gaussian kernel
      * Delta basis kernel
      * Adaptive Delta Basis kernel

    Each method can (optionally) simultaneously fit and remove common background.

    Usage:

        >>> from ois import optimal_system
        >>> difference, optimalImage, optimalKernel, background =
                                        optimal_system(image, referenceImage)


    (c) Martin Beroiz
    email: <martinberoiz@gmail.com>
    University of Texas at San Antonio
"""


__version__ = "0.2"

import numpy as np
from scipy import signal
from scipy import ndimage

__all__ = [
    "EvenSideKernelError",
    "convolve2d_adaptive",
    "eval_adpative_kernel",
    "optimal_system",
]


class EvenSideKernelError(ValueError):
    pass


def _has_mask(image):
    is_masked_array = isinstance(image, np.ma.MaskedArray)
    if is_masked_array and isinstance(image.mask, np.ndarray):
        return True
    return False


class SubtractionStrategy(object):
    def __init__(self, image, refimage, kernelshape, bkgdegree):
        self.k_shape = kernelshape

        # Check here for dimensions
        if image.ndim != 2:
            raise ValueError("Wrong dimensions for image")
        if refimage.ndim != 2:
            raise ValueError("Wrong dimensions for refimage")
        if image.shape != refimage.shape:
            raise ValueError("Images have different shapes")
        self.h, self.w = image.shape
        self.image, self.refimage, self.badpixmask = self.separate_data_mask(
            image, refimage
        )

        self.coeffs = None
        self.bkgdegree = bkgdegree
        self.optimal_image = None
        self.background = None
        self.kernel = None
        self.difference = None

    def separate_data_mask(self, image, refimage):
        def ret_data(image):
            if isinstance(image, np.ma.MaskedArray):
                image_data = image.data
            else:
                image_data = image
            return image_data

        badpixmask = None
        if _has_mask(refimage):
            badpixmask = ndimage.binary_dilation(
                refimage.mask.astype("uint8"), structure=np.ones(self.k_shape)
            ).astype("bool")
            if _has_mask(image):
                badpixmask += image.mask
        elif _has_mask(image):
            badpixmask = image.mask
        return ret_data(image), ret_data(refimage), badpixmask

    def coeffstobackground(self, coeffs):
        "Given a list of coefficients, return an array with the polynomial background"
        bkgdeg = int(-1.5 + 0.5 * np.sqrt(9 + 8 * (len(coeffs) - 1)))
        h, w = self.h, self.w
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(bkgdeg + 1)]
        allys = [pow(y, i) for i in range(bkgdeg + 1)]
        mybkg = np.zeros(self.image.shape)
        ind = 0
        for i, anX in enumerate(allxs):
            for aY in allys[: bkgdeg + 1 - i]:
                mybkg += coeffs[ind] * anX * aY
                ind += 1
        return mybkg

    def get_cmatrices_background(self):
        h, w = self.refimage.shape
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(self.bkgdegree + 1)]
        allys = [pow(y, i) for i in range(self.bkgdegree + 1)]
        bkg_c = [
            anX * aY
            for i, anX in enumerate(allxs)
            for aY in allys[: self.bkgdegree + 1 - i]
        ]
        return bkg_c

    def get_coeffs(self):
        "Override this function to solve the matrix minimization system"
        return self.coeffs

    def get_optimal_image(self):
        if self.optimal_image is not None:
            return self.optimal_image
        opt_image = signal.convolve2d(
            self.refimage, self.get_kernel(), mode="same"
        )
        if self.bkgdegree is not None:
            opt_image += self.get_background()
        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image
        return self.optimal_image

    def get_background(self):
        if self.background is not None:
            return self.background
        if self.bkgdegree is not None:
            bkgdof = (self.bkgdegree + 1) * (self.bkgdegree + 2) // 2
            coeffs = self.get_coeffs()
            self.background = self.coeffstobackground(coeffs[-bkgdof:])
        else:
            self.background = np.zeros(self.image.shape)
        return self.background

    def get_kernel(self):
        "Override this method to return the kernel"
        return self.kernel

    def get_difference(self):
        if self.difference is not None:
            return self.difference
        opt_image = self.get_optimal_image()
        if self.badpixmask is not None:
            self.difference = np.ma.array(
                self.image - opt_image, mask=self.badpixmask
            )
        else:
            self.difference = self.image - opt_image
        return self.difference


class AlardLuptonStrategy(SubtractionStrategy):
    def __init__(self, image, refimage, kernelshape, bkgdegree, gausslist):
        super(AlardLuptonStrategy, self).__init__(
            image, refimage, kernelshape, bkgdegree
        )
        if gausslist is None:
            self.gausslist = [{}]
        else:
            self.gausslist = gausslist
        self.clean_gausslist()

    def gauss(self, center, sx, sy):
        h, w = self.k_shape
        x0, y0 = center
        x, y = np.meshgrid(list(range(w)), list(range(h)))
        k = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
        norm = k.sum()
        return k / norm

    def clean_gausslist(self):
        for agauss in self.gausslist:
            if "center" not in agauss:
                h, w = self.k_shape
                agauss["center"] = ((h - 1) // 2.0, (w - 1) // 2.0)
            if "modPolyDeg" not in agauss:
                agauss["modPolyDeg"] = 2
            if "sx" not in agauss:
                agauss["sx"] = 2.0
            if "sy" not in agauss:
                agauss["sy"] = 2.0

    def get_cmatrices(self):
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        c = []
        for aGauss in self.gausslist:
            n = aGauss["modPolyDeg"] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(
                center=aGauss["center"], sx=aGauss["sx"], sy=aGauss["sy"]
            )
            newc = [
                signal.convolve2d(self.refimage, gaussk * aU * aV, mode="same")
                for i, aU in enumerate(allus)
                for aV in allvs[: n - i]
            ]
            c.extend(newc)
        return c

    def get_kernel(self):
        if self.kernel is not None:
            return self.kernel
        nkcoeffs = 0
        for aGauss in self.gausslist:
            n = aGauss["modPolyDeg"] + 1
            nkcoeffs += n * (n + 1) // 2
        coeffs = self.get_coeffs()
        kcoeffs = coeffs[:nkcoeffs]
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        kernel = np.zeros((kh, kw))
        for aGauss in self.gausslist:
            n = aGauss["modPolyDeg"] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(
                center=aGauss["center"], sx=aGauss["sx"], sy=aGauss["sy"]
            )
            ind = 0
            for i, aU in enumerate(allus):
                for aV in allvs[: n - i]:
                    kernel += kcoeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussk
        self.kernel = kernel
        return self.kernel

    def get_coeffs(self):
        if self.coeffs is not None:
            return self.coeffs
        c = self.get_cmatrices()
        if self.bkgdegree is not None:
            c_bkg = self.get_cmatrices_background()
            c.extend(c_bkg)

        n_c = len(c)
        m = np.zeros((n_c, n_c))
        b = np.zeros(n_c)
        if self.badpixmask is None:
            for j, cj in enumerate(c):
                for i in range(j, n_c):
                    m[j, i] = np.vdot(cj, c[i])
                    m[i, j] = m[j, i]
                b[j] = np.vdot(self.image, cj)

            # m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            # b = np.array([(self.image * ci).sum() for ci in c])
        else:
            for j, cj in enumerate(c):
                for i in range(j, n_c):
                    m[j, i] = (c[i] * cj)[~self.badpixmask].sum()
                    m[i, j] = m[j, i]
                b[j] = (self.image * cj)[~self.badpixmask].sum()

            # These next two lines take most of the computation time
            # ~ m = np.array([[(ci * cj)[~self.badpixmask].sum() for ci in c] for cj in c])
            # ~ b = np.array([(self.image * ci)[~self.badpixmask].sum() for ci in c])
        self.coeffs = np.linalg.solve(m, b)
        return self.coeffs


class BramichStrategy(SubtractionStrategy):
    def get_cmatrices(self):
        kh, kw = self.k_shape
        h, w = self.refimage.shape
        c = []
        for i in range(kh):
            for j in range(kw):
                cij = np.zeros(self.refimage.shape)
                max_r = min(h, h - kh // 2 + i)
                min_r = max(0, i - kh // 2)
                max_c = min(w, w - kw // 2 + j)
                min_c = max(0, j - kw // 2)
                max_r_ref = min(h, h - i + kh // 2)
                min_r_ref = max(0, kh // 2 - i)
                max_c_ref = min(w, w - j + kw // 2)
                min_c_ref = max(0, kw // 2 - j)
                cij[min_r:max_r, min_c:max_c] = self.refimage[
                    min_r_ref:max_r_ref, min_c_ref:max_c_ref
                ]
                c.extend([cij])

        # This is more pythonic but much slower (50 times slower)
        # canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
        # c.extend([signal.convolve2d(refimage, kij, mode='same')
        #                 for kij in canonBasis])
        # canonBasis = None
        return c

    def get_kernel(self):
        if self.kernel is not None:
            return self.kernel
        coeffs = self.get_coeffs()
        kh, kw = self.k_shape
        self.kernel = coeffs[: (kh * kw)].reshape(self.k_shape)
        return self.kernel

    def get_coeffs(self):
        if self.coeffs is not None:
            return self.coeffs
        c = self.get_cmatrices()
        if self.bkgdegree is not None:
            c_bkg = self.get_cmatrices_background()
            c.extend(c_bkg)

        n_c = len(c)
        m = np.zeros((n_c, n_c))
        b = np.zeros(n_c)
        if self.badpixmask is None:
            for j, cj in enumerate(c):
                cj = np.asarray(cj, order="C")
                for i in range(j, n_c):
                    m[j, i] = np.tensordot(cj, np.asarray(c[i], order="C"))
                    m[i, j] = m[j, i]
                b[j] = np.vdot(self.image, cj.flatten())
            # ~ m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            # ~ b = np.array([(self.image * ci).sum() for ci in c])
        else:
            for j, cj in enumerate(c):
                for i in range(j, n_c):
                    m[j, i] = (c[i] * cj)[~self.badpixmask].sum()
                    m[i, j] = m[j, i]
                b[j] = (self.image * cj)[~self.badpixmask].sum()

            # These next two lines take most of the computation time
            # ~ m = np.array([[(ci * cj)[~self.badpixmask].sum() for ci in c] for cj in c])
            # ~ b = np.array([(self.image * ci)[~self.badpixmask].sum() for ci in c])
        self.coeffs = np.linalg.solve(m, b)
        return self.coeffs


class AdaptiveBramichStrategy(SubtractionStrategy):
    def __init__(self, image, refimage, kernelshape, bkgdegree, poly_degree=2):
        self.poly_deg = poly_degree
        self.poly_dof = (poly_degree + 1) * (poly_degree + 2) // 2
        self.k_side = kernelshape[0]

        super(AdaptiveBramichStrategy, self).__init__(
            image, refimage, kernelshape, bkgdegree
        )

    def get_optimal_image(self):
        # AdaptiveBramich has to override this function because it uses a
        # special type of convolution for optimal_image
        if self.optimal_image is not None:
            return self.optimal_image
        import varconv

        opt_image = varconv.convolve2d_adaptive(
            self.refimage, self.get_kernel(), self.poly_deg
        )
        if self.bkgdegree is not None:
            opt_image += self.get_background()
        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image
        return self.optimal_image

    def get_kernel(self):
        if self.kernel is not None:
            return self.kernel
        poly_dof = (self.poly_deg + 1) * (self.poly_deg + 2) // 2
        k_dof = self.k_side * self.k_side * poly_dof
        ks = self.k_side
        coeffs = self.get_coeffs()
        self.kernel = coeffs[:k_dof].reshape((ks, ks, self.poly_dof))
        return self.kernel

    def get_coeffs(self):
        if self.coeffs is not None:
            return self.coeffs
        import varconv

        m, b = varconv.gen_matrix_system(
            self.image,
            self.refimage,
            self.badpixmask is not None,
            self.badpixmask,
            self.k_side,
            self.poly_deg,
            self.bkgdegree or -1,
        )
        self.coeffs = np.linalg.solve(m, b)
        return self.coeffs


def convolve2d_adaptive(image, kernel, poly_degree):
    "Convolve image with the adaptive kernel of `poly_degree` degree."
    import varconv

    # Check here for dimensions
    if image.ndim != 2:
        raise ValueError("Wrong dimensions for image")
    if kernel.ndim != 3:
        raise ValueError("Wrong dimensions for kernel")

    conv = varconv.convolve2d_adaptive(image, kernel, poly_degree)
    return conv


def eval_adpative_kernel(kernel, x, y):
    "Return the adaptive kernel at position (x, y) = (col, row)."
    if kernel.ndim == 2:
        return kernel

    kh, kw, dof = kernel.shape
    # The conversion from degrees of freedom (dof) to the polynomial degree
    # The last 0.5 is to round to nearest integer
    deg = int(-1.5 + np.sqrt(1 + 8 * dof) / 2 + 0.5)
    k_rolled = np.rollaxis(kernel, 2, 0)
    k_xy = np.zeros((kh, kw))
    d = 0
    for powx in range(deg + 1):
        for powy in range(deg - powx + 1):
            k_xy += k_rolled[d] * np.power(y, powy) * np.power(x, powx)
            d += 1
    return k_xy


def optimal_system(
    image,
    refimage,
    kernelshape=(11, 11),
    bkgdegree=None,
    method="Bramich",
    gridshape=None,
    **kwargs
):
    """Do Optimal Image Subtraction and return optimal image, kernel
    and background.

    This is an implementation of a few Optimal Image Subtraction algorithms.
    They all (optionally) simultaneously fit a background.

    Args:
        gridshape: A tuple containing the number of vertical and horizontal
            divisions of a grid. Subtraction will be performed on each grid
            element. ``None`` is equivalent to a ``(1, 1)`` grid (no grid).

        kernelshape: Shape of the kernel to use. Must be of odd size.

        bkgdegree: Degree of the polynomial to fit the background.
            To turn off background fitting set this to ``None``.

        method: One of the following strings.

            * ``"Bramich"``: A Delta basis for the kernel (all pixels fit
              independently)
            * ``"AdaptiveBramich"``: Same as Bramich, but with a polynomial
              variation across the image.
              It needs the parameter ``poly_degree``, which is the polynomial
              degree of the variation.
            * ``"Alard-Lupton"``: A modulated multi-Gaussian kernel.
              It needs the gausslist keyword.

        poly_degree: Needed only for AdaptiveBramich. It is the degree
            of the polynomial for the kernel spatial variation.

        gausslist: Needed only for Alard-Lupton. A list of dictionaries with
            info for the modulated multi-Gaussian.
            Dictionary keys are:

                * center: a (row, column) tuple for the center of the Gaussian.
                  Default: kernel center.
                * modPolyDeg: the degree of the modulating polynomial.
                  Default: 2
                * sx: sigma in x direction. Default: 2.
                * sy: sigma in y direction. Deafult: 2.

            All keys are optional. Example::

                gausslist=[{center: (5, 5), sx: 2., sy: 2., modPolyDeg: 3},
                           {sx: 1.0, sy: 2.5, modPolyDeg: 1},
                           {sx: 3.0, sy: 1.0},]

    Returns:
        difference, optimal_image, kernel, background

    Raises:
        EvenSideKernelError: If any dimension of ``kernelshape`` is even.

    """

    kh, kw = kernelshape

    if (kw % 2 == 0) or (kh % 2 == 0):
        raise EvenSideKernelError("Kernel sides must be odd.")

    DefaultStrategy = BramichStrategy  # noqa
    all_strategies = {
        "AdaptiveBramich": AdaptiveBramichStrategy,
        "Bramich": BramichStrategy,
        "Alard-Lupton": AlardLuptonStrategy,
    }
    try:
        DiffStrategy = all_strategies[method]  # noqa
    except KeyError:
        raise ValueError("No method named {}".format(method))

    if gridshape is None or gridshape == (1, 1):
        # If there's no grid, do without it
        subt_strat = DiffStrategy(
            image, refimage, kernelshape, bkgdegree, **kwargs
        )
        opt_image = subt_strat.get_optimal_image()
        kernel = subt_strat.get_kernel()
        background = subt_strat.get_background()
        difference = subt_strat.get_difference()
        return difference, opt_image, kernel, background

    else:
        ny, nx = gridshape
        h, w = image.shape

        # normal slices with no border
        stamps_y = [
            slice(h * i // ny, h * (i + 1) // ny, None) for i in range(ny)
        ]
        stamps_x = [
            slice(w * i // nx, w * (i + 1) // nx, None) for i in range(nx)
        ]

        # slices with borders where possible
        # Slices should be in (h * i // ny, h * (i + 1) // ny) but we add and
        # subtract the kernel spill k_spill and then we clip to keep it inside
        # image boundaries.
        k_spill = (kh - 1) // 2
        slc_wborder_y = [
            slice(
                np.clip(h * i // ny - k_spill, 0, h),
                np.clip(h * (i + 1) // ny + k_spill, 0, h),
                None,
            )
            for i in range(ny)
        ]
        slc_wborder_x = [
            slice(
                np.clip(w * i // nx - k_spill, 0, w),
                np.clip(w * (i + 1) // nx + k_spill, 0, w),
                None,
            )
            for i in range(nx)
        ]

        img_stamps = [
            image[sly, slx] for sly in slc_wborder_y for slx in slc_wborder_x
        ]
        ref_stamps = [
            refimage[sly, slx]
            for sly in slc_wborder_y
            for slx in slc_wborder_x
        ]

        # After we do the subtraction we need to crop the extra borders in the
        # stamps.
        # The recover_slices are the prescription for what to crop on each stamp.
        recover_slices = []
        for i in range(ny):
            start_border_y = slc_wborder_y[i].start
            stop_border_y = slc_wborder_y[i].stop
            # Slice should end at h * (i + 1) // ny, any other pixels should
            # be trimmed. sly_stop is either negative or 0.
            # In the special case where 0 pixels need to be trimmed
            # we use None so slice goes to the end.
            sly_stop = (h * (i + 1) // ny - stop_border_y) or None
            # Same with initial pixels, but sly_start is positive or 0.
            # Zero is not a special case now (0 is array initial pixel)
            sly_start = h * i // ny - start_border_y
            sly = slice(sly_start, sly_stop, None)
            for j in range(nx):
                start_border_x = slc_wborder_x[j].start
                stop_border_x = slc_wborder_x[j].stop
                slx_stop = (w * (j + 1) // nx - stop_border_x) or None
                slx_start = w * j // nx - start_border_x
                slx = slice(slx_start, slx_stop, None)
                recover_slices.append([sly, slx])

        # Here do the subtraction on each stamp
        if _has_mask(image) or _has_mask(refimage):
            optimal_collage = np.ma.empty(image.shape)
            subtract_collage = np.ma.empty(image.shape)
        else:
            optimal_collage = np.empty(image.shape)
            subtract_collage = np.empty(image.shape)
        bkg_collage = np.empty(image.shape)
        kernel_collage = []
        stamp_slices = [[asly, aslx] for asly in stamps_y for aslx in stamps_x]
        for ind, ((sly_out, slx_out), (sly_in, slx_in)) in enumerate(
            zip(recover_slices, stamp_slices)
        ):

            subt_strat = DiffStrategy(
                img_stamps[ind],
                ref_stamps[ind],
                kernelshape,
                bkgdegree,
                **kwargs
            )
            opti = subt_strat.get_optimal_image()
            ki = subt_strat.get_kernel()
            bgi = subt_strat.get_background()
            di = subt_strat.get_difference()
            optimal_collage[sly_in, slx_in] = opti[sly_out, slx_out]
            bkg_collage[sly_in, slx_in] = bgi[sly_out, slx_out]
            subtract_collage[sly_in, slx_in] = di[sly_out, slx_out]
            kernel_collage.append(ki)

        return subtract_collage, optimal_collage, kernel_collage, bkg_collage
