"""
    Optimal Image Subtraction (OIS) module
    --------------------------------------

    A collection of tools to perform optimal image differencing
    for the Transient Optical Robotic Observatory of the South (TOROS).

    ### Usage example (from python):

        >>> import ois
        >>> difference, optimalImage, optimalKernel, background =
                                   ois.optimal_system(image, referenceImage)

    (conv_image is the least square optimal approximation to image)

    See optimalkernelandbkg docstring for more options.

    ### Command line arguments:
    * -h, --help: Prints this help and exits.
    * -v, --version: Prints version information and exits.

    (c) Martin Beroiz

    email: <martinberoiz@gmail.com>

    University of Texas at San Antonio
"""

__version__ = '0.1.2'

import numpy as np
from scipy import signal
from scipy import ndimage


class EvenSideKernelError(ValueError):
    pass


def _has_mask(image):
    is_masked_array = isinstance(image, np.ma.MaskedArray)
    if is_masked_array and isinstance(image.mask, np.ndarray):
        return True
    return False


class SubtractionStrategy(object):

    def __init__(self, image, refimage, kernelshape, bkgdegree):
        self.image = image
        self.refimage = refimage
        # Check here for dimensions
        if self.image.ndim != 2:
            raise ValueError("Wrong dimensions for image")
        if self.refimage.ndim != 2:
            raise ValueError("Wrong dimensions for refimage")
        if self.image.shape != self.refimage.shape:
            raise ValueError("Images have different shapes")
        self.h, self.w = image.shape

        self.k_shape = kernelshape
        self.k_side = kernelshape[0]
        self.bkgdegree = bkgdegree

        self.image_data, self.refimage_data, self.badpixmask =\
            self.separate_data_mask()

        self.optimal_image = None
        self.background = None
        self.kernel = None
        self.difference = None

    def separate_data_mask(self):
        def ret_data(image):
            if isinstance(image, np.ma.MaskedArray):
                image_data = image.data
            else:
                image_data = image
            return image_data
        badpixmask = None
        if _has_mask(self.refimage):
            badpixmask = ndimage.binary_dilation(
                self.refimage.mask.astype('uint8'),
                structure=np.ones(self.k_shape)).astype('bool')
            if _has_mask(self.image):
                badpixmask += self.image.mask
        elif _has_mask(self.image):
            badpixmask = self.image.mask
        return ret_data(self.image), ret_data(self.refimage), badpixmask

    def coeffstobackground(self, coeffs):
        bkgdeg = int(-1.5 + 0.5 * np.sqrt(9 + 8 * (len(coeffs) - 1)))
        h, w = self.h, self.w
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(bkgdeg + 1)]
        allys = [pow(y, i) for i in range(bkgdeg + 1)]
        mybkg = np.zeros(self.image.shape)
        ind = 0
        for i, anX in enumerate(allxs):
            for aY in allys[:bkgdeg + 1 - i]:
                mybkg += coeffs[ind] * anX * aY
                ind += 1
        return mybkg

    def get_cmatrices_background(self):
        h, w = self.refimage.shape
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(self.bkgdegree + 1)]
        allys = [pow(y, i) for i in range(self.bkgdegree + 1)]
        bkg_c = [anX * aY for i, anX in enumerate(allxs)
                 for aY in allys[:self.bkgdegree + 1 - i]]
        return bkg_c

    def make_system():
        pass

    def get_optimal_image(self):
        if self.optimal_image is None:
            self.make_system()
        return self.optimal_image

    def get_background(self):
        if self.background is None:
            self.make_system()
        return self.background

    def get_kernel(self):
        if self.kernel is None:
            self.make_system()
        return self.kernel

    def get_difference(self):
        if self.difference is None:
            self.make_system()
        return self.difference


class AlardLuptonStrategy(SubtractionStrategy):

    def __init__(self, image, refimage, kernelshape, bkgdegree, gausslist):
        super(AlardLuptonStrategy, self).\
            __init__(image, refimage, kernelshape, bkgdegree)
        if gausslist is None:
            self.gausslist = [{}]
        else:
            self.gausslist = gausslist
        self.clean_gausslist()

    def gauss(self, center, sx, sy):
        h, w = self.k_shape
        x0, y0 = center
        x, y = np.meshgrid(range(w), range(h))
        k = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
        norm = k.sum()
        return k / norm

    def clean_gausslist(self):
        for agauss in self.gausslist:
            if 'center' not in agauss:
                h, w = self.k_shape
                agauss['center'] = ((h - 1) / 2., (w - 1) / 2.)
            if 'modPolyDeg' not in agauss:
                agauss['modPolyDeg'] = 2
            if 'sx' not in agauss:
                agauss['sx'] = 2.
            if 'sy' not in agauss:
                agauss['sy'] = 2.

    def get_cmatrices(self):
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        c = []
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(center=aGauss['center'],
                                sx=aGauss['sx'], sy=aGauss['sy'])
            newc = [signal.convolve2d(self.refimage, gaussk * aU * aV,
                                      mode='same')
                    for i, aU in enumerate(allus)
                    for aV in allvs[:n - i]
                    ]
            c.extend(newc)
        return c

    def coeffstokernel(self, coeffs):
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        kernel = np.zeros((kh, kw))
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(center=aGauss['center'],
                                sx=aGauss['sx'], sy=aGauss['sy'])
            ind = 0
            for i, aU in enumerate(allus):
                for aV in allvs[:n - i]:
                    kernel += coeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussk
        return kernel

    def make_system(self):

        c = self.get_cmatrices()
        if self.bkgdegree is not None:
            c_bkg = self.get_cmatrices_background()
            c.extend(c_bkg)

        if self.badpixmask is None:
            m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            b = np.array([(self.image * ci).sum() for ci in c])
        else:
            # These next two lines take most of the computation time
            m = np.array([[(ci * cj)[~self.badpixmask].sum()
                         for ci in c] for cj in c])
            b = np.array([(self.image * ci)[~self.badpixmask].sum()
                         for ci in c])
        coeffs = np.linalg.solve(m, b)

        nkcoeffs = 0
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            nkcoeffs += n * (n + 1) // 2

        self.kernel = self.coeffstokernel(coeffs[:nkcoeffs])
        opt_image = signal.convolve2d(self.refimage_data, self.kernel,
                                      mode='same')
        if self.bkgdegree is not None:
            self.background = self.coeffstobackground(coeffs[nkcoeffs:])
            opt_image += self.background
        else:
            self.background = np.zeros(self.image.shape)

        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image
        self.difference = self.image - self.optimal_image


class BramichStrategy(SubtractionStrategy):

    def get_cmatrices(self):
        kh, kw = self.k_shape
        h, w = self.refimage_data.shape
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
                cij[min_r:max_r, min_c:max_c] = \
                    self.refimage[min_r_ref:max_r_ref, min_c_ref:max_c_ref]
                c.extend([cij])

        # This is more pythonic but much slower (50 times slower)
        # canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
        # c.extend([signal.convolve2d(refimage, kij, mode='same')
        #                 for kij in canonBasis])
        # canonBasis = None

        return c

    def make_system(self):
        c = self.get_cmatrices()
        if self.bkgdegree is not None:
            c_bkg = self.get_cmatrices_background()
            c.extend(c_bkg)

        if self.badpixmask is None:
            m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            b = np.array([(self.image * ci).sum() for ci in c])
        else:
            # These next two lines take most of the computation time
            m = np.array([[(ci * cj)[~self.badpixmask].sum()
                         for ci in c] for cj in c])
            b = np.array([(self.image * ci)[~self.badpixmask].sum()
                         for ci in c])
        coeffs = np.linalg.solve(m, b)

        kh, kw = self.k_shape
        nkcoeffs = kh * kw
        self.kernel = coeffs[:nkcoeffs].reshape(self.k_shape)
        opt_image = signal.convolve2d(self.refimage_data, self.kernel,
                                      mode='same')
        if self.bkgdegree is not None:
            self.background = self.coeffstobackground(coeffs[nkcoeffs:])
            opt_image += self.background
        else:
            self.background = np.zeros(self.image.shape)

        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image
        self.difference = self.image - self.optimal_image


class AdaptiveBramichStrategy(SubtractionStrategy):
    def __init__(self, image, refimage, kernelshape, bkgdegree, poly_degree=2):
        self.poly_deg = poly_degree
        self.poly_dof = (poly_degree + 1) * (poly_degree + 2) / 2
        super(AdaptiveBramichStrategy, self).\
            __init__(image, refimage, kernelshape, bkgdegree)

    def make_system(self):
        import varconv

        # Check here for types
        if self.image_data.dtype != np.float64:
            img64 = self.image_data.astype('float64')
        else:
            img64 = self.image_data
        if self.refimage_data.dtype != np.float64:
            ref64 = self.refimage_data.astype('float64')
        else:
            ref64 = self.refimage_data

        c_bkgdegree = -1 if self.bkgdegree is None else self.bkgdegree
        m, b, conv = varconv.gen_matrix_system(img64, ref64,
                                               self.badpixmask is not None,
                                               self.badpixmask,
                                               self.k_side, self.poly_deg,
                                               c_bkgdegree)
        coeffs = np.linalg.solve(m, b)
        poly_dof = (self.poly_deg + 1) * (self.poly_deg + 2) / 2
        k_dof = self.k_side * self.k_side * poly_dof
        ks = self.k_side
        self.kernel = coeffs[:k_dof].reshape((ks, ks, self.poly_dof))
        opt_conv = varconv.convolve2d_adaptive(ref64, self.kernel,
                                               self.poly_deg)
        if self.bkgdegree is not None:
            self.background = self.coeffstobackground(coeffs[k_dof:])
            self.optimal_image = opt_conv + self.background
        else:
            self.background = np.zeros(self.image.shape)
            self.optimal_image = opt_conv

        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(self.optimal_image,
                                             mask=self.badpixmask)

        self.difference = self.image - self.optimal_image


def convolve2d_adaptive(image, kernel, poly_degree):
    import varconv

    # Check here for dimensions
    if image.ndim != 2:
        raise ValueError("Wrong dimensions for image")
    if kernel.ndim != 3:
        raise ValueError("Wrong dimensions for kernel")

    # Check here for types
    if image.dtype != np.float64:
        img64 = image.astype('float64')
    else:
        img64 = image
    if kernel.dtype != np.float64:
        k64 = kernel.astype('float64')
    else:
        k64 = kernel

    conv = varconv.convolve2d_adaptive(img64, k64, poly_degree)
    return conv


def optimal_system(image, refimage, kernelshape=(11, 11), bkgdegree=3,
                   method="Bramich", **kwargs):
    """Do Optimal Image Subtraction and return optimal image, kernel
    and background.

    This is an implementation of a few Optimal Image Subtraction algorithms.
    They all (optionally) simultaneously fit a background.

    kernelshape: shape of the kernel to use. Must be of odd size.

    bkgdegree: degree of the polynomial to fit the background.
    To turn off background fitting set this to None.

    method: One of the following strings
    * Bramich: A Delta basis for the kernel (all pixels fit
      independently)
    * AdaptiveBramich: Same as Bramich, but with a polynomial variation across
      the image.
      It needs the parameter poly_degree, which is the polynomial degree of the
      variation.
    * Alard-Lupton: A modulated multi-Gaussian kernel.
      It needs the gausslist keyword.
      gausslist is a list of dictionaries containing data of the gaussians
      used in the decomposition of the kernel. Dictionary keywords are:
      center, sx, sy, modPolyDeg

    Extra parameters are passed to the individual methods.
    poly_degree: needed only for AdaptiveBramich. It is the degree
    of the polynomial for the kernel spatial variation.

    gausslist: needed only for Alard-Lupton. A list of dictionaries with info
    for the modulated multi-Gaussian.
        Dictionary keys are:
        center: a (row, column) tuple for the center of the Gaussian.
            Default: kernel center.
        modPolyDeg: the degree of the modulating polynomial. Default: 2
        sx: sigma in x direction. Default: 2.
        sy: sigma in y direction. Deafult: 2.
        All keys are optional.

    Return (difference, optimal_image, kernel, background)
    """

    kh, kw = kernelshape

    if (kw % 2 == 0) or (kh % 2 == 0):
        raise EvenSideKernelError("Kernel sides must be odd.")

    DefaultStrategy = BramichStrategy # noqa
    all_strategies = {"AdaptiveBramich": AdaptiveBramichStrategy,
                      "Bramich": BramichStrategy,
                      "Alard-Lupton": AlardLuptonStrategy}
    DiffStrategy = all_strategies.get(method, DefaultStrategy) # noqa

    subt_strat = DiffStrategy(image, refimage, kernelshape, bkgdegree,
                              **kwargs)
    opt_image = subt_strat.get_optimal_image()
    kernel = subt_strat.get_kernel()
    background = subt_strat.get_background()
    difference = subt_strat.get_difference()
    return difference, opt_image, kernel, background


def subtractongrid(image, refimage, kernelshape=(11, 11), bkgdegree=3,
                   gridshape=(2, 2), method="Bramich", **kwargs):
    """This implements optimal_system on each section of a grid on the image.

    The parameters are the same as for optimal_system except for
    gridshape: a tuple containing the number of vertical and horizontal
    divisions of the grid.
    (This method does not interpolate between the grids.)

    kernelshape: shape of the kernel to use. Must be of odd size.

    bkgdegree: degree of the polynomial to fit the background.
    To turn off background fitting set this to None.

    method: One of the following strings
    * Bramich: A Delta basis for the kernel (all pixels fit
      independently)
    * AdaptiveBramich: Same as Bramich, but with a polynomial variation across
      the image.
      It needs the parameter poly_degree, which is the polynomial degree of the
      variation.
    * Alard-Lupton: A modulated multi-Gaussian kernel.
      It needs the gausslist keyword.
      gausslist is a list of dictionaries containing data of the gaussians
      used in the decomposition of the kernel. Dictionary keywords are:
      center, sx, sy, modPolyDeg

    Extra parameters are passed to the individual methods.
    poly_degree: needed only for AdaptiveBramich. It is the degree
    of the polynomial for the kernel spatial variation.

    gausslist: needed only for Alard-Lupton. A list of dictionaries with info
    for the modulated multi-Gaussian.
        Dictionary keys are:
        center: a (row, column) tuple for the center of the Gaussian.
            Default: kernel center.
        modPolyDeg: the degree of the modulating polynomial. Default: 2
        sx: sigma in x direction. Default: 2.
        sy: sigma in y direction. Deafult: 2.
        All keys are optional.

    Return (difference, optimal_image, kernel, background)
    """
    ny, nx = gridshape
    h, w = image.shape
    kh, kw = kernelshape

    if (kw % 2 == 0) or (kh % 2 == 0):
        raise EvenSideKernelError("Kernel sides must be odd.")

    DefaultStrategy = BramichStrategy # noqa
    all_strategies = {"AdaptiveBramich": AdaptiveBramichStrategy,
                      "Bramich": BramichStrategy,
                      "Alard-Lupton": AlardLuptonStrategy}
    DiffStrategy = all_strategies.get(method, DefaultStrategy) # noqa

    # normal slices with no border
    stamps_y = [slice(h * i / ny, h * (i + 1) / ny, None) for i in range(ny)]
    stamps_x = [slice(w * i / nx, w * (i + 1) / nx, None) for i in range(nx)]

    # slices with borders where possible
    slc_wborder_y = [slice(max(0, h * i / ny - (kh - 1) / 2),
                           min(h, h * (i + 1) / ny + (kh - 1) / 2), None)
                     for i in range(ny)]
    slc_wborder_x = [slice(max(0, w * i / nx - (kw - 1) / 2),
                           min(w, w * (i + 1) / nx + (kw - 1) / 2), None)
                     for i in range(nx)]

    img_stamps = [image[sly, slx] for sly in slc_wborder_y
                  for slx in slc_wborder_x]
    ref_stamps = [refimage[sly, slx] for sly in slc_wborder_y
                  for slx in slc_wborder_x]

    # After we do the subtraction we need to crop the extra borders in the
    # stamps.
    # The recover_slices are the prescription for what to crop on each stamp.
    recover_slices = []
    for i in range(ny):
        start_border_y = slc_wborder_y[i].start
        stop_border_y = slc_wborder_y[i].stop
        sly_stop = h * (i + 1) / ny - stop_border_y
        if sly_stop == 0:
            sly_stop = None
        sly = slice(h * i / ny - start_border_y, sly_stop, None)
        for j in range(nx):
            start_border_x = slc_wborder_x[j].start
            stop_border_x = slc_wborder_x[j].stop
            slx_stop = w * (j + 1) / nx - stop_border_x
            if slx_stop == 0:
                slx_stop = None
            slx = slice(w * j / nx - start_border_x, slx_stop, None)
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
    for ind, ((sly_out, slx_out), (sly_in, slx_in)) in \
            enumerate(zip(recover_slices, stamp_slices)):

        subt_strat = DiffStrategy(img_stamps[ind], ref_stamps[ind],
                                  kernelshape,
                                  bkgdegree,
                                  **kwargs)
        opti = subt_strat.get_optimal_image()
        ki = subt_strat.get_kernel()
        bgi = subt_strat.get_background()
        di = subt_strat.get_difference()
        optimal_collage[sly_in, slx_in] = opti[sly_out, slx_out]
        bkg_collage[sly_in, slx_in] = bgi[sly_out, slx_out]
        subtract_collage[sly_in, slx_in] = di[sly_out, slx_out]
        kernel_collage.append(ki)

    return subtract_collage, optimal_collage, kernel_collage, bkg_collage
