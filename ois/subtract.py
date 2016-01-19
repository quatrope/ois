"""@package subtraction

    Subtraction module
    -----------------

    A collection of tools to perform optimal image differencing
    for the Transient Optical Robotic Observatory of the South (TOROS).

    ### Usage example (from python):

        >>> from toros import subtraction
        >>> conv_image, optimalKernel, background =
            ois.getOptimalKernelAndBkg(image, referenceImage)

    (conv_image is the least square optimal approximation to image)

    See getOptimalKernelAndBkg docstring for more options.

    ### Command line arguments:
    * -h, --help: Prints this help and exits.
    * -v, --version: Prints version information and exits.

    Martin Beroiz - 2014

    email: <martinberoiz@phys.utb.edu>

    University of Texas at San Antonio
"""

import numpy as np
import math
from scipy import signal
from scipy import ndimage


def _gauss(shape=(10, 10), center=None, sx=2, sy=2):
    h, w = shape
    if center is None:
        center = ((h - 1) / 2., (w - 1) / 2.)
    x0, y0 = center
    x, y = np.meshgrid(range(w), range(h))
    k = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
    norm = k.sum()
    return k / norm


def __CVectorsForGaussBasis(refImage, kernelShape, gaussList,
                            modBkgDeg, badPixMask):
    # degMod is the degree of the modulating polyomial
    kh, kw = kernelShape

    v, u = np.mgrid[:kh, :kw]
    C = []
    for aGauss in gaussList:
        if 'modPolyDeg' in aGauss:
            degMod = aGauss['modPolyDeg']
        else:
            degMod = 2

        allUs = [pow(u, i) for i in range(degMod + 1)]
        allVs = [pow(v, i) for i in range(degMod + 1)]

        if 'center' in aGauss:
            center = aGauss['center']
        else:
            center = None
        gaussK = _gauss(shape=(kh, kw), center=center,
                        sx=aGauss['sx'], sy=aGauss['sy'])

        if badPixMask is not None:
            newC = [np.ma.array(
                    signal.convolve2d(refImage, gaussK * aU * aV, mode='same'),
                    mask=badPixMask)
                    for i, aU in enumerate(allUs)
                    for aV in allVs[:degMod + 1 - i]
                    ]
        else:
            newC = [signal.convolve2d(refImage, gaussK * aU * aV, mode='same')
                    for i, aU in enumerate(allUs)
                    for aV in allVs[:degMod + 1 - i]
                    ]

        C.extend(newC)
    return C


def __CVectorsForDeltaBasis(refImage, kernelShape, badPixMask):
    kh, kw = kernelShape
    h, w = refImage.shape

    C = []
    for i in range(kh):
        for j in range(kw):
            Cij = np.zeros(refImage.shape)
            Cij[max(0, i - kh // 2):min(h, h - kh // 2 + i),
                max(0, j - kw // 2):min(w, w - kw // 2 + j)] =
                refImage[max(0, kh // 2 - i):min(h, h - i + kh // 2), 
                max(0, kw // 2 - j):min(w, w - j + kw // 2)]
            if badPixMask is not None:
                C.extend([np.ma.array(Cij, mask=badPixMask)])
            else:
                C.extend([Cij])

    # This is more pythonic but much slower (50 times slower)
    # canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
    # C.extend([signal.convolve2d(refImage, kij, mode='same')
    #                 for kij in canonBasis])
    # canonBasis = None

    return C


def __getCVectors(refImage, kernelShape, gaussList, modBkgDeg=2,
                  badPixMask=None):

    C = []
    if gaussList is not None:
        C.extend(__CVectorsForGaussBasis(refImage, kernelShape, gaussList, \
                                         modBkgDeg, badPixMask))
    else:
        C.extend(__CVectorsForDeltaBasis(refImage, kernelShape, badPixMask))

    #finally add here the background variation coefficients:
    h, w = refImage.shape
    y, x = np.mgrid[:h,:w]
    allXs = [pow(x,i) for i in range(modBkgDeg + 1)]
    allYs = [pow(y,i) for i in range(modBkgDeg + 1)]

    if badPixMask is not None:
        newC = [np.ma.array(anX * aY, mask = badPixMask) \
                for i, anX in enumerate(allXs) for aY in allYs[:modBkgDeg+1-i]]
    else:
        newC = [anX * aY for i, anX in enumerate(allXs) \
                for aY in allYs[:modBkgDeg+1-i]]

    C.extend(newC)
    return C


def __coeffsToKernel(coeffs, gaussList, kernelShape = (10,10)):
    kh, kw = kernelShape
    if gaussList is None:
        kernel = coeffs[:kw*kh].reshape(kh,kw)
    else:
        v,u = np.mgrid[:kh,:kw]
        kernel = np.zeros((kh,kw))
        for aGauss in gaussList:
            if 'modPolyDeg' in aGauss: degMod = aGauss['modPolyDeg']
            else: degMod = 2

            allUs = [pow(u,i) for i in range(degMod + 1)]
            allVs = [pow(v,i) for i in range(degMod + 1)]

            if 'center' in aGauss: center = aGauss['center']
            else: center = None
            gaussK = _gauss(shape=kernelShape, center=center, sx=aGauss['sx'],\
                             sy=aGauss['sy'])

            ind = 0
            for i, aU in enumerate(allUs):
                for aV in allVs[:degMod+1-i]:
                    kernel += coeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussK
    return kernel


def __coeffsToBackground(shape, coeffs, bkgDeg = None):
    if bkgDeg is None: bkgDeg = int(-1.5 + 0.5*math.sqrt(9 + 8*(len(coeffs) - 1)))

    h, w = shape
    y, x = np.mgrid[:h,:w]
    allXs = [pow(x,i) for i in range(bkgDeg + 1)]
    allYs = [pow(y,i) for i in range(bkgDeg + 1)]

    mybkg = np.zeros(shape)

    ind = 0
    for i, anX in enumerate(allXs):
        for aY in allYs[:bkgDeg+1-i]:
            mybkg += coeffs[ind] * anX * aY
            ind += 1

    return mybkg


def getOptimalKernelAndBkg(image, refImage, gaussList=None,
                           bkgDegree=3, kernelShape=(11, 11)):
    """Do Optimal Image Subtraction and return optimal kernel and background.

    This is an implementation of the Optimal Image Subtraction algorith of
    Alard&Lupton. It returns the best kernel and background fit that match the
    two images.

    gaussList is a list of dictionaries containing data of the gaussians
    used in the decomposition of the kernel. Dictionary keywords are:
    center, sx, sy, modPolyDeg
    If gaussList is None (default value), the OIS will try to optimize
    the value of each pixel in the kernel.

    bkgDegree is the degree of the polynomial to fit the background.

    kernelShape is the shape of the kernel to use.

    Return (optimal_image, kernel, background)
    """

    kh, kw = kernelShape
    if kw % 2 != 1 or kh % 2 != 1:
        print("This can only work with kernels of odd sizes.")
        return None, None, None

    badPixMask = None
    if isinstance(refImage, np.ma.MaskedArray):
        refMask = ndimage.binary_dilation(refImage.mask.astype('uint8'),
                                          structure=np.ones(kernelShape))
        badPixMask = refMask.astype('bool')
        if isinstance(image, np.ma.MaskedArray):
            badPixMask += image.mask
    elif isinstance(image, np.ma.MaskedArray):
        badPixMask = image.mask

    C = __getCVectors(refImage, kernelShape, gaussList, bkgDegree, badPixMask)
    m = np.array([[(ci * cj).sum() for ci in C] for cj in C])
    b = np.array([(image * ci).sum() for ci in C])
    coeffs = np.linalg.solve(m, b)

    # nKCoeffs is the number of coefficients related to the kernel fit,
    # not the background fit
    if gaussList is None:
        nKCoeffs = kh * kw
    else:
        nKCoeffs = 0
        for aGauss in gaussList:
            if 'modPolyDeg' in aGauss:
                degMod = aGauss['modPolyDeg']
            else:
                degMod = 2
            nKCoeffs += (degMod + 1) * (degMod + 2) // 2

    kernel = __coeffsToKernel(coeffs[:nKCoeffs], gaussList, kernelShape)
    background = __coeffsToBackground(image.shape, coeffs[nKCoeffs:])
    if isinstance(refImage, np.ma.MaskedArray) or \
            isinstance(image, np.ma.MaskedArray):
        background = np.ma.array(background, mask=badPixMask)
    optimal_image = signal.convolve2d(refImage, kernel, mode='same') \
        + background

    return optimal_image, kernel, background


def optimalSubtractOnGrid(image, refImage, gaussList=None, bkgDegree=3,
                          kernelShape=(11, 11), gridShape=(2, 2)):
    """Implement Optimal Image Subtraction on a grid and return the optimal
    subtraction

    This is an implementation of the Optimal Image Subtraction algorith of
    Alard&Lupton(1998) and Bramich(2010)
    It returns the optimal subtraction between image and refImage.

    gaussList is a list of dictionaries containing data of the gaussians
    used in the multigaussian decomposition of the kernel [Alard&Lupton, 1998].
    Dictionary keywords are:
    center, sx, sy, modPolyDeg

    If gaussList is None (default value), the OIS will try to optimize the
    value of each pixel in the kernel [Bramich, 2010].

    bkgDegree is the degree of the polynomial to fit the background.

    kernelShape is the shape of the kernel to use.

    gridShape is a tuple containing the number of vertical and horizontal
    divisions of the grid.

    This method does not interpolate between the grids.

    Return (subtraction_array)
    """
    ny, nx = gridShape
    h, w = image.shape
    kh, kw = kernelShape

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

    imgStamps = [image[sly, slx] for sly in slc_wborder_y
                 for slx in slc_wborder_x]
    refStamps = [refImage[sly, slx] for sly in slc_wborder_y
                 for slx in slc_wborder_x]

    # After we do the subtraction we need to crop the extra borders in the
    # stamps.
    # The recover_slices are the prescription for what to crop on each stamp.
    recover_slices = []
    for i in range(ny):
        start_border_y, stop_border_y = slc_wborder_y[i].start, slc_wborder_y[i].stop
        sly_stop = h * (i + 1) / ny - stop_border_y
        if sly_stop == 0:
            sly_stop = None
        sly = slice(h * i / ny - start_border_y, sly_stop, None)
        for j in range(nx):
            start_border_x, stop_border_x = slc_wborder_x[j].start, slc_wborder_x[j].stop
            slx_stop = w * (j + 1) / nx - stop_border_x
            if slx_stop == 0:
                slx_stop = None
            slx = slice(w * j / nx - start_border_x, slx_stop, None)
            recover_slices.append([sly, slx])

    # Here do the subtraction on each stamp
    subtractCollage = np.ma.empty(image.shape)
    stamp_slices = [[sly, slx] for sly in stamps_y for slx in stamps_x]
    for ind, ((sly_out, slx_out), (sly_in, slx_in)) in \
            enumerate(zip(recover_slices, stamp_slices)):
        opti, ki, bgi = getOptimalKernelAndBkg(imgStamps[ind],
                                               refStamps[ind],
                                               gaussList,
                                               bkgDegree,
                                               kernelShape)

        subtractCollage[sly_in, slx_in] = \
            (imgStamps[ind] - opti)[sly_out, slx_out]

    return subtractCollage
