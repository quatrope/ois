import unittest
from ois import ois
import numpy as np


class TestSubtract(unittest.TestCase):
    def setUp(self):
        import urllib
        import cStringIO
        from PIL import Image
        # http://homepages.cae.wisc.edu/~ece533/images/cameraman.tif
        # http://links.uwaterloo.ca/Repository/TIF/camera.tif
        f = cStringIO.StringIO(urllib.urlopen(
            'http://homepages.cae.wisc.edu/~ece533/images/cameraman.tif')
            .read())
        self.ref_img = np.array(Image.open(f), dtype='float32')

    def tearDown(self):
        pass

    def degradereference(self):
        from scipy import signal

        # Set some arbitrary kernel to convolve with
        def gauss(shape=(11, 11), center=None, sx=2, sy=2):
            h, w = shape
            if center is None:
                center = ((h - 1) / 2., (w - 1) / 2.)
            x0, y0 = center
            x, y = np.meshgrid(range(w), range(h))
            norm = np.sqrt(2 * np.pi * (sx ** 2) * (sy ** 2))
            return np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 +
                          (y - y0) ** 2 / sy ** 2)) / norm

        def createkernel(coeffs, gausslist, kernelshape=(10, 10)):
            kh, kw = kernelshape
            v, u = np.mgrid[:kh, :kw]
            mykernel = np.zeros((kh, kw))
            for aGauss in gausslist:
                if 'modPolyDeg' in aGauss:
                    degmod = aGauss['modPolyDeg']
                else:
                    degmod = 2
                allus = [pow(u, i) for i in range(degmod + 1)]
                allvs = [pow(v, i) for i in range(degmod + 1)]
                if 'center' in aGauss:
                    center = aGauss['center']
                else:
                    center = None
                gaussk = gauss(shape=kernelshape, center=center,
                               sx=aGauss['sx'], sy=aGauss['sy'])
                ind = 0
                for i, aU in enumerate(allus):
                    for aV in allvs[:degmod + 1 - i]:
                        mykernel += coeffs[ind] * aU * aV
                        ind += 1
                mykernel *= gaussk
            # mykernel /= mykernel.sum()
            return mykernel

        # mygausslist = [{'sx': 2., 'sy': 2., 'modPolyDeg': 3},
        # {'sx': 1., 'sy': 3.}, {'sx': 3., 'sy': 1.}]
        mygausslist = [{'sx': 2., 'sy': 2., 'modPolyDeg': 3}]
        # mykcoeffs = np.random.rand(10) * 90 + 10
        mykcoeffs = np.array([0., -7.3, 0., 0., 0., 2., 0., 1.5, 0., 0.])

        mykernel = createkernel(mykcoeffs, mygausslist, kernelshape=(11, 11))
        # mykernel = gauss()
        kh, kw = mykernel.shape

        self.image = signal.convolve2d(self.ref_img, mykernel, mode='same')

        # Add a varying background:
        bkgdeg = 2

        h, w = self.ref_img.shape
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(bkgdeg + 1)]
        allys = [pow(y, i) for i in range(bkgdeg + 1)]

        mybkg = np.zeros(self.ref_img.shape)
        mybkgcoeffs = np.random.rand(6) * 1E-3

        ind = 0
        for i, anX in enumerate(allxs):
            for aY in allys[:bkgdeg + 1 - i]:
                mybkg += mybkgcoeffs[ind] * anX * aY
                ind += 1

        self.image += mybkg

    def test_optimalkernelandbkg(self):
        self.degradereference()
        ruined_image, optKernel, bkg = ois.subtract.getOptimalKernelAndBkg(
            self.image, self.ref_img, bkgDegree=2, kernelshape=(11, 11))
        norm_diff = np.linalg.norm(ruined_image - self.image)
        self.assertLess(norm_diff, 1E-6)

    def test_subtractongrid(self):
        pass


if __name__ == "__main__":
    unittest.main()
