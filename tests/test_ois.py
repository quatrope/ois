import unittest
import ois
import numpy as np
import os
import varconv
import scipy


class TestPSFCorrect(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (100, 100)
        n_stars = 10
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0
        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f
        self.ref = self.img.copy()

        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.7, mode='constant')
        self.ref = gaussian_filter(self.ref, sigma=0.8, mode='constant')

    def test_AlardLupton_diffPSF(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Alard-Lupton",
            gausslist=[{'sx': 1.5, 'sy': 1.5}])
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_Bramich_diffPSF(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Bramich")
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_AdaptiveBramich_diffPSF(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="AdaptiveBramich")
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))


class TestAlignmentCorrect(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (100, 100)
        n_stars = 10
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0
        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f
        self.ref = np.zeros(img_shape)
        self.x0, self.y0 = 2, 1
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.ref[y + self.y0, x + self.x0] = f  # We add a small translation here
        # Let's add a Gaussian PSF response with same seeing for both images
        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.5, mode='constant')
        self.ref = gaussian_filter(self.ref, sigma=1.5, mode='constant')

    def test_AlardLupton_align(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Alard-Lupton",
            kernelshape=(5, 5),
            gausslist=[{'sx': 0.01, 'sy':0.01, 'center': (2 - self.x0, 2 - self.y0), 'modPolyDeg':0}])
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_Bramich_align(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Bramich")
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_AdaptiveBramich_align(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="AdaptiveBramich")
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))


class TestSmallRotationCorrect(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (100, 100)
        n_stars = 10
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0

        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f

        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.5, mode='constant')

        from scipy.ndimage import rotate
        self.ref = rotate(self.img, 0.05)
        hr, wr = self.ref.shape
        self.ref = self.ref[(hr - h) // 2 + (hr - h) % 2: -(hr - h) // 2 or None,
                    (wr - w) // 2 + (wr - w) % 2: -(wr - w) // 2 or None]

    def test_AdaptiveBramich_rotation(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="AdaptiveBramich",
            poly_degree=2)
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))


class TestMasking(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (100, 100)
        n_stars = 10
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0
        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f
        self.ref = self.img.copy()

        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.7, mode='constant')
        self.ref = gaussian_filter(self.ref, sigma=0.8, mode='constant')

        mask = np.zeros(img_shape, dtype='bool')
        self.img_masked = self.img.copy()
        self.img_masked[50:60, 50:60] = 5000
        mask[50:60, 50:60] = True    
        self.img_masked = np.ma.array(self.img_masked, mask=mask)

        mask = np.zeros(img_shape, dtype='bool')
        self.ref_masked = self.ref.copy()
        self.ref_masked[80:90, 80:90] = 5000
        mask[80:90, 80:90] = True    
        self.ref_masked = np.ma.array(self.ref_masked, mask=mask)

    def test_AlardLupton_mask(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img_masked, self.ref_masked,
            method="Alard-Lupton",
            gausslist=[{'sx': 1.5, 'sy': 1.5}])
        norm_diff = np.linalg.norm(diff.compressed()) \
            / np.linalg.norm(self.ref_masked.compressed())
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's returning masked arrays
        self.assertTrue(isinstance(diff, np.ma.MaskedArray))
        self.assertTrue(isinstance(opt, np.ma.MaskedArray))

    def test_Bramich_mask(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img_masked, self.ref_masked,
            method="Bramich")
        norm_diff = np.linalg.norm(diff.compressed()) \
            / np.linalg.norm(self.ref_masked.compressed())
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's returning masked arrays
        self.assertTrue(isinstance(diff, np.ma.MaskedArray))
        self.assertTrue(isinstance(opt, np.ma.MaskedArray))

    def test_AdaptiveBramich_mask(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img_masked, self.ref_masked,
            method="AdaptiveBramich")
        norm_diff = np.linalg.norm(diff.compressed()) \
            / np.linalg.norm(self.ref_masked.compressed())
        # Assert it's a good subtraction
        self.assertLess(norm_diff, 1E-3)
        # Assert it's returning masked arrays
        self.assertTrue(isinstance(diff, np.ma.MaskedArray))
        self.assertTrue(isinstance(opt, np.ma.MaskedArray))


class TestBackground(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (100, 100)
        n_stars = 10
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0
        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f
        self.ref = self.img.copy()

        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.7, mode='constant')
        self.ref = gaussian_filter(self.ref, sigma=0.8, mode='constant')

        xv, yv = np.meshgrid(np.arange(h, dtype='float'), np.arange(w, dtype='float'))
        self.bkg = 2 * (xv - 5) ** 2 + (yv - 4) ** 2 + 3 * xv * yv
        self.img += self.bkg

    def test_background_AlardLupton(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Alard-Lupton",
            gausslist=[{'sx': 1.5, 'sy': 1.5}],
            bkgdegree=2)
        # Assert it's a good subtraction
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        self.assertLess(norm_diff, 1E-3)
        # Assert we had a good background estimation
        norm_bkg_diff = np.linalg.norm(bkg - self.bkg) / np.linalg.norm(bkg)
        self.assertLess(norm_bkg_diff, 1E-3)

        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_background_Bramich(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="Bramich",
            bkgdegree=2)
        # Assert it's a good subtraction
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        self.assertLess(norm_diff, 1E-3)
        # Assert we had a good background estimation
        norm_bkg_diff = np.linalg.norm(bkg - self.bkg) / np.linalg.norm(bkg)
        self.assertLess(norm_bkg_diff, 1E-3)

        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))

    def test_background_AdaptiveBramich(self):
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method="AdaptiveBramich",
            bkgdegree=2)
        # Assert it's a good subtraction
        norm_diff = np.linalg.norm(diff) / np.linalg.norm(self.ref)
        self.assertLess(norm_diff, 1E-3)
        # Assert we had a good background estimation
        norm_bkg_diff = np.linalg.norm(bkg - self.bkg) / np.linalg.norm(bkg)
        self.assertLess(norm_bkg_diff, 1E-3)

        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))


class TestGrid(unittest.TestCase):
    def setUp(self):
        h, w = img_shape = (200, 300)
        n_stars = 60
        pos_x = np.random.randint(10, w - 10, n_stars)
        pos_y = np.random.randint(10, h - 10, n_stars)
        fluxes = 200.0 + np.random.rand(n_stars) * 300.0
        self.img = np.zeros(img_shape)
        for x, y, f in zip(pos_x, pos_y, fluxes):
            self.img[y, x] = f
        self.ref = self.img.copy()

        from scipy.ndimage.filters import gaussian_filter
        self.img = gaussian_filter(self.img, sigma=1.7, mode='constant')
        self.ref = gaussian_filter(self.ref, sigma=0.8, mode='constant')

    def the_grid_test(self, method_name, **kwargs):
        gh, gw = 2, 2
        diff, opt, krn, bkg = ois.optimal_system(
            self.img, self.ref,
            method=method_name,
            gridshape=(gh, gw),
            kernelshape=(11, 11),
            **kwargs)

        # Assert it's not returning masked arrays
        self.assertFalse(isinstance(diff, np.ma.MaskedArray))
        self.assertFalse(isinstance(opt, np.ma.MaskedArray))
        
        h, w = self.img.shape
        diff_grid = diff[:h // gh, :w // gw]
        k_spill = 5
        img_crop = self.img[:h // gh + k_spill, :w // gw + k_spill]
        ref_crop = self.ref[:h // gh + k_spill, :w // gw + k_spill]
        diff_crop, opt, krn, bkg = ois.optimal_system(
            img_crop, ref_crop,
            method=method_name,
            gridshape=None,
            kernelshape=(11, 11),
            **kwargs)
        diff_crop = diff_crop[:h // gh, :w // gw]
        norm_diff = np.linalg.norm(diff_grid - diff_crop)
        # Assert it does the same on grid or not
        self.assertLess(norm_diff, 1E-10)

        h, w = self.img.shape
        diff_grid = diff[:h // gh, w // gw:]  
        k_spill = 5
        img_crop = self.img[:h // gh + k_spill, w // gw - k_spill:]
        ref_crop = self.ref[:h // gh + k_spill, w // gw - k_spill:]
        diff_crop, opt, krn, bkg = ois.optimal_system(
            img_crop, ref_crop,
            method=method_name,
            gridshape=None,
            kernelshape=(11, 11),
            **kwargs)
        diff_crop = diff_crop[:h // gh, k_spill:]
        norm_diff = np.linalg.norm(diff_grid - diff_crop)
        # Assert it does the same on grid or not
        self.assertLess(norm_diff, 1E-10)

    def test_AlardLupton_grid(self):
        self.the_grid_test("Alard-Lupton", gausslist=[{'sx': 1.5, 'sy': 1.5}])

    def test_Bramich_grid(self):
        self.the_grid_test("Bramich")

    def test_AdaptiveBramich_grid(self):
        self.the_grid_test("AdaptiveBramich", poly_degree=2)


class TestVarConv(unittest.TestCase):
    def test_gen_matrix_system_sizes(self):
        deg = 2
        bkg_deg = 0
        k_side = 3
        n, m = 10, 10
        image = np.random.random((n, m))
        refimage = image.copy()
        mm, b, c = varconv.gen_matrix_system(image, refimage, 0, None,
                                             k_side, deg, bkg_deg)
        pol_dof = (deg + 1) * (deg + 2) // 2
        bkg_dof = (bkg_deg + 1) * (bkg_deg + 2) // 2
        k_size = k_side * k_side
        m_dof = pol_dof * k_size + bkg_dof
        self.assertEqual(mm.shape, (m_dof, m_dof))
        self.assertEqual(b.shape, (m_dof,))
        self.assertEqual(c.shape, (k_size, pol_dof, bkg_dof, n * m))

    def test_gen_matrix_system_constantkernel(self):
        deg = 0
        k_side = 3
        n, m = 10, 10
        image = np.random.random((n, m))
        refimage = image.copy()
        mm, b, c = varconv.gen_matrix_system(image, refimage, 0, None,
                                             k_side, deg, -1)
        coeffs = np.linalg.solve(mm, b)
        kc = k_side // 2
        result_kernel = coeffs.reshape((k_side, k_side))
        best_kernel = np.zeros((k_side, k_side))
        best_kernel[kc, kc] = 1.0
        self.assertLess(np.linalg.norm(result_kernel - best_kernel), 1E-10)

    def test_gen_matrix_system_constantkernel_masked(self):
        deg = 0
        k_side = 3
        n, m = 10, 10
        image = np.random.random((n, m))
        refimage = image.copy()
        mask = np.zeros((n, m), dtype='bool')
        mask[3:5, 3:5]
        mm, b, c = varconv.gen_matrix_system(image, refimage, 1, mask,
                                             k_side, deg, -1)
        coeffs = np.linalg.solve(mm, b)
        kc = k_side // 2
        result_kernel = coeffs.reshape((k_side, k_side))
        best_kernel = np.zeros((k_side, k_side))
        best_kernel[kc, kc] = 1.0
        self.assertLess(np.linalg.norm(result_kernel - best_kernel), 1E-10)

    def test_convolve2d_adaptive_idkernel(self):
        kernel = np.zeros((3, 3, 1), dtype="float64")
        kernel[1, 1, 0] = 1.0
        image = np.random.random((10, 10))
        # image = np.arange(100, dtype="float64").reshape((10, 10))
        conv = varconv.convolve2d_adaptive(image, kernel, 0)
        self.assertEqual(conv.shape, image.shape)
        self.assertLess(np.linalg.norm(image - conv), 1E-10)

    def test_convolve2d_adaptive_undoing(self):
        deg = 2
        k_side = 3
        pol_dof = (deg + 1) * (deg + 2) // 2
        kernel = np.random.random((k_side, k_side, pol_dof))
        refimage = np.random.random((10, 10))
        image = varconv.convolve2d_adaptive(refimage, kernel, deg)
        mm, b, c = varconv.gen_matrix_system(image, refimage, 0, None,
                                             k_side, deg, -1)
        coeffs = np.linalg.solve(mm, b)
        result_kernel = coeffs.reshape((k_side, k_side, pol_dof))
        opt_ref = varconv.convolve2d_adaptive(refimage, result_kernel, deg)
        self.assertLess(np.linalg.norm(opt_ref - image, ord=np.inf) /
                        np.linalg.norm(image, ord=np.inf), 1E-8)
        self.assertLess(np.linalg.norm((kernel - result_kernel).flatten(),
                                       ord=np.inf) /
                        np.linalg.norm(kernel.flatten(), ord=np.inf), 1E-8)

    def test_convolve2d_adaptive_cameraman(self):
        from PIL import Image
        this_dir = os.path.abspath(os.path.dirname(__file__))
        cameraman_path = os.path.join(this_dir, "cameraman.tif")
        refimage = np.array(Image.open(cameraman_path), dtype='float64')

        # degrade reference
        deg = 2
        k_side = 3
        pol_dof = (deg + 1) * (deg + 2) // 2
        kernel = np.random.random((k_side, k_side, pol_dof))
        image = varconv.convolve2d_adaptive(refimage, kernel, deg)

        mm, b, c = varconv.gen_matrix_system(image, refimage, 0, None,
                                             k_side, deg, -1)
        coeffs = np.linalg.solve(mm, b)
        result_kernel = coeffs.reshape((k_side, k_side, pol_dof))

        opt_ref = varconv.convolve2d_adaptive(refimage, result_kernel, deg)
        self.assertLess(np.linalg.norm(opt_ref - image, ord=np.inf) /
                        np.linalg.norm(image, ord=np.inf), 1E-8)

    def test_convolve2d_adaptive_cameraman_masked(self):
        from PIL import Image
        this_dir = os.path.abspath(os.path.dirname(__file__))
        cameraman_path = os.path.join(this_dir, "cameraman.tif")
        refimage = np.array(Image.open(cameraman_path), dtype='float64')
        mask = np.zeros(refimage.shape, dtype='bool')
        mask[3:5, 3:5]

        # degrade reference
        deg = 2
        k_side = 3
        pol_dof = (deg + 1) * (deg + 2) // 2
        kernel = np.random.random((k_side, k_side, pol_dof))
        image = varconv.convolve2d_adaptive(refimage, kernel, deg)

        mm, b, c = varconv.gen_matrix_system(image, refimage, 1, mask,
                                             k_side, deg, -1)
        coeffs = np.linalg.solve(mm, b)
        result_kernel = coeffs.reshape((k_side, k_side, pol_dof))

        opt_ref = varconv.convolve2d_adaptive(refimage, result_kernel, deg)
        self.assertLess(np.linalg.norm(opt_ref - image, ord=np.inf) /
                        np.linalg.norm(image, ord=np.inf), 1E-8)

    def test_both_bramich_consistency(self):
        k_side = 3
        image = np.random.random((10, 10))
        refimage = np.random.random((10, 10))
        k_shape = (k_side, k_side)

        diff, opt_img, opt_k, bkg = ois.optimal_system(
            image, refimage, kernelshape=k_shape, bkgdegree=None,
            method="Bramich")
        diff, opt_img, opt_vark, bkg = ois.optimal_system(
            image, refimage, kernelshape=k_shape, bkgdegree=None,
            method="AdaptiveBramich", poly_degree=0)

        self.assertEqual(opt_vark.shape, (k_side, k_side, 1))
        opt_vark = opt_vark.reshape((k_side, k_side))

        diff_norm = np.linalg.norm((opt_k - opt_vark).flatten(), ord=np.inf)
        kernel_norm = np.linalg.norm(opt_k.flatten(), ord=np.inf)
        self.assertLess(diff_norm / kernel_norm, 1E-8)

    def test_eval_adpative_kernel(self):
        test_k = np.array([[[ 2.], [ 3.]], [[ 1.], [ 4.]]])
        self.assertLess(np.max(np.abs(ois.eval_adpative_kernel(test_k, 0, 0)
                                      - np.array([[ 2.,  3.],[ 1.,  4.]]))), 1E-10)
        self.assertLess(np.max(np.abs(ois.eval_adpative_kernel(test_k, 0, 1)
                                      - np.array([[ 2.,  3.],[ 1.,  4.]]))), 1E-10)
        self.assertLess(np.max(np.abs(ois.eval_adpative_kernel(test_k, 1, 0)
                                      - np.array([[ 2.,  3.],[ 1.,  4.]]))), 1E-10)
        self.assertLess(np.max(np.abs(ois.eval_adpative_kernel(test_k, 1, 1)
                                      - np.array([[ 2.,  3.],[ 1.,  4.]]))), 1E-10)

    def test_convolve2d_adaptive_dtype_check(self):
        kernel = np.random.random((3, 3, 1))
        ois.convolve2d_adaptive(np.random.randint(low=0, high=100, size=(100, 100), dtype='int32'), kernel, 0)
        ois.convolve2d_adaptive(np.random.random((100, 100)), kernel.astype('int32'), 0)


class TestExceptions(unittest.TestCase):
    def setUp(self):
        self.img = np.random.random((100, 100))
        self.ref = np.random.random((100, 100))

    def test_wrong_method_name(self):
        with self.assertRaises(ValueError):
            diff, opt_image, krn, bkg = ois.optimal_system(
                self.img, self.ref, method="WrongName")

    def test_even_side_kernel(self):
        for bad_shape in ((8, 9), (9, 8), (8, 8)):
            with self.assertRaises(ois.EvenSideKernelError):
                ois.optimal_system(self.img, self.ref, bad_shape)

    def test_image_dims(self):
        with self.assertRaises(ValueError):
            diff, opt_image, krn, bkg = ois.optimal_system(
                self.img, np.random.random((10, 10, 100)))
        with self.assertRaises(ValueError):
            diff, opt_image, krn, bkg = ois.optimal_system(
                np.random.random((10, 10, 100)), self.ref)
        with self.assertRaises(ValueError):
            diff, opt_image, krn, bkg = ois.optimal_system(
                np.zeros((5, 5)), np.zeros((7, 7)))

    def test_convolve2d_array_dims(self):
        with self.assertRaises(ValueError):
            ois.convolve2d_adaptive(np.zeros((10, 10, 2)), np.ones((3, 3, 6)), 2)
        with self.assertRaises(ValueError):
            ois.convolve2d_adaptive(np.zeros((10, 10)), np.ones((9, 6)), 2)


if __name__ == "__main__":
    unittest.main()
