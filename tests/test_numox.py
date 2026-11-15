import gc
import sys
import unittest
import numpy as np

import numox as nc


class TestNumox(unittest.TestCase):
    def test_constructors_and_shape(self):
        m0 = nc.Matrix(2, 3)
        self.assertEqual(m0.shape, (2, 3))
        self.assertEqual(m0.to_list(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        m1 = nc.Matrix(2, 3, 2.5)
        self.assertEqual(m1.to_list(), [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]])

        m_1d_row = nc.Matrix(1, 4, [1, 2.5, 3, 4])
        self.assertEqual(m_1d_row.shape, (4,))
        self.assertEqual(m_1d_row.to_list(), [1.0, 2.5, 3.0, 4.0])

        m_1d_col = nc.Matrix(4, 1, [1, 2, 3, 4])
        self.assertEqual(m_1d_col.shape, (4,))
        self.assertEqual(m_1d_col.to_list(), [1.0, 2.0, 3.0, 4.0])

        m2 = nc.Matrix([[1, 2], [3, 4.5]])
        self.assertEqual(m2.shape, (2, 2))
        self.assertEqual(m2.to_list(), [[1.0, 2.0], [3.0, 4.5]])

        r1 = nc.Matrix(5, 5, rand=True, low=-2.0, high=3.0, seed=42)
        r2 = nc.Matrix(5, 5, rand=True, low=-2.0, high=3.0, seed=42)
        self.assertEqual(r1.to_list(), r2.to_list())
        arr = np.asarray(r1)
        self.assertTrue(np.all(arr >= -2.0) and np.all(arr < 3.0))

    def test_shape_readonly_and_invalid_init(self):
        m = nc.Matrix(2, 2, 1.0)
        with self.assertRaises(AttributeError):
            m.shape = (1, 4)
        with self.assertRaises(ValueError):
            nc.Matrix(0, 3)
        with self.assertRaises(ValueError):
            nc.Matrix(2, 2, [1, 2, 3])
        with self.assertRaises(ValueError):
            nc.Matrix([])
        with self.assertRaises(TypeError):
            nc.Matrix(2, 2, ["a", "b", "c", "d"])

    def test_arithmetic_against_numpy(self):
        rng = np.random.default_rng(2026)
        for m, k, n in [(1, 1, 1), (7, 9, 11), (37, 53, 41), (130, 140, 129)]:
            a_np = rng.uniform(-5.0, 5.0, (m, k))
            b_np = rng.uniform(-5.0, 5.0, (k, n))
            c_np = rng.uniform(-5.0, 5.0, (m, k))

            a_nc = nc.Matrix(a_np.tolist())
            b_nc = nc.Matrix(b_np.tolist())
            c_nc = nc.Matrix(c_np.tolist())

            self.assertTrue(np.allclose(np.asarray(a_nc + c_nc), a_np + c_np, atol=1e-12))
            self.assertTrue(np.allclose(np.asarray(a_nc - c_nc), a_np - c_np, atol=1e-12))
            self.assertTrue(np.allclose(np.asarray(-a_nc), -a_np, atol=1e-12))
            self.assertTrue(np.allclose(np.asarray(abs(a_nc)), np.abs(a_np), atol=1e-12))
            self.assertTrue(np.allclose(np.asarray(a_nc * b_nc), a_np @ b_np, atol=1e-10))
            self.assertTrue(np.allclose(np.asarray(a_nc @ b_nc), a_np @ b_np, atol=1e-10))

    def test_pow(self):
        a_np = np.array([[1.0, 1.0], [1.0, 0.0]])
        a_nc = nc.Matrix(a_np.tolist())

        self.assertTrue(np.allclose(np.asarray(a_nc ** 0), np.eye(2), atol=1e-12))
        self.assertTrue(np.allclose(np.asarray(a_nc ** 1), a_np, atol=1e-12))
        self.assertTrue(np.allclose(np.asarray(a_nc ** 10), np.linalg.matrix_power(a_np, 10), atol=1e-10))

        with self.assertRaises(TypeError):
            _ = a_nc ** 1.5
        with self.assertRaises(ValueError):
            _ = a_nc ** -1
        with self.assertRaises(ValueError):
            _ = nc.Matrix(2, 3, 1.0) ** 2

    def test_indexing_slicing_and_nested_views(self):
        # 1D row & col vector slicing
        rv = nc.Matrix(1, 5, [10, 20, 30, 40, 50])
        self.assertEqual(rv[2], 30.0)
        self.assertEqual(rv[1:4].to_list(), [20.0, 30.0, 40.0])
        rv[1:3] = [200, 300.5]
        self.assertEqual(rv.to_list(), [10.0, 200.0, 300.5, 40.0, 50.0])

        cv = nc.Matrix(5, 1, [1, 2, 3, 4, 5])
        self.assertEqual(cv[3], 4.0)
        self.assertEqual(cv[1:4].to_list(), [2.0, 3.0, 4.0])

        # 2D slicing & nested slice views
        m = nc.Matrix(4, 4, list(range(16)))
        sub1 = m[1:4, 1:4]
        sub2 = sub1[0:2, 0:2]
        self.assertEqual(sub2.to_list(), [[5.0, 6.0], [9.0, 10.0]])
        sub2[0, 1] = 99.5
        self.assertEqual(m[1, 2], 99.5)
        self.assertEqual((sub2 ** 0).to_list(), [[1.0, 0.0], [0.0, 1.0]])

        # 2D slice assignment
        m[0] = [10, 20, 30, 40]
        self.assertEqual(m[0].to_list(), [10.0, 20.0, 30.0, 40.0])
        m[2:4, 2:4] = [[1.5, 2.5], [3.5, 4.5]]
        self.assertEqual(m[2:4, 2:4].to_list(), [[1.5, 2.5], [3.5, 4.5]])

        # Bounds & invalid keys
        with self.assertRaises(IndexError):
            _ = m[10, 0]
        with self.assertRaises(IndexError):
            m[-1, 0] = 5.0
        with self.assertRaises(ValueError):
            _ = m[0:4:2]
        with self.assertRaises(ValueError):
            _ = m[2:2]

        del sub1, sub2, m
        gc.collect()

    def test_buffer_protocol_zero_copy(self):
        m = nc.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = np.asarray(m)
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float64)

        arr[1, 2] = 42.0
        self.assertEqual(m.get(1, 2), 42.0)
        m.set(0, 1, -7.5)
        self.assertEqual(arr[0, 1], -7.5)

        # Strided slice view via buffer protocol
        sub = m[0:2, 1:3]
        sub_arr = np.asarray(sub)
        self.assertEqual(sub_arr.shape, (2, 2))
        self.assertTrue(np.allclose(sub_arr, [[-7.5, 3.0], [5.0, 42.0]]))
        sub_arr[0, 0] = 100.0
        self.assertEqual(m.get(0, 1), 100.0)

    def test_no_refcount_leaks(self):
        m = nc.Matrix(3, 3, 1.0)
        _ = repr(m)
        gc.collect()

        before = sys.gettotalrefcount() if hasattr(sys, "gettotalrefcount") else None
        m_ref_before = sys.getrefcount(m)
        for _ in range(500):
            _ = repr(m)
            mv = memoryview(m)
            del mv
            tmp = nc.Matrix(1, 300, 2.0)
            del tmp
        gc.collect()
        m_ref_after = sys.getrefcount(m)
        self.assertEqual(m_ref_after, m_ref_before)
        if before is not None:
            after = sys.gettotalrefcount()
            self.assertLess(after - before, 20)

    def test_uninitialized_and_reinit(self):
        u = nc.Matrix.__new__(nc.Matrix)
        with self.assertRaises(AttributeError):
            _ = u.shape
        with self.assertRaises(BufferError):
            _ = memoryview(u)

        for fn in (
            lambda: repr(u),
            lambda: u.to_list(),
            lambda: nc.to_list(u),
            lambda: u + u,
            lambda: u - u,
            lambda: u * u,
            lambda: u @ u,
            lambda: -u,
            lambda: abs(u),
            lambda: u ** 2,
            lambda: u[0],
            lambda: u.get(0, 0),
            lambda: u.set(0, 0, 1.0),
        ):
            with self.assertRaises(RuntimeError):
                fn()

        with self.assertRaises(RuntimeError):
            u[0] = 1.0

        valid = nc.Matrix(2, 2, 1.0)
        with self.assertRaises(RuntimeError):
            _ = valid + u
        with self.assertRaises(RuntimeError):
            _ = valid @ u

        valid.__init__(3, 3, 2.0)
        self.assertEqual(valid.shape, (3, 3))
        self.assertEqual(valid.get(2, 2), 2.0)

        mv = memoryview(valid)
        with self.assertRaises(BufferError):
            valid.__init__(4, 4, 1.0)
        self.assertEqual(valid.shape, (3, 3))
        mv.release()
        del mv

        valid.__init__(4, 4, 3.0)
        self.assertEqual(valid.shape, (4, 4))
        self.assertEqual(valid.get(3, 3), 3.0)

    def test_package_metadata(self):
        self.assertEqual(nc.Matrix.__module__, "numox")
        if hasattr(nc, "_core"):
            self.assertIsInstance(nc.__version__, str)
            self.assertIn("Matrix", nc.__all__)
            self.assertIn("to_list", nc.__all__)


if __name__ == "__main__":
    unittest.main()
