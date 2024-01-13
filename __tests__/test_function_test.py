import unittest
from utils.test_function import *


class TestBealeSuite(unittest.TestCase):
    def test_objective_function(self):
        # It should display a positive value.
        cases = [[-1, -1], [1, -1], [-1, 1]]
        for case in cases:
            self.assertTrue(beale_function(case) >= 0)

        # It should display the minimum value.
        self.assertAlmostEqual(beale_function([3, 0.5]), 0)

    def test_contraint(self):
        # It should return true when the input is between the boundaries
        cases = [[-4.49, 2], [-4, 4.49]]
        for case in cases:
            self.assertTrue(constraint1_beale(case))

        # It should return false when the input has a value out of the bounds.
        cases = [[-4.51, 1], [1, 4.51], [-4.6, 0]]
        for case in cases:
            self.assertFalse(constraint1_beale(case))

        # It should return the doc string when execute the beale constraint.
        constraint1_beale([3, 0.5])
        self.assertEqual(
            constraint1_beale.__doc__,
            "x1: -4.5 <= 3.00 <= 4.5 \n x2: -4.5 <= 0.50 <= 4.5",
        )


class TestAckleySuite(unittest.TestCase):
    def test_objective_function(self):
        # It should display a positive value.
        cases = [[-1, -1], [1, -1, 1, 2, 3, 4], [-1, 1, 1]]
        for case in cases:
            self.assertTrue(ackley_function(np.array(case)) >= 0)

        # It should display the minimum value.
        for num_var in range(1, 10):
            self.assertAlmostEqual(ackley_function(np.array([0] * num_var)), 0)

    def test_contraint(self):
        # It should return true when the input is between the boundaries
        cases = [[-30, 2], [-4, 30, 4], [-23, 3, 13, -23]]
        for case in cases:
            self.assertTrue(constraint1_ackley(case))

        # It should return false when the input has a value out of the bounds.
        cases = [[-31, 1, 0, 24], [-1, 32], [0, 0, 0, 0, 0, 30, 31]]
        for case in cases:
            self.assertFalse(constraint1_ackley(case))

        # It should return the doc string when execute the beale constraint.
        constraint1_ackley([3, 0.5])
        self.assertEqual(
            constraint1_ackley.__doc__,
            "x1: -30 <= 3.00 <= 30 \n x2: -30 <= 0.50 <= 30 \n ",
        )


class TestBukinSuite(unittest.TestCase):
    def test_objective_function(self):
        # It should display a positive value.
        cases = [[-15, -3], [0, 0], [1, -2], [-1, 1]]
        for case in cases:
            self.assertTrue(bukin_function(np.array(case)) >= 0)

        # It should display the minimum value.
        self.assertAlmostEqual(bukin_function(np.array([-10, 1])), 0)

    def test_contraint(self):
        # It should return true when the input is between the boundaries
        cases = [[-15, 2], [-6, -3], [-5, 1], [-6, 3]]
        for case in cases:
            self.assertTrue(constraint1_bukin(case))

        # It should return false when the input has a value out of the bounds.
        cases = [[-16, 1], [-4, 1], [-6, -4], [-6, 4]]
        for case in cases:
            self.assertFalse(constraint1_bukin(case))

        # It should return the doc string when execute the beale constraint.
        constraint1_bukin([-6, -3])
        self.assertEqual(
            constraint1_bukin.__doc__,
            "x1: -15 <= -6.00 <= -5 \n x2: -3 <= -3.00 <= 3 \n ",
        )


class TestHimmelblauSuite(unittest.TestCase):
    def test_objective_function(self):
        # It should display a positive value.
        cases = [[-15, -3], [0, 0], [1, -2], [-1, 1]]
        for case in cases:
            self.assertTrue(himmelblau_function(np.array(case)) >= 0)

        # It should display the minimum value.
        self.assertAlmostEqual(
            himmelblau_function(np.array([-0.27085, -0.923039])), 181.617, places=2
        )

    def test_contraint(self):
        # It should return true when the input is between the boundaries
        cases = [[-5, 1], [5, 1], [1, -5], [1, 5]]
        for case in cases:
            self.assertTrue(constraint1_himmelblau(case))

        # It should return false when the input has a value out of the bounds.
        cases = [[-6, 1], [6, 1], [1, -6], [1, 6]]
        for case in cases:
            self.assertFalse(constraint1_himmelblau(case))

        # It should return the doc string when execute the beale constraint.
        constraint1_himmelblau([-3, -3])
        self.assertEqual(
            constraint1_himmelblau.__doc__,
            "x1: -5 <= -3.00 <= 5 \n x2: -5 <= -3.00 <= 5 \n ",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
