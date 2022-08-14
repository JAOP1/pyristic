import unittest
from unittest.mock import patch
import numpy
from utils.operators.crossover import *

population = numpy.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])
left_ind_selected = [0,0,1]
right_ind_selected = [1,1,2]

class TestDiscreteCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """

    def test_crossover_function(self):
        """
        Suite of cases for the function operator.
        """

        with patch('numpy.random.randint') as mock_random_binary_number:
            mock_random_binary_number.side_effect = [1,0,0,1,1,1,0,1,0]
            result = discrete_crossover(
                population, 
                left_ind_selected,
                right_ind_selected
                )

            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 3)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[1,5,6], [1,2,3], [7,5,9]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

    def test_crossover_class_method(self):
        """
        Suite of cases for the class operator.
        """
        #It should return new population when create the crossover as a class.
        method = DiscreteCrossover()
        with patch('numpy.random.randint') as mock_random_binary_number:
            mock_random_binary_number.side_effect = [1,0,0,1,1,1,0,1,0]
            result = method(
                population,
                left_ind_selected,
                right_ind_selected
                )

            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 3)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[1,5,6], [1,2,3], [7,5,9]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

        #It should raise an error when the indices are not the same size.
        p_ind1 = []
        p_ind2 = [1]
        with self.assertRaises(AssertionError):
            method(population, p_ind1, p_ind2)

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, 'Discrete')

class TestIntermediateCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """
    def test_crossover_function(self):
        """
        Suite of cases for the function operator.
        """

        result = intermediate_crossover(
            population,
            [0,2],
            [1,0]
            )

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[2.5,3.5,4.5], [4,5,6]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

    def test_crossover_class_method(self):
        """
        Suite of cases for the class operator.
        """
        #It should return new population when create the crossover as a class.
        method = IntermediateCrossover(alpha=0.5)
        result = method(
            population,
            [0,2],
            [1,0]
            )

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[2.5,3.5,4.5], [4,5,6]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

        #It should raise an error when the indices are not the same size.
        p_ind1 = []
        p_ind2 = [1]
        with self.assertRaises(AssertionError):
            method(population, p_ind1, p_ind2)

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Intermediate\n\tArguments:\n\t\t-Alpha:0.5")

class TestNpointsCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """

    def test_crossover_function(self):
        """
        Suite of cases for the function operator.
        """

        with patch('numpy.random.choice') as random_permutation:
            random_permutation.side_effect = [
                numpy.array([0]),
                numpy.array([1])
                ]
            result = n_point_crossover(
                population,
                [0,2],
                [1,0]
                )
            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 4)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[1,5,6], [4,2,3], [7,8,3], [1,2,9]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

    def test_crossover_class_method(self):
        """
        Suite of cases for the class operator.
        """
        #It should return new population when create the crossover as a class.
        method = NPointCrossover(n_cross=1)
        with patch('numpy.random.choice') as random_permutation:
            random_permutation.side_effect = [
                numpy.array([0]),
                numpy.array([1])
                ]
            result = method(
                population,
                [0,2],
                [1,0]
                )
            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 4)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[1,5,6], [4,2,3], [7,8,3], [1,2,9]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

            #It should raise an error when the indices are not the same size.
            p_ind1 = []
            p_ind2 = [1]
            with self.assertRaises(AssertionError):
                method(population, p_ind1, p_ind2)

            #It should save the method type in the doc string.
            self.assertEqual(method.__doc__, "n point\n\tArguments:\n\t\t-n_cross: 1")

class TestUniformCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """

    def test_crossover_function(self):
        """
        Suite of cases for the function operator.
        """

        with patch('numpy.random.rand') as random_number:
            random_number.side_effect = [
                0.13,
                0.54,
                0.76
            ]
            result = uniform_crossover(
                population,
                [0],
                [1],
                flip_prob=0.6
            )
            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 2)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[4, 5, 3], [1, 2, 6]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

    def test_crossover_class_method(self):
        """
        Suite of cases for the class operator.
        """

        #It should return new population when create the crossover as a class.
        method = UniformCrossover(flip_prob=0.6)
        with patch('numpy.random.rand') as random_number:
            random_number.side_effect = [
                0.13,
                0.54,
                0.76
            ]
            result = method(
                population,
                [0],
                [1]
                )
            #It should return a numpy array.
            self.assertEqual(type(result).__module__, 'numpy')

            #It should be the same size as the original population.
            rows,cols = result.shape
            self.assertEqual(rows, 2)
            self.assertEqual(cols, 3)

            #It should return this result.
            valid_result = [[4, 5, 3], [1, 2, 6]]
            for i in range(rows):
                self.assertListEqual(list(result[i]), valid_result[i])

            #It should raise an error when the indices are not the same size.
            p_ind1 = []
            p_ind2 = [1]
            with self.assertRaises(AssertionError):
                method(population, p_ind1, p_ind2)

            #It should save the method type in the doc string.
            self.assertEqual(method.__doc__, "Uniform\n\tArguments:\n\t\t-prob: 0.6")

class TestPermutationOrderCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """

    @patch('numpy.random.choice')
    def test_crossover_function(self, random_segments):
        """
        Suite of cases for the function operator.
        """

        permutations = [ 
            numpy.array([1,2,3]),
            numpy.array([2,3,1])
        ]
        random_segments.side_effect = [
            numpy.array([2,1]),
            numpy.array([0,2])
        ]
        result = permutation_order_crossover(
            permutations,
            [0],
            [1],
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        # #It should return this result.
        valid_result = [[3,2,1], [2, 3, 1]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

    @patch('numpy.random.choice')
    def test_crossover_class_method(self, random_segments):
        """
        Suite of cases for the class operator.
        """

        permutations = [
            numpy.array([1,2,3]),
            numpy.array([2,3,1])
        ]
        random_segments.side_effect = [
            numpy.array([2,1]),
            numpy.array([0,2])
        ]

        #It should return new population when create the crossover as a class.
        method = PermutationOrderCrossover()

        result = method(
            permutations,
            [0],
            [1]
            )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[3,2,1], [2, 3, 1]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

        #It should raise an error when the indices are not the same size.
        p_ind1 = []
        p_ind2 = [1]
        with self.assertRaises(AssertionError):
            method(population, p_ind1, p_ind2)

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Permutation order")

class TestSimulatedBinaryCrossover(unittest.TestCase):
    """
    Tests for the function and class method of discrete crossover.
    """

    def transform(self, array):
        """
        Transform de floating numbers with more decimals to two decimals.
        """
        return [f"{item:.2f}" for item in array]

    @patch('numpy.random.rand')
    def test_crossover_function(self, random_number):
        """
        Suite of cases for the function operator.
        """

        random_number.return_value = 0.78
        result = simulated_binary_crossover(
            population,
            [0],
            [1],
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        # #It should return this result.
        valid_result = [[0.23866492, 1.23866492, 2.23866492], [4.76133508, 5.76133508, 6.76133508]]
        for i in range(rows):
            self.assertListEqual(self.transform(list(result[i])), self.transform(valid_result[i]))

    @patch('numpy.random.rand')
    def test_crossover_class_method(self, random_number):
        """
        Suite of cases for the class operator.
        """

        random_number.return_value = 0.78

        #It should return new population when create the crossover as a class.
        method = SimulatedBinaryCrossover(nc=1)

        result = method(
            population,
            [0],
            [1]
            )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[0.23866492, 1.23866492, 2.23866492], [4.76133508, 5.76133508, 6.76133508]]
        for i in range(rows):
            self.assertListEqual(self.transform(list(result[i])), self.transform(valid_result[i]))

        #It should raise an error when the indices are not the same size.
        p_ind1 = []
        p_ind2 = [1]
        with self.assertRaises(AssertionError):
            method(population, p_ind1, p_ind2)

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Simulated binary\n\tArguments:\n\t\t-nc: 1")

if __name__ == '__main__':
    unittest.main(verbosity=3)
