from importlib.resources import path
import unittest
from unittest import result
from unittest.mock import patch
import numpy
from utils.operators.selection import *

def transform(array):
    """
    Transform de floating numbers with more decimals to two decimals.
    """
    return [f"{item:.2f}" for item in array]

class TestUtilsSelection(unittest.TestCase):
    """
    Tests complementary functions that need every selection operator.
    """

    def test_expected_value(self):
        """
        Suite of cases for complementary function.
        """
        
        aptitude_array =  numpy.array([1,2,3],dtype=float)
        result = get_expected_values(aptitude_array)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),3)

        #It should return a valid numpy array.
        self.assertListEqual(list(result), [0.5, 1.,  1.5])

    @patch('numpy.argpartition')
    def test_candidates_by_aptitude(self, pseudo_sorting):
        """
        Suite of cases for complementary function.
        """

        pseudo_sorting.return_value = numpy.array([1,1,2,2,3,5,6,7,8])
        input_array = numpy.array([3,2,5,1,2,6,1,8,7])
        result = get_candidates_by_aptitude(input_array,3)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the 3 biggest numbers.
        self.assertEqual(len(result),3)

        self.assertEqual(list(result), [6,7,8])

class TestProporcionalSamplingSelection(unittest.TestCase):

    @patch('numpy.arange')
    def test_proporcional_function(self, numpy_range):
        numpy_range.return_value = numpy.array([0,1,2,3,4,5])
        input_array = numpy.array([1,2,3,4,5,5])
        result = proporcional_sampling(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),6)

        #It should return a valid numpy array.
        self.assertListEqual(list(result), list(range(6)))

    @patch('numpy.arange')
    def test_proporcional_class(self, numpy_range):
        numpy_range.return_value = numpy.array([0,1,2,3,4,5])
        input_array = numpy.array([1,2,3,4,5,5])
        method = ProporcionalSampler()
        result = method(input_array)
        
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        self.assertEqual(len(result),6)

        #It should return this result.
        self.assertListEqual(list(result), list(range(6)))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Proporcional sampling")

class TestRouletteSamplingSelection(unittest.TestCase):

    @patch('numpy.random.uniform')
    def test_proporcional_function(self, random_numbers):
        random_numbers.return_value = numpy.array([0.3,0.78,0.89,0.3,0.1,1])
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        result = roulette_sampling(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),6)

        # #It should return this result.
        self.assertListEqual(list(result), [0,1,1,0,0,2])

    @patch('numpy.random.uniform')
    def test_proporcional_class(self, random_numbers):
        random_numbers.return_value = numpy.array([0.3,0.78,0.89,0.3,0.1,1])
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        method = RouletteSampler()
        result = method(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        self.assertEqual(len(result),6)

        #It should return this result.
        self.assertListEqual(list(result), [0,1,1,0,0,2])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Roulette sampling")

class TestStochasticUniversalSamplingSelection(unittest.TestCase):

    @patch('numpy.random.uniform')
    def test_proporcional_function(self, random_number):
        random_number.return_value = 0.78
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        result = stochastic_universal_sampling(input_array)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),6)

        # #It should return this result.
        self.assertListEqual(list(result), [1,2,3,4,5,5])

    @patch('numpy.random.uniform')
    def test_proporcional_class(self, random_number):
        random_number.return_value = 0.78
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        method = StochasticUniversalSampler()
        result = method(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        self.assertEqual(len(result),6)

        #It should return this result.
        self.assertListEqual(list(result), [1,2,3,4,5,5])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Stochastic universal sampling")

class TestDeterministicSamplingSelection(unittest.TestCase):

    def test_proporcional_function(self):
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        result = deterministic_sampling(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),6)

        # #It should return this result.
        self.assertListEqual(list(result), [3,4,5,5,1,2])

    def test_proporcional_class(self):
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        method = DeterministicSampler()
        result = method(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        self.assertEqual(len(result),6)

        #It should return this result.
        self.assertListEqual(list(result), [3,4,5,5,1,2])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Deterministic sampling")

class TestTournamentSamplingSelection(unittest.TestCase):

    @patch('numpy.random.permutation')
    @patch('numpy.random.rand')
    def test_proporcional_function(self, random_number, random_permutation):
        random_number.side_effect = [0.76,0.23,0.45,0.98,0.47,0.28,0.18,0.84,0.65]
        random_permutation.side_effect = [
            numpy.array([0,2,1,4,5,3]),
            numpy.array([4,3,2,5,1,0]),
            numpy.array([1,3,2,4,0,5])
        ]
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        result = tournament_sampling(input_array, chunks=2, prob=0.5)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should return a numpy array with the same size as the input.
        self.assertEqual(len(result),6)

        # #It should return this result.
        self.assertListEqual(list(result), [0,4,5,3,5,1])

    @patch('numpy.random.permutation')
    @patch('numpy.random.rand')
    def test_proporcional_class(self, random_number, random_permutation):
        random_number.side_effect = [0.76,0.23,0.45,0.98,0.47,0.28,0.18,0.84,0.65]
        random_permutation.side_effect = [
            numpy.array([0,2,1,4,5,3]),
            numpy.array([4,3,2,5,1,0]),
            numpy.array([1,3,2,4,0,5])
        ]
        input_array = numpy.array([0.10,0.2,0.3,0.4,0.5,0.5])
        method = TournamentSampler()
        result = method(input_array)

        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        self.assertEqual(len(result),6)

        #It should return this result.
        self.assertListEqual(list(result), [2,4,5,4,5,1])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__,\
             "Tournament sampling\n\t Arguments:\n\t\t-Chunks: 2\n\t\t-prob: 1.0")

class TestMergeSelector(unittest.TestCase):

    def test_class_method(self):
        aptitude_parents = numpy.array([0.1,0.3,0.2])
        aptitude_offspring = numpy.array([0.3,0.3,0.01])
        population_features = {}
        population_features['solutions'] = [
            [
                numpy.array([1,2,3]),
                numpy.array([2,3,1]),
                numpy.array([5,3,1])
            ],
            [
                numpy.array([4,3,1]),
                numpy.array([4,3,1]),
                numpy.array([1,1,1])
            ]
        ]

        method = MergeSelector()
        result = method(aptitude_parents, aptitude_offspring, population_features)

        #It should return a python dictionary.
        self.assertEqual(type(result), dict)

        #It should return only one key.
        self.assertListEqual(list(result.keys()), ['parent_population_f', 'solutions'])

        #It should return this result.
        valid_result = [[2,3,1], [4,3,1], [4,3,1]]
        for ind,array in enumerate(result['solutions']):
            self.assertListEqual(list(array), valid_result[ind])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Merge population")

        #It should catch error when the feature has only one array instead of 2.
        population_features['solutions'] = [[1,2]]
        with self.assertRaises(Exception):
            method(aptitude_parents, aptitude_offspring, population_features)

class TestReplacementSelector(unittest.TestCase):

    def test_class_method(self):
        aptitude_parents = numpy.array([0.1,0.3,0.2])
        aptitude_offspring = numpy.array([0.3,0.3,0.01])
        population_features = {}
        population_features['solutions'] = [
            numpy.array([
                numpy.array([1,2,3]),
                numpy.array([2,3,1]),
                numpy.array([5,3,1])
            ]),
            numpy.array([
                numpy.array([4,3,1]),
                numpy.array([4,3,1]),
                numpy.array([1,1,1])
            ])
        ]

        method = ReplacementSelector()
        result = method(aptitude_parents, aptitude_offspring, population_features)
        #It should return a python dictionary.
        self.assertEqual(type(result), dict)

        #It should return only one key.
        self.assertListEqual(list(result.keys()), ['parent_population_f', 'solutions'])

        #It should return this result.
        valid_result = [[1,1,1], [4,3,1], [4,3,1]]
        for ind,array in enumerate(result['solutions']):
            self.assertListEqual(list(array), valid_result[ind])

        #It should catch error when the feature has only one array instead of 2.
        population_features['solutions'] = [[1,2]]
        with self.assertRaises(Exception):
            method(aptitude_parents, aptitude_offspring, population_features)

        #It should catch an exception when the offspring size is lower to the parent size. 
        with self.assertRaises(AssertionError):
            method(aptitude_parents,[], population_features)
if __name__ == '__main__':
    unittest.main(verbosity=3)
