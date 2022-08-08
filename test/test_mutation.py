import unittest
from unittest.mock import patch
import numpy
from utils.operators.mutation import *

POPULATION = numpy.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
])

def transform(array):
    return ["{:.2f}".format(item) for item in array]

class TestInsertionMutation(unittest.TestCase):

    @patch('numpy.random.choice')
    def test_mutation_function(self, random_segment):
        random_segment.side_effect = [
            numpy.array([0,1]),
            numpy.array([2,1]),
            numpy.array([0,2]),
            numpy.array([0,1]),
            numpy.array([1,2]),
            numpy.array([0,1])
        ]
        result = insertion_mutation(
            POPULATION,
            2
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[3,2,1], [4,6,5], [8,9,7]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])
   
    @patch('numpy.random.choice')
    def test_mutation_class_method(self, random_segment):
        random_segment.side_effect = [
            numpy.array([0,1]),
            numpy.array([2,1]),
            numpy.array([0,2]),
            numpy.array([0,1]),
            numpy.array([1,2]),
            numpy.array([0,1])
        ]
        method = InsertionMutator(n_elements=2)
        result = method(POPULATION)
        
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [[3,2,1], [4,6,5], [8,9,7]]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Insertion \n\t Arguments:\n\t\t -n_elements: 2")

class TestSigmaEpAdaptiveMutation(unittest.TestCase):

    @patch('numpy.random.normal')
    def test_mutation_function(self, random_segment):
        random_segment.return_value =  numpy.array([
            [ 0.37060191, -0.48970787,  0.83365679],
            [-0.07146474, -0.34716583,  0.97081791],
            [-1.45822813, -0.96113428, -0.46841437]
       ])
        
        result = sigma_ep_adaptive(
            POPULATION,
            alpha= 0.5
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [1.18530095, 1.51029213, 4.25048519],
            [3.85707052, 4.13208542, 8.91245373],
            [1.89620155, 4.15546288, 6.89213533],
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))
   
    @patch('numpy.random.normal')
    def test_mutation_class_method(self, random_segment):
        random_segment.return_value =  numpy.array([
            [ 0.37060191, -0.48970787,  0.83365679],
            [-0.07146474, -0.34716583,  0.97081791],
            [-1.45822813, -0.96113428, -0.46841437]
       ])
        method = SigmaEpAdaptiveMutator(
            decision_variables=3,
            alpha=0.5
        )
        result = method(POPULATION)
        
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [1.18530095, 1.51029213, 4.25048519],
            [3.85707052, 4.13208542, 8.91245373],
            [1.89620155, 4.15546288, 6.89213533],
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Sigma EP.\n\t\t-Alpha: 0.5")

class TestSingleSigmaAdaptiveMutation(unittest.TestCase):
    
    @patch('numpy.random.normal')
    def test_mutation_function(self, random_segment):
        random_segment.return_value = 0.786
        
        result = single_sigma_adaptive(
            POPULATION,
            gamma=0.35
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 1.31666233,  2.63332467,  3.949987  ],
            [ 5.26664934,  6.58331167,  7.89997401],
            [ 9.21663634, 10.53329868, 11.84996101]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))
   
    @patch('numpy.random.normal')
    def test_mutation_class_method(self, random_segment):
        random_segment.return_value = 0.786
        method = SingleSigmaAdaptiveMutator(decision_variables=3)
        result = method(POPULATION)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 2.19460044,  4.38920089,  6.58380133],
            [ 8.77840178, 10.97300222, 13.16760267],
            [15.36220311, 17.55680355, 19.751404  ]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Single Sigma")

if __name__ == '__main__':
    unittest.main(verbosity=3)
