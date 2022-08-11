import unittest
from unittest.mock import patch
import numpy
from utils.operators.mutation import *

POPULATION = numpy.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
], dtype=float)

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

class TestMultSigmaAdaptiveMutation(unittest.TestCase):
    
    @patch('numpy.random.normal')
    def test_mutation_function(self, random_segment):
        random_segment.side_effect = [
            0.786, 
            numpy.array([
                [0.55,0.23,0.78],
                [0.34, 0.67, 0.97],
                [0.51,0.28,0.33]
            ])]
        
        result = mult_sigma_adaptive(
            POPULATION,
            gamma=0.35,
            gamma_prime=0.54
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 1.85324842,  3.31377221,  6.02581228],
            [ 6.88768036,  9.66371271, 12.88030588],
            [12.79238599, 13.48909447, 15.44313514]
            ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))
   
    @patch('numpy.random.normal')
    def test_mutation_class_method(self, random_segment):
        random_segment.side_effect = [
            0.786, 
            numpy.array([
                [0.55,0.23,0.78],
                [0.34, 0.67, 0.97],
                [0.51,0.28,0.33]
            ])]
        method = MultSigmaAdaptiveMutator(decision_variables=3)
        result = method(POPULATION)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 1.8522292 ,  3.11929146,  6.28758918],
            [ 6.61840501,  9.87792032, 13.92671645],
            [12.68992838, 12.81689841, 14.81161651]
            ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Sigma mult")

class TestMutationBySigmaMutation(unittest.TestCase):

    @patch('numpy.random.normal')
    def test_mutation_function(self, random_segment):
        random_segment.return_value =  0.786

        result = mutation_by_sigma(
            POPULATION,
            0.43
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [1.33798, 2.33798, 3.33798],
            [4.33798, 5.33798, 6.33798],
            [7.33798, 8.33798, 9.33798]
            ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

class TestBinaryMutation(unittest.TestCase):

    @patch('numpy.random.rand')
    def test_mutation_function(self, random_numbers):
        random_numbers.side_effect=[0.23,0.54,0.34,0.76]
        result = binary_mutation(
            numpy.array([
                [0,0],
                [1,1]
            ]),
            pm=0.5
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 2)

        #It should return this result.
        valid_result = [
            [1,0],
            [0,1]
            ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))
   
    @patch('numpy.random.rand')
    def test_mutation_class_method(self, random_numbers):
        random_numbers.side_effect=[0.23,0.54,0.34,0.76]
        method = BinaryMutator(pm=0.5)
        result = method(
            numpy.array([
                [0,0],
                [1,1]
            ]))
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 2)
        self.assertEqual(cols, 2)

        #It should return this result.
        valid_result = [
            [1,0],
            [0,1]
            ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Binary mutation \n\t Arguments:\n\t\t - probability to flip: 0.5")

class TestExchangeMutation(unittest.TestCase):
    @patch('numpy.random.choice')
    def test_mutation_function(self, random_segment):
        random_segment.side_effect = [
            numpy.array([2,1]),
            numpy.array([0,2]),
            numpy.array([1,0])
        ]

        result = exchange_mutation(
            POPULATION
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [1, 3, 2],
            [6, 5, 4],
            [8, 7, 9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

class TestBoundaryMutation(unittest.TestCase):

    @patch('numpy.random.randint')
    @patch('numpy.random.rand')
    def test_mutation_function_array_bound(self, random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,0.23,0.51]
        random_int_number.side_effect = [0,2,1]
        result = boundary_mutationArray(
            POPULATION,
            [-3,-5,-1],
            [3,5,1]
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 3,  2,  3],
            [ 4,  5, -5],
            [ 7,  1,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

    @patch('numpy.random.randint')
    @patch('numpy.random.rand')
    def test_mutation_function_number_bound(self, random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,0.23,0.51]
        random_int_number.side_effect = [0,2,1]
        result = boundary_mutation(
            POPULATION,
            -3,
            3
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 3,  2,  3],
            [ 4,  5, -3],
            [ 7,  3,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

    @patch('numpy.random.randint')
    @patch('numpy.random.rand')
    def test_mutation_class_method_number(self,  random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,0.23,0.51]
        random_int_number.side_effect = [0,2,1]  

        #It should return a matrix with individuals mutated when the bound is a list.      
        method = BoundaryMutator([-3,3])
        result = method(POPULATION)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ 3,  2,  3],
            [ 4,  5, -3],
            [ 7,  3,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Boundary\n\t Arguments:\n\t\t -Lower bound: -3\n\t\t -Upper bound: 3")

    @patch('numpy.random.randint')
    @patch('numpy.random.rand')
    def test_mutation_class_method_array(self,  random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,0.23,0.51]
        random_int_number.side_effect = [0,2,1]  

        #It should return a matrix with individuals mutated when the bound is a list.      
        method = BoundaryMutator(
            [[-3,-5,-1],
            [3,5,1]]
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
            [ 3,  2,  3],
            [ 4,  5, -5],
            [ 7,  1,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Boundary\n\t Arguments:\n\t\t -Lower bound: [-3, -5, -1]\n\t\t -Upper bound: [3, 5, 1]")

class TestUniformMutation(unittest.TestCase):

    @patch('numpy.random.randint')
    @patch('numpy.random.uniform')
    def test_mutation_function_array_bound(self, random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,-4.23,0.01]
        random_int_number.side_effect = [0,2,1]
        result = uniform_mutationArray(
            POPULATION,
            [-3,-5,-1],
            [3,5,1]
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [0.67,2,3],
            [4,5,-4.23],
            [7,0.01,9]
        ]
        for i in range(rows):
            self.assertListEqual(list(result[i]), valid_result[i])

    @patch('numpy.random.randint')
    @patch('numpy.random.uniform')
    def test_mutation_function_number_bound(self, random_float_number, random_int_number):
        random_float_number.side_effect = [-2.78,0.23,1.51]
        random_int_number.side_effect = [0,2,1]
        result = uniform_mutation(
            POPULATION,
            -3,
            3
        )
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ -2.78,  2,  3],
            [ 4,  5, 0.23],
            [ 7,  1.51,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

    @patch('numpy.random.randint')
    @patch('numpy.random.uniform')
    def test_mutation_class_method_number(self,  random_float_number, random_int_number):
        random_float_number.side_effect = [-2.78,0.23,1.51]
        random_int_number.side_effect = [0,2,1]

        #It should return a matrix with individuals mutated when the bound is a list.      
        method = UniformMutator([-3,3])
        result = method(POPULATION)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [ -2.78,  2,  3],
            [ 4,  5, 0.23],
            [ 7,  1.51,  9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Uniform\n\t Arguments:\n\t\t -Lower bound: -3\n\t\t -Upper bound: 3")

    @patch('numpy.random.randint')
    @patch('numpy.random.uniform')
    def test_mutation_class_method_array(self,  random_float_number, random_int_number):
        random_float_number.side_effect = [0.67,0.23,0.51]
        random_int_number.side_effect = [0,2,1]  

        #It should return a matrix with individuals mutated when the bound is a list.      
        method = UniformMutator(
            [[-3,-5,-1],
            [3,5,1]]
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
            [0.67,2,3],
            [4,5,0.23],
            [7,0.51,9]
        ]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Uniform\n\t Arguments:\n\t\t -Lower bound: [-3, -5, -1]\n\t\t -Upper bound: [3, 5, 1]")

class TestNoneUniformMutation(unittest.TestCase):

    @patch('numpy.random.normal')
    def test_mutation_function(self, random_segment):
        random_segment.side_effect =  [
            numpy.array([0.12,0.43,0.67]),
            numpy.array([0.76,0.23,0.79]),
            numpy.array([0.12,0.89,0.45])
        ]

        result = none_uniform_mutation(
            POPULATION,
            sigma=1.0
        )
        print(result)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [0.12, 0.86, 2.01],
            [3.04, 1.15, 4.74],
            [0.84, 7.12, 4.05]]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

    @patch('numpy.random.normal')
    def test_mutation_class_method(self, random_segment):
        random_segment.side_effect =  [
            numpy.array([0.12,0.43,0.67]),
            numpy.array([0.76,0.23,0.79]),
            numpy.array([0.12,0.89,0.45])
        ]
        method = NoneUniformMutator(sigma=1.0)
        result = method(POPULATION)
        #It should return a numpy array.
        self.assertEqual(type(result).__module__, 'numpy')

        #It should be the same size as the original population.
        rows,cols = result.shape
        self.assertEqual(rows, 3)
        self.assertEqual(cols, 3)

        #It should return this result.
        valid_result = [
            [0.12, 0.86, 2.01],
            [3.04, 1.15, 4.74],
            [0.84, 7.12, 4.05]]
        for i in range(rows):
            self.assertListEqual(transform(list(result[i])), transform(valid_result[i]))

        #It should save the method type in the doc string.
        self.assertEqual(method.__doc__, "Non Uniform\n\t Arguments:\n\t\t -Sigma: 1.0")

if __name__ == '__main__':
    unittest.main(verbosity=3)
