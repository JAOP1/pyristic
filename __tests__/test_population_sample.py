import unittest
from utils.operators.population_sample import *

class TestPopulationSampler(unittest.TestCase):

    def test_random_uniform_population(self):
        #It should return a matrix 2x2 when the bounds are a uni-dimensional array.
        sampler = RandomUniformPopulation(2,[0,1])
        result = sampler(2)
        self.assertEqual(type(result).__module__, 'numpy')
        self.assertEqual((2,2), result.shape)
        for i in range(2):
            for j in range(2):
                self.assertTrue(0<= result[i][j] <= 1)

        #It should return a matrix 1x2 when the bounds are 2 arrays.
        sampler = RandomUniformPopulation(3,[[0,2,4],[1,2,5]])
        result = sampler(1)
        self.assertEqual(type(result).__module__, 'numpy')
        self.assertEqual(3, result.size)
        self.assertTrue(0<= result[0][0] <= 1)
        self.assertTrue(2<= result[0][1] <= 2)
        self.assertTrue(4<= result[0][2] <= 5)

    def test_random_permutation_population(self):
        #It should return a matrix where every row is a permutation of [0,...,n]
        #when the input is a integer.
        sampler = RandomPermutationPopulation(3)
        result = sampler(2)
        self.assertEqual((2,3), result.shape)
        for row in result:
            row.sort()
            self.assertTrue(list(row) == [0,1,2])

        #It shoud return a matrix where every row is a permutation of the list passed as
        # argument.
        sampler = RandomPermutationPopulation([3,4,5])
        result = sampler(2)
        self.assertEqual((2,3), result.shape)
        for row in result:
            row.sort()
            self.assertTrue(list(row) == [3,4,5])
     
if __name__ == '__main__':
    unittest.main(verbosity=3)