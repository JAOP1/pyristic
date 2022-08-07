import unittest
from unittest.mock import patch
import numpy
from utils.operators.crossover import *


class TestDiscreteCrossover(unittest.TestCase):

    def test_crossover_function(self):
        population = numpy.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])
        left_ind_selected = [0,0,1]
        right_ind_selected = [1,1,2]
        with patch('numpy.random.randint') as mock_random_binary_number:
            mock_random_binary_number.side_effect = [1,0,0,1,1,1,0,1,0]
            result = discrete_cross(
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

if __name__ == '__main__':
    unittest.main(verbosity=2)