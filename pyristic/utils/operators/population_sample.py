import typing
import numpy as np

__all__ = [ 'RandomUniformPopulation', 'RandomPermutationPopulation' ]

class RandomUniformPopulation:
    """
    Description:
      Creates a matrix where every row represents an individual.
      Those individuals are created using an uniform distribution.
    Arguments:
        - num_variables: Integer that represents the number of variables
                         of the problem.
        - bounds: Uni dimensional array with two integer. The first integer
                  is the lower bound and the second one is the upper bound or
                  Array with two array, every array represent the bounds by variable.
    """
    def __init__(self,
        num_variables: int,
        bounds: typing.Union[ list[float], list[list[int]]]
    ) -> None:
        self.bounds = bounds
        self.num_variables = num_variables

    def __call__(self, num_individulas:int ) -> np.ndarray:
        return np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            size=(
                num_individulas,
                self.num_variables
            )
        )

class RandomPermutationPopulation:
    """
    Description:
      Creates a matrix where every row represents an individual.
      Those individuals are created doing a new permutation of the
      first [0,...,n] elements.
    Arguments:
        - x: int or array_like. If x is an integer, randomly permute np.arange(x).
             If x is an array, make a copy and shuffle the elements randomly.
        - bounds: Uni dimensional array with two integer. The first integer
                  is the lower bound and the second one is the upper bound or
                  Array with two array, every array represent the bounds by variable.
    """
    def __init__(self, x: int) -> None:
        self.x = x

    def __call__(self, num_individuals: int) -> np.ndarray:
        individuals = []

        for _ in range(num_individuals):
            individuals += [np.random.permutation(self.x)]

        return np.array(individuals)
