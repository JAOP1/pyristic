import numpy as np

__all__ = [
    "DiscreteCrossover",
    "IntermediateCrossover",
    "NPointCrossover",
    "UniformCrossover",
    "PermutationOrderCrossover",
    "SimulatedBinaryCrossover",
    "discrete_crossover",
    "intermediate_crossover",
    "n_point_crossover",
    "uniform_crossover",
    "permutation_order_crossover",
    "simulated_binary_crossover",
    "NoneCrossover",
]

"""
---------------------------------------------------------------------------------
                                Evolution Strategy.
---------------------------------------------------------------------------------
"""

# Crossover operators.
def discrete_crossover(
    population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two elements.
        X_{i,s} if v == 1
        X_{i,t} if v == 0
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
    ------------------------------------------------------
    """

    rows, cols = population.shape
    rows_new = len(parent_ind1)
    components_selected_fist_parent = np.zeros((rows_new, cols))
    components_selected_second_parent = np.zeros((rows_new, cols))

    for i_row in range(rows_new):
        for j_col in range(cols):
            if np.random.randint(2):
                components_selected_fist_parent[i_row, j_col] = 1
            else:
                components_selected_second_parent[i_row, j_col] = 1

    return (
        population[parent_ind1] * components_selected_fist_parent
        + population[parent_ind2] * components_selected_second_parent
    )


def intermediate_crossover(
    population: np.ndarray,
    parent_ind1: np.ndarray,
    parent_ind2: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining  the average of two elements.
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
    ------------------------------------------------------
    """
    return alpha * population[parent_ind1] + (1 - alpha) * population[parent_ind2]


"""
---------------------------------------------------------------------------------
                                Genetic Algorithm.
---------------------------------------------------------------------------------
"""


def n_point_crossover(
    population: np.ndarray,
    parent_ind1: np.ndarray,
    parent_ind2: np.ndarray,
    n_cross: int = 1,
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        -n_cross: integer value which says you the number of cuts (by default is 1).

        Note:
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """
    n_cross = int(n_cross)
    num_individuals = len(parent_ind1)
    decision_variables = len(population[0])
    new_population = np.ones(
        (num_individuals * 2, decision_variables), dtype=np.float64
    )

    for i in range(0, num_individuals * 2):
        if i % 2 == 0:
            new_population[i] = population[parent_ind1[(i // 2)]]
        else:
            new_population[i] = population[parent_ind2[(i // 2)]]

    for row in range(num_individuals):
        cross_points = np.random.choice(decision_variables - 1, n_cross, replace=False)
        cross_points += 1
        cross_points.sort()
        for i in range(0, n_cross, 2):
            start = cross_points[i]
            final = decision_variables
            if i + 1 < n_cross:
                final = cross_points[i + 1]
            new_population[2 * row][start:final] = population[parent_ind2[row]][
                start:final
            ]
            new_population[2 * row + 1][start:final] = population[parent_ind1[row]][
                start:final
            ]

    return new_population


def uniform_crossover(
    population: np.ndarray,
    parent_ind1: np.ndarray,
    parent_ind2: np.ndarray,
    flip_prob: int = 0.5,
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        -flip_prob: float value which says the likely to change a position (by default is 0.5).
        Note:
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """

    num_individuals = len(parent_ind1)
    decision_variables = len(population[0])
    new_population = np.ones(
        (num_individuals * 2, decision_variables), dtype=np.float64
    )

    for i in range(0, num_individuals * 2):
        if i % 2 == 0:
            new_population[i] = population[parent_ind1[(i // 2)]]
        else:
            new_population[i] = population[parent_ind2[(i // 2)]]

    for i in range(num_individuals):
        for decision_var in range(decision_variables):
            if np.random.rand() <= flip_prob:
                new_population[2 * i][decision_var] = population[parent_ind2[i]][
                    decision_var
                ]
                new_population[2 * i + 1][decision_var] = population[parent_ind1[i]][
                    decision_var
                ]

    return new_population


def permutation_order_crossover(
    population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        Note:
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """
    num_individuals = len(parent_ind1)
    decision_variables = len(population[0])
    new_population = np.ones((num_individuals * 2, decision_variables))

    def create_child(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:

        interval = np.random.choice(decision_variables + 1, 2, replace=False)
        interval.sort()

        individual = np.full(decision_variables, np.inf)
        segment = parent1[interval[0] : interval[1]]
        individual[interval[0] : interval[1]] = segment
        remainder = []
        for component in parent2:
            if component not in segment:
                remainder.append(component)

        individual[individual == np.inf] = remainder
        return individual

    for i in range(num_individuals):
        new_population[2 * i] = create_child(
            population[parent_ind1[i]], population[parent_ind2[i]]
        )
        new_population[2 * i + 1] = create_child(
            population[parent_ind2[i]], population[parent_ind1[i]]
        )

    return new_population


def simulated_binary_crossover(
    population: np.ndarray,
    parent_ind1: np.ndarray,
    parent_ind2: np.ndarray,
    nc: int = 1,
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Function which create new elements combining two individuals.
    Arguments:
        -population: Matrix with size m x n, where m is the current size of
        population and n is the problem variables. Every row is an element.
        -parent_ind1: the chosen parents, where every position is the index in X.
        -parent_ind2: the chosen parents, where every position is the index in X.
        -nc: integer number. Default is 1.
        Note:
            parent_ind1 and parent_ind2 should be m elements.
    Return:
        Matriz with size 2m x n.
    ------------------------------------------------------
    """
    nc = int(nc)
    random_float_number = np.random.rand()
    B = -1
    exponent = 1 / (nc + 1)
    if random_float_number <= 0.5:
        B = np.power((2 * random_float_number), exponent)
    else:
        B = np.power(1 / (2 * (1 - random_float_number)), exponent)

    first_parent_population = population[parent_ind1]
    second_parent_population = population[parent_ind2]

    A = first_parent_population + second_parent_population
    B = np.absolute(second_parent_population - first_parent_population) * B

    return np.concatenate((0.5 * (A - B), 0.5 * (A + B)), axis=0)


class IntermediateCrossover:
    """
    Description:
      Class crossover operator based on intermediate_crossover.
    Arguments:
        - alpha: Float number (default = 0.5).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.__doc__ = f"Intermediate\n\tArguments:\n\t\t-Alpha:{self.alpha}"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return intermediate_crossover(population, parent_ind1, parent_ind2, self.alpha)


class NPointCrossover:
    """
    Description:
      Class crossover operator based on the 1 point crossover. This is a
      generic crossover operator for n partitions.
    Arguments:
        - n: Integer number (default = 1).
    """

    def __init__(self, n_cross: int = 1):
        self.n_cross = n_cross
        self.__doc__ = f"n point\n\tArguments:\n\t\t-n_cross: {self.n_cross}"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)

        return n_point_crossover(population, parent_ind1, parent_ind2, self.n_cross)


class UniformCrossover:
    """
    Description:
      Class crossover operator based on the uniform crossover.
    Arguments:
        - flip_prob: float number between [0,1] (default = 0.5).
    """

    def __init__(self, flip_prob: float = 0.5):
        self.prob = flip_prob
        self.__doc__ = f"Uniform\n\tArguments:\n\t\t-prob: {self.prob}"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return uniform_crossover(population, parent_ind1, parent_ind2, self.prob)


class PermutationOrderCrossover:
    """
    Description:
      Class crossover operator based on the permutation order crossover.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Permutation order"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return permutation_order_crossover(population, parent_ind1, parent_ind2)


class SimulatedBinaryCrossover:
    """
    Description:
      Class crossover operator based on the simulated binary crossover.
    Arguments:
        - nc: Integer number (default = 1).
    """

    def __init__(self, nc: int = 1):
        self.nc = nc
        self.__doc__ = f"Simulated binary\n\tArguments:\n\t\t-nc: {self.nc}"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)

        return simulated_binary_crossover(population, parent_ind1, parent_ind2, self.nc)


class NoneCrossover:
    """
    Description:
      Class crossover operator to avoid include crossover operator.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return population


# Crossover operator for ES.
class DiscreteCrossover:
    """
    Description:
      Class crossover operator based on the discrete crossover.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Discrete"

    def __call__(
        self, population: np.ndarray, parent_ind1: np.ndarray, parent_ind2: np.ndarray
    ) -> np.ndarray:
        assert len(parent_ind1) == len(parent_ind2)
        return discrete_crossover(population, parent_ind1, parent_ind2)
