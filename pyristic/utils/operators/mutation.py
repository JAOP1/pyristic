import numpy as np

__all__ = [
    "InsertionMutator",
    "ExchangeMutator",
    "BoundaryMutator",
    "UniformMutator",
    "NoneUniformMutator",
    "BinaryMutator",
    "NoneMutator",
    "SigmaMutator",
    "MultSigmaAdaptiveMutator",
    "SingleSigmaAdaptiveMutator",
    "SigmaEpAdaptiveMutator",
    "sigma_ep_adaptive",
    "single_sigma_adaptive",
    "mult_sigma_adaptive",
    "mutation_by_sigma",
    "insertion_mutation",
    "insertion_mutation",
    "exchange_mutation",
    "boundary_mutation",
    "uniform_mutation",
    "none_uniform_mutation",
    "binary_mutation",
    "boundary_mutation_array",
    "uniform_mutation_array",
]


def sigma_ep_adaptive(population: np.ndarray, alpha: float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Mutation operator for Evolutionary programming.
    ------------------------------------------------------
    """
    return population * (1.0 + alpha * np.random.normal(0, 1, size=population.shape))


def single_sigma_adaptive(population: np.ndarray, gamma: float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Every individual has a sigma parameter which helps you to
        update in a fixed portion the individual.

        Note:
            This function is for Evolution Strategy search.
    Arguments:
        - X: Numpy array where every N-th row is an individual and
          M columns that are the decision variables.
        - gamma: Float value, additional parameter.

    ------------------------------------------------------
    """
    exponent = np.random.normal(0, 1) * gamma
    return np.exp(exponent) * population


def mult_sigma_adaptive(
    population: np.ndarray, gamma: float, gamma_prime: float
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Every individual has a sigma for every decision variable which helps you to
        update in a fixed portion the individual.

        Note:
            This function is for Evolution Strategy search.
    Arguments:
        - X: Numpy array where every N-th row is an individual and
          M columns that are the decision variables.
        - gamma: Float value, additional parameter.
        - gamma_prime: Float value, additional parameter.
    ------------------------------------------------------
    """
    firts_ = np.random.normal(0, 1) * gamma_prime
    second_ = np.random.normal(0, 1, size=population.shape) * gamma

    exponent = firts_ + second_
    return population * np.exp(exponent)


def mutation_by_sigma(population: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Update function in the decision variables of every individual.

        Note:
            This function is for Evolution Strategy search.
    Arguments:
        - X: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - sigma: Numpy Array with N rows which means every row is a individual, M columns differ in
        what kind of sigma has chosen (single or multiple).
    ------------------------------------------------------
    """
    normal_value = np.random.normal(0, 1)
    return population + sigma * normal_value


def insertion_mutation(population: np.ndarray, n_elements: int = 1) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Discrete mutation operator. It takes a segment and the remaining elements
        are introduced.

    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - n_elements: Integer number (default 1). It indicates the number of
          elements in the segment.
    ------------------------------------------------------
    """
    n_elements = int(n_elements)
    num_individuals, decision_variables = population.shape
    x_mutated = np.ones((num_individuals, decision_variables))

    for ind in range(num_individuals):
        individual = np.full(decision_variables, np.inf)
        indices_elements = np.random.choice(
            decision_variables, n_elements, replace=False
        )
        elements = population[ind][indices_elements]
        indices_position = np.random.choice(
            decision_variables, n_elements, replace=False
        )
        individual[indices_position] = elements

        indices_elements.sort()
        remaining = []
        start_ = 0
        for decision_var in range(decision_variables):
            if (
                start_ < len(indices_elements)
                and decision_var == indices_elements[start_]
            ):
                start_ += 1
                continue
            remaining.append(population[ind][decision_var])

        individual[individual == np.inf] = remaining
        x_mutated[ind] = individual

    return x_mutated


def exchange_mutation(population: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Discrete mutation operator. It takes two random positions
        and swap the values.

    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    tmp_x = np.copy(population)
    for ind in range(num_individuals):
        exchange_points = np.random.choice(decision_variables, 2, replace=False)
        first_decision_var = exchange_points[0]
        second_decision_var = exchange_points[1]

        tmp_point = tmp_x[ind][first_decision_var]
        tmp_x[ind][first_decision_var] = tmp_x[ind][second_decision_var]
        tmp_x[ind][second_decision_var] = tmp_point

    return tmp_x


def binary_mutation(population: np.ndarray, prob_mutation: float) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Binary mutation operator. This operator flip every position
        to 0 when the value is 1 or viceversa with a probability.

    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - prob_mutation: float number. It indicates the probability of flip.
    ------------------------------------------------------
    """
    assert 0 <= prob_mutation <= 1.0
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for i_individual in range(num_individuals):
        for i_variable in range(decision_variables):
            if np.random.rand() < prob_mutation:
                num = 0
                if mutated_x[i_individual][i_variable] == 0:
                    num = 1
                mutated_x[i_individual][i_variable] = num

    return mutated_x


def boundary_mutation_array(
    population: np.ndarray, lower_bounds: list, upper_bounds: list
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        continuous mutation operator. This operator select a
        random position of the individual and change the value to
        one of the boundaries.

        Note: this function is when every decision variable  has
        a different search space.
    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - lower_bounds: list of float numbers. The lower bound for the i-th
            decision variable.
        - upper_bounds: list of float numbers. The upper bound for the i-th
            decision variable.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for ind in range(num_individuals):
        variable = np.random.randint(0, decision_variables)
        lower_bound = lower_bounds[ind]
        upper_bound = upper_bounds[ind]
        mutated_x[ind][variable] = lower_bound
        if np.random.rand() > 0.5:
            mutated_x[ind][variable] = upper_bound
    return mutated_x


def boundary_mutation(
    population: np.ndarray, lower_bound: float, upper_bound: float
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        continuous mutation operator. This operator select a
        random position of the individual and change the value to
        one of the boundaries.

        Note: this function is when every decision variable  has
        the same search space.
    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - lower_bound: float number. The lower bound for the i-th
            decision variable.
        - upper_bound: float number. The upper bound for the i-th
            decision variable.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for ind in range(num_individuals):
        variable = np.random.randint(0, decision_variables)
        mutated_x[ind][variable] = lower_bound
        if np.random.rand() > 0.5:
            mutated_x[ind][variable] = upper_bound
    return mutated_x


def uniform_mutation_array(
    population: np.ndarray, lower_bounds: list, upper_bounds: list
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        continuous mutation operator. This operator select a
        random position of the individual and change the value by
        a random value between the boundaries for that i-th position.

        Note: this function is when every decision variable  has
        different search space.
    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - lower_bounds: array of float numbers. The lower bound for the i-th
            decision variable.
        - upper_bounds: array of float numbers. The upper bound for the i-th
            decision variable.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for ind in range(num_individuals):
        variable = np.random.randint(0, decision_variables)
        lower_bound = lower_bounds[ind]
        upper_bound = upper_bounds[ind]
        mutated_x[ind][variable] = np.random.uniform(lower_bound, upper_bound)
    return mutated_x


def uniform_mutation(
    population: np.ndarray, lower_bound: float, upper_bound: float
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        continuous mutation operator. This operator select a
        random position of the individual and change the value by
        a random value between the boundaries for that i-th position.

        Note: this function is when every decision variable  has
        the same search space.
    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - lower_bound: float number. The lower bound for the i-th
            decision variable.
        - upper_bound: float number. The upper bound for the i-th
            decision variable.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for ind in range(num_individuals):
        variable = np.random.randint(0, decision_variables)
        mutated_x[ind][variable] = np.random.uniform(lower_bound, upper_bound)
    return mutated_x


def none_uniform_mutation(population: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        continuous mutation operator. This operator applies
        a random noise with a normal distribution between 0
        up to sigma for every decision variable.

    Arguments:
        - population: Numpy Matrix, where every N-th row is an individual and
          M columns that are the decision variables.
        - sigma: float number. it is the size of numbers acepted as
            noise.
    ------------------------------------------------------
    """
    num_individuals, decision_variables = population.shape
    mutated_x = np.copy(population)
    for ind in range(num_individuals):
        noise = np.random.normal(0, sigma, size=decision_variables)
        mutated_x[ind] *= noise

    return mutated_x


class InsertionMutator:
    """
    Description:
      Class mutation operator based on insertion_mutation.
    Arguments:
        - n_elements:
    """

    def __init__(self, n_elements: int = 1):
        self.n_elements = n_elements
        self.__doc__ = f"Insertion \n\t Arguments:\n\t\t -n_elements: {n_elements}"

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return insertion_mutation(population, self.n_elements)


class ExchangeMutator:
    """
    Description:
      Class mutation operator based on exchange_mutation.
    Arguments:
        This method doesn't need arguments.
    """

    def __init__(self):
        self.__doc__ = "Exchange"

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return exchange_mutation(population)


class BoundaryMutator:
    """
    Description:
      Class mutation operator based on boundary_mutation.
    Arguments:
        - bounds:
    """

    def __init__(self, bounds: list):
        self.bounds = bounds
        self.type_bound = list if isinstance(self.bounds[0], list) else float
        self.__doc__ = (
            "Boundary\n\t Arguments:\n\t\t"
            f" -Lower bound: {self.bounds[0]}\n\t\t -Upper bound: {self.bounds[1]}"
        )

    def __call__(self, population: np.ndarray) -> np.ndarray:
        if self.type_bound != list:
            return boundary_mutation(population, self.bounds[0], self.bounds[1])

        return boundary_mutation_array(population, self.bounds[0], self.bounds[1])


class UniformMutator:
    """
    Description:
      Class mutation operator based on unfirom_mutation.
    Arguments:
        - bounds:
    """

    def __init__(self, bounds: list):
        self.bounds = bounds
        self.type_bound = list if isinstance(self.bounds[0], list) else float
        self.__doc__ = (
            "Uniform\n\t Arguments:\n\t\t"
            f" -Lower bound: {self.bounds[0]}\n\t\t -Upper bound: {self.bounds[1]}"
        )

    def __call__(self, population: np.ndarray) -> np.ndarray:

        if self.type_bound != list:
            return uniform_mutation(population, self.bounds[0], self.bounds[1])

        return uniform_mutation_array(population, self.bounds[0], self.bounds[1])


class NoneUniformMutator:
    """
    Description:
      Class mutation operator based on none_uniform_mutation.
    Arguments:
        - sigma:
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.__doc__ = f"Non Uniform\n\t Arguments:\n\t\t -Sigma: {self.sigma}"

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return none_uniform_mutation(population, self.sigma)


class NoneMutator:
    """
    Description:
      Class mutation operator based on none_mutation.
    Arguments:
        This methos doesn't need arguments.
    """

    def __init__(self):
        self.__doc__ = "None"

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return population


class BinaryMutator:
    """
    Description:
      Class mutation operator based on binary_mutation.
    Arguments:
        - prob_mutation: float number (default 0.2).
    """

    def __init__(self, prob_mutation: float = 0.2):
        self.__doc__ = (
            "Binary mutation \n\t Arguments:\n\t\t "
            f"- probability to flip: {prob_mutation}"
        )
        self.prob_mutation = prob_mutation

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return binary_mutation(population, self.prob_mutation)


# Mutatio operator for EP.
class SigmaEpAdaptiveMutator:
    """
    Description:
      Class mutation operator based on sigma_ep_adaptive.
    Arguments:
        - decision_variables: integer number. It describe
            the number of decision variables.
        - alpha: float number. It means how much impact
            the noise applied to the individual.
    """

    def __init__(self, decision_variables: int, alpha: float):
        self._alpha = alpha
        self._length = int(decision_variables)
        self.__doc__ = f"Sigma EP.\n\t\t-Alpha: {self._alpha}"

    @property
    def length(self):
        """
        Return the number of decision variables.
        """
        return self._length

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return sigma_ep_adaptive(population, self._alpha)


# Mutation operator for ES.
class SigmaMutator:
    """
    Description:
      Class mutation operator based on mutation_by_sigma.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Sigma"

    def __call__(self, population: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return mutation_by_sigma(population, sigma)


class MultSigmaAdaptiveMutator:
    """
    Description:
      Class mutation operator based on mult_sigma_adaptive.
    Arguments:
        - None
    """

    def __init__(self, decision_variables: int):
        self.__doc__ = "Sigma mult"
        self._length = int(decision_variables)
        self._gamma_prime = 1 / np.sqrt(2 * self._length)
        self._gamma = 1 / np.sqrt(2 * np.sqrt(self._length))

    @property
    def length(self):
        """
        Return the number of decision variables.
        """
        return self._length

    def __call__(self, sigma: np.ndarray) -> np.ndarray:
        return mult_sigma_adaptive(sigma, self._gamma, self._gamma_prime)


class SingleSigmaAdaptiveMutator:
    """
    Description:
      Class mutation operator based on single_sigma_adaptive.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Single Sigma"
        self._length = 1
        self._tau = 1 / np.sqrt(self._length)

    @property
    def length(self):
        """
        Return the number of decision variables.
        """
        return self._length

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return single_sigma_adaptive(population, self._tau)
