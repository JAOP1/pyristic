from copy import deepcopy
import numpy as np

__all__ = [
    "ProporcionalSampler",
    "RouletteSampler",
    "StochasticUniversalSampler",
    "DeterministicSampler",
    "TournamentSampler",
    "MergeSelector",
    "ReplacementSelector",
    "proporcional_sampling",
    "roulette_sampling",
    "stochastic_universal_sampling",
    "deterministic_sampling",
    "tournament_sampling",
    "get_expected_values",
    "get_candidates_by_aptitude",
]


def get_expected_values(aptitude: np.array) -> np.array:
    """
    Description:
        return the expected number of copies of every individual.
    """
    average_aptitude = np.sum(aptitude)
    number_individuals = len(aptitude)
    return aptitude / average_aptitude * number_individuals


"""
---------------------------------------------------------------------------------
                            |   Genetic Algorithms.   |
---------------------------------------------------------------------------------
"""


"""
---------------------------------------------------------------------------------
                                Parent selection.
---------------------------------------------------------------------------------
"""


def proporcional_sampling(expected_vals: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        return an array with the indices from 0 up to n, where n
        is the number the elements of the input array.
    Arguments:
        probabilities: 1-D numpy array which is the probability of every
        individual based on the aptitude.
    About:
        If you want more information, check:

    ------------------------------------------------------
    """
    return np.arange(len(expected_vals))


def roulette_sampling(expected_vals: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        selection algorithm O(nlogn) for genetic algorithm.
    Arguments:
        probabilities: 1-D numpy array which is the probability of every
        individual based on the aptitude.
    About:
        If you want more information, check:

    ------------------------------------------------------
    """
    expected_vals = get_expected_values(expected_vals)
    number_individuals = len(expected_vals)

    expected_cumulative_sum = np.cumsum(expected_vals)
    random_number = np.random.uniform(
        0.0, expected_cumulative_sum[-1], number_individuals
    )
    return np.searchsorted(expected_cumulative_sum, random_number)  # sample


def stochastic_universal_sampling(expected_vals: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Arguments:
        probabilities: 1-D numpy array which is the probability of every
        individual based on the aptitude.
    About:
        If you want more information, check:

    ------------------------------------------------------
    """
    expected_vals = get_expected_values(expected_vals)

    number_individuals = len(expected_vals)

    random_number = np.random.uniform(0, 1)
    current_sum = 0
    sample = []
    for i in range(number_individuals):
        current_sum += expected_vals[i]
        while random_number < current_sum:
            sample.append(i)
            random_number += 1
    return np.array(sample)


def deterministic_sampling(expected_vals: np.ndarray) -> np.ndarray:
    """
    ------------------------------------------------------
    Arguments:
        probabilities: 1-D numpy array which is the probability of every
        individual based on the aptitude.
    About:
        If you want more information, check:
    ------------------------------------------------------
    """
    expected_vals = get_expected_values(expected_vals)

    number_individuals = len(expected_vals)

    integer_part = np.array(expected_vals, dtype=int)
    indices = np.arange(number_individuals)
    sample = np.repeat(indices, integer_part)

    if len(sample) < number_individuals:
        float_part = expected_vals - integer_part
        ind = np.argpartition(float_part, len(sample) - number_individuals)
        ind = ind[len(sample) - number_individuals :]
        sample = np.concatenate((sample, indices[ind]))

    return sample[:number_individuals]


def tournament_sampling(
    expected_vals: np.ndarray, chunks: int = 2, prob: float = 1.0
) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        selection algorithm O(NP) for genetic algorithm where N is the population
        size and P is the number of chunks.
    Arguments:
        probabilities: 1-D numpy array which is the probability of every
        individual based on the aptitude.
        chunks: Integer value which is the number of chunks (by default is 2).
        prob: Float value which means if choose the max or min value in every chunk
        (by default is 1).
        About:
            If you want more information, check:

    ------------------------------------------------------
    """
    chunks = int(chunks)
    expected_vals = get_expected_values(expected_vals)
    num_items = len(expected_vals)
    indices = np.arange(num_items)
    chunks_ = np.ceil(num_items / chunks)

    sample = []
    ind = -1
    for tournament in range(chunks):
        groups = np.array_split(np.random.permutation(indices), chunks_)
        for group in groups:
            if prob >= np.random.rand():
                ind = np.argmax(expected_vals[group])
            else:
                ind = np.argmin(expected_vals[group])
            sample.append(group[ind])

    return np.array(sample)


"""
---------------------------------------------------------------------------------
                                Survivor selection.
---------------------------------------------------------------------------------
"""


def get_candidates_by_aptitude(value: np.ndarray, number_items: int) -> np.ndarray:
    """
    ------------------------------------------------------
    Description:
        Return m indices with the biggest fitness.
    Arguments:
        value: float array.
        number_items: int number of indices to return with the biggest value.
    ------------------------------------------------------
    """
    pseudo_sorted_array = np.argpartition(value, -1 * number_items)
    return pseudo_sorted_array[-1 * number_items :]


"""
---------------------------------------------------------------------------------
                                Classes
---------------------------------------------------------------------------------
"""

"""
----------------------------
    Parent Selection.
----------------------------
"""


class ProporcionalSampler:
    """
    Description:
      Selection class as method based on the proporcional_sampler.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Proporcional sampling"

    def __call__(self, population_f: np.ndarray) -> np.ndarray:
        return proporcional_sampling(population_f)


class RouletteSampler:
    """
    Description:
      Selection class as method based on the roulette_sampler.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Roulette sampling"

    def __call__(self, population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        return roulette_sampling(vals)


class StochasticUniversalSampler:
    """
    Description:
      Selection class as method based on the stochastic_universal_sampler.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Stochastic universal sampling"

    def __call__(self, population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        return stochastic_universal_sampling(vals)


class DeterministicSampler:
    """
    Description:
      Selection class as method based on the deterministic_sampler.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Deterministic sampling"

    def __call__(self, population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        return deterministic_sampling(vals)


class TournamentSampler:
    """
    Description:
      Selection class as method based on the tournament_sampler.
    Arguments:
        - chunks: Integer value which is the number of chunks (by default is 2).
        - prob: Float value which means if choose the max or min value in every chunk
        (by default is 1).
    """

    def __init__(self, chunks_: int = 2, prob_: float = 1.0):
        self.chunks = chunks_
        self.prob = prob_
        self.__doc__ = f"Tournament sampling\n\t Arguments:\n\t\t-Chunks: {self.chunks}\n\t\t-prob: {self.prob}"

    def __call__(self, population_f: np.ndarray) -> np.ndarray:
        vals = np.copy(population_f)
        return tournament_sampling(vals, self.chunks, self.prob)


"""
----------------------------
    Survivor Selection.
----------------------------
"""


class MergeSelector:
    """
    Description:
      Selection survivors class as method based on the lambda + mu selection.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Merge population"

    def __call__(
        self, parent_f: np.ndarray, offspring_f: np.ndarray, features: dict
    ) -> dict:
        """
        ------------------------------------------------------
        Description:
            survivor selection algorithm, where choose a individuals between parents and
            offspring that have great f for the next generation.
        Arguments:
            parent_f: 1-D numpy array.
            offspring_f: 1-D numpy array.
            features: dictionary where every key is a list of two numpy arrays, which means
                    key- [parent population feature , offspring population feature].
                Note: This features will be saved to the next generation.
            About:
                If you want more information, check:

        ------------------------------------------------------
        """
        result = {}

        tmp_f = np.concatenate((parent_f, offspring_f))
        indices = get_candidates_by_aptitude(tmp_f, len(parent_f))
        result["parent_population_f"] = tmp_f[indices]

        for feature in features.keys():
            if len(features[feature]) != 2:
                raise Exception(
                    (
                        "Lenght of list have to be 2 "
                        "(parent population feature and offspring population feature)."
                    )
                )
            tmp = np.concatenate((features[feature][0], features[feature][1]), axis=0)
            result[feature] = deepcopy(tmp[indices])
        return result


class ReplacementSelector:
    """
    Description:
      Selection survivors class as method based on the (lambda, mu) selection.
    Arguments:
        - None
    """

    def __init__(self):
        self.__doc__ = "Replacement population"

    def __call__(
        self, parent_f: np.ndarray, offspring_f: np.ndarray, features: dict
    ) -> dict:
        """
        ------------------------------------------------------
        Description:
            survivor selection algorithm, where choose m individuals in
            offspring that have great f for the next generation.
        Arguments:
            parent_f: 1-D numpy array.
            offspring_f: 1-D numpy array.
            features: dictionary where every key is a list of two numpy arrays, which means
                    key- [parent population feature , offspring population feature].
                Note: This features will be saved to the next generation.
            About:
                If you want more information, check:

        ------------------------------------------------------
        """
        assert len(parent_f) <= len(offspring_f)

        result = {}
        indices = get_candidates_by_aptitude(offspring_f, len(parent_f))
        result["parent_population_f"] = offspring_f[indices]

        for feature in features.keys():
            if len(features[feature]) != 2:
                raise Exception(
                    (
                        "Lenght of list have to be 2 "
                        "(parent population feature and offspring population feature)."
                    )
                )
            result[feature] = deepcopy(features[feature][1][indices])

        return result
