import unittest
from utils.evolutionary_config import *
import numpy as np

class MockOperator:
    """
    class that simulate a operator of pyristic.
    """
    def __init__(self,operator_type):
        self.__doc__ = f"mock {operator_type} operator"

    def __call__(self, individual: np.ndarray) -> np.ndarray:
        return individual

class TestOptimizerConfig(unittest.TestCase):
    """
    Suite for OptimizerConfig.
    """
    def test_initialize_optimizer_config_class(self):
        """
        Test the methods of OptimizerConfig.
        """
        #It should keep two variables when the user declares optimizerConfig.
        config = OptimizerConfig()

        self.assertEqual(type(config.methods), dict)
        self.assertEqual(len(config.methods.keys()), 0)


    def test_attach_operators(self):
        """
        Test if a method is append.
        """
        config = OptimizerConfig()
        header_string =\
            "--------------------------------\n\tConfiguration\n--------------------------------\n"
        #It should return the string that display when the developer use print.
        result_string = str(config)
        self.assertEqual(result_string,header_string)

        operators = [
            ('crossover_operator', config.cross),
            ('mutation_operator', config.mutate),
            ('survivor_selector', config.survivor_selection),
            ('setter_invalid_solution', config.fixer_invalide_solutions)
        ]
        #It should attach the operator.
        for key_name, config_function in operators:
            operator = MockOperator(key_name)
            config_returned = config_function(operator)
            header_string += f"{key_name} - {operator.__doc__}\n"
            self.assertIn(key_name, list(config.methods.keys()))
            self.assertIsInstance(config_returned, OptimizerConfig)
            self.assertEqual(
                str(config),
                header_string)

class TestGeneticConfig(unittest.TestCase):
    """
    Suite for genetic config.
    """
    def test_initialize_optimizer_config_class(self):
        """
        Test the methods of OptimizerConfig.
        """
        #It should keep two variables when the user declares optimizerConfig.
        config = GeneticConfig()

        self.assertEqual(type(config.methods), dict)
        self.assertEqual(len(config.methods.keys()), 0)

    def test_attach_operators(self):
        """
        Test if a method is append.
        """
        config = GeneticConfig()
        header_string =\
            "--------------------------------\n\tConfiguration\n--------------------------------\n"
        #It should return the string that display when the developer use print.
        result_string = str(config)
        self.assertEqual(result_string,header_string)

        operators = [
            ('parent_selector', config.parent_selection),
        ]
        #It should attach the operator.
        for key_name, config_function in operators:
            operator = MockOperator(key_name)
            config_returned = config_function(operator)
            header_string += f"{key_name} - {operator.__doc__}\n"
            self.assertIn(key_name, list(config.methods.keys()))
            self.assertIsInstance(config_returned, OptimizerConfig)
            self.assertEqual(
                str(config),
                header_string)

class TestEvolutionStrategyConfig(unittest.TestCase):
    """
    Suite for genetic config.
    """
    def test_initialize_optimizer_config_class(self):
        """
        Test the methods of OptimizerConfig.
        """
        #It should keep two variables when the user declares optimizerConfig.
        config = EvolutionStrategyConfig()

        self.assertEqual(type(config.methods), dict)
        self.assertEqual(len(config.methods.keys()), 0)


    def test_attach_operators(self):
        """
        Test if a method is append.
        """
        config = EvolutionStrategyConfig()
        header_string =\
            "--------------------------------\n\tConfiguration\n--------------------------------\n"
        #It should return the string that display when the developer use print.
        result_string = str(config)
        self.assertEqual(result_string,header_string)

        operators = [
            ('adaptive_crossover_operator', config.adaptive_crossover),
            ('adaptive_mutation_operator', config.adaptive_mutation)
        ]
        #It should attach the operator.
        for key_name, config_function in operators:
            operator = MockOperator(key_name)
            config_returned = config_function(operator)
            header_string += f"{key_name} - {operator.__doc__}\n"
            self.assertIn(key_name, list(config.methods.keys()))
            self.assertIsInstance(config_returned, OptimizerConfig)
            self.assertEqual(
                str(config),
                header_string)

class TestEvolutionaryProgrammingConfig(unittest.TestCase):
    """
    Suite for genetic config.
    """
    def test_initialize_optimizer_config_class(self):
        """
        Test the methods of OptimizerConfig.
        """
        #It should keep two variables when the user declares optimizerConfig.
        config = EvolutionaryProgrammingConfig()

        self.assertEqual(type(config.methods), dict)
        self.assertEqual(len(config.methods.keys()), 0)


    def test_attach_operators(self):
        """
        Test if a method is append.
        """
        config = EvolutionaryProgrammingConfig()
        header_string =\
            "--------------------------------\n\tConfiguration\n--------------------------------\n"
        #It should return the string that display when the developer use print.
        result_string = str(config)
        self.assertEqual(result_string,header_string)

        operators = [
            ('adaptive_mutation_operator', config.adaptive_mutation),
        ]
        #It should attach the operator.
        for key_name, config_function in operators:
            operator = MockOperator(key_name)
            config_returned = config_function(operator)
            header_string += f"{key_name} - {operator.__doc__}\n"
            self.assertIn(key_name, list(config.methods.keys()))
            self.assertIsInstance(config_returned, OptimizerConfig)
            self.assertEqual(
                str(config),
                header_string)

if __name__ == '__main__':
    unittest.main(verbosity=3)
