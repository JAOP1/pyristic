from pyristic.utils.operators import selection, mutation, crossover
from pyristic.utils.helpers import function_type, ContinuosFixer, EvolutionStrategyConfig
from tqdm import tqdm
import numpy as np

__all__= ['EvolutionStrategy']


class EvolutionStrategy:
    def __init__(self,  function: function_type,\
                        decision_variables:int,\
                        constraints:list=[],\
                        bounds: list=[],\
                        config = EvolutionStrategyConfig()):
        """
        ------------------------------------------------------
        Description:
            Initializing every variable necessary to the search.
        Arguments: 
            - Function: Objective function to minimize.
            - Constraints: Constraints to be a feasible solution.
            - Bounds: bound for every variable, this should be a matrix 2 x N
            where N is the variables number. The first element is lowert limit 
            and another one is the upper limit. 
        ------------------------------------------------------ 
        """

        self.f = function
        self.Constraints = constraints
        self.Bounds = bounds
        self.Decision_variables = decision_variables
        
        #Configuration.
        self._mutation_operator           = config.mutation_op  if config.mutation_op != None else mutation.sigma_mutator()
        self._crossover_operator          = config.cross_op if config.cross_op != None else crossover.discrete_crossover()
        self._survivor_selector           = config.survivor_selector if config.survivor_selector != None else selection.merge_selector()
        self._fixer                       = config.fixer if config.fixer != None else  ContinuosFixer(self.Bounds)
        self._adaptive_crossover_operator = config.adaptive_crossover_op if config.adaptive_crossover_op != None else crossover.intermediate_crossover()
        self._adaptive_mutation_operator  = config.adaptive_mutation_op  if config.adaptive_mutation_op != None else mutation.mult_sigma_adaptive_mutator(self.Decision_variables)

        #Search information.
        self.logger = {}
        self.logger['best_individual']      = None
        self.logger['best_f']        = None
        self.logger['current_iter']    = None
        self.logger['total_iter']      = None

    def __str__(self):
        printable = "Evolution Strategy search: \n f(X) = {} \n X = {} \n ".format(self.logger['best_f'], self.logger['best_individual'])
        first = True
        
        for i in range(len(self.Constraints)):
            if self.Constraints[i].__doc__ != None:
                
                if first:
                    first = False
                    printable += "Constraints: \n "
                    
                self.Constraints[i](self.logger['best_individual'])
                printable += "{} \n".format( self.Constraints[i].__doc__)
                    
        return printable

    def optimize(self,  generations:     int ,\
                        population_size: int,\
                        offspring_size:  int,\
                        eps_sigma:       float=0.001,\
                        verbose=True,\
                        **kwargs) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using tabu search.
        Arguments:
            - generations: integer that represent the total iterations.
            - population_size: the population that 
        ------------------------------------------------------
        """
        #Reset global solution.
        self.logger['best_individual']      = None
        self.logger['best_f']        = None
        self.logger['current_iter']    = 0
        self.logger['total_iter']      = generations
        self.logger['parent_population_size'] = population_size
        self.logger['offspring_population_size']  = offspring_size       
        #offspring_size//=2
        self.logger['parent_population_x']= self.initialize_population(**kwargs)
        self.logger['parent_population_sigma'] = self.initialize_step_weights(eps_sigma,**kwargs)
        self.logger['parent_population_f'] = np.apply_along_axis( self.f ,\
                                                                        1,\
                                                                        self.logger['parent_population_x'])   

        try:
            for g in tqdm(range(generations), disable = not verbose):
                   

                #Crossover.
                first_parent_indices, second_parent_indices = self.get_pairs(**kwargs)
                self.logger['offspring_population_x']       = self.crossover_operator(  first_parent_indices,\
                                                                                        second_parent_indices,\
                                                                                        **kwargs)

                self.logger['offspring_population_sigma'] = self.adaptive_crossover(    first_parent_indices,\
                                                                                        second_parent_indices,\
                                                                                        **kwargs)
                #mutate.
                self.logger['offspring_population_sigma'] = self.adaptive_mutation(**kwargs)
                self.logger['offspring_population_x']     = self.mutation_operator(**kwargs)
                
                #Fixing solutions and getting aptitude.
                f_offspring = []
                for i in range(len(self.logger['offspring_population_x'])):
                    if self.is_invalid(self.logger['offspring_population_x'][i]):
                        self.logger['offspring_population_x'][i] = self.fixer(i)
                    f_offspring.append(self.f(self.logger['offspring_population_x'][i]))

                self.logger['offspring_population_f'] = np.array(f_offspring)
                
                next_generation = self.survivor_selection(**kwargs)
                self.logger['parent_population_x']        = next_generation['parent_population_x']
                self.logger['parent_population_sigma']    = next_generation['parent_population_sigma']
                self.logger['parent_population_f']  = next_generation['parent_population_f']
                self.logger['current_iter'] += 1

        except KeyboardInterrupt:
            print("Interrupted, saving best solution found so far.")
        
        ind = np.argmin(self.logger['parent_population_f'])

        self.logger['best_individual'] = self.logger['parent_population_x'][ind]
        self.logger['best_f']   = self.logger['parent_population_f'][ind]

    def initialize_step_weights(self, eps_sigma:float, **kwargs) -> np.ndarray :
        steps = np.random.uniform(0,1, size=(self.logger['parent_population_size'], self._adaptive_mutation_operator.length))
        return np.maximum(steps,eps_sigma)

    def initialize_population(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            How to distribute the population. This function create a
            population using uniform distribution.
            This should return an matrix of size (M x N)
            where M is the number of population and N is the number of 
            variables.
        Arguments:
            -size_: Tnteger n where n is the number of variables about the problem.
        ------------------------------------------------------
        """
        return np.random.uniform( self.Bounds[0], self.Bounds[1], size=(self.logger['parent_population_size'],self.Decision_variables))

    def fixer(self, ind:int) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
        Function which helps to move solution in a valid region.
        Arguments:
            - ind: index of individual.
        ------------------------------------------------------
        """
        return self._fixer(self.logger['offspring_population_x'], ind)
    
    def crossover_operator(self, parent_ind1: np.ndarray,\
                                 parent_ind2: np.ndarray,\
                                 **kwargs) -> np.ndarray:
        return self._crossover_operator(self.logger['parent_population_x'],\
                                        parent_ind1,\
                                        parent_ind2)

    def adaptive_crossover(self, parent_ind1: np.ndarray,\
                                 parent_ind2: np.ndarray,\
                                 **kwargs) -> np.ndarray:
        return self._adaptive_crossover_operator(   self.logger['parent_population_sigma'],\
                                                    parent_ind1,\
                                                    parent_ind2)

    def mutation_operator(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            The current population is updated by specific change
            using the adaptive control. 
            This function should mutate the population.
        ------------------------------------------------------
        """
        return  self._mutation_operator(self.logger['offspring_population_x'],\
                                        self.logger['offspring_population_sigma'])

    def adaptive_mutation(self, **kwargs) -> np.ndarray:
        return self._adaptive_mutation_operator(self.logger['offspring_population_sigma'])

    def survivor_selection(self,**kwargs) -> dict:
        individuals = {}
        individuals['parent_population_x']     = [self.logger['parent_population_x'],\
                                                  self.logger['offspring_population_x']]
        individuals['parent_population_sigma'] = [self.logger['parent_population_sigma'],\
                                                  self.logger['offspring_population_sigma']]

        return self._survivor_selector( self.logger['parent_population_f'],\
                                        self.logger['offspring_population_f'],\
                                        individuals)
    #-----------------------------------------------------
                    #Private functions.
    #-----------------------------------------------------
    def is_invalid(self, x : np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            Check if the current solution is invalid. 
        ------------------------------------------------------
        """
        for constraint in self.Constraints:
            if not constraint(x):
                return True
        return False
    
    def get_pairs(self, **kwargs ):
        parent_ind1 = np.random.randint(self.logger['parent_population_size'], size=(self.logger['offspring_population_size'],))
        parent_ind2 = np.random.randint(self.logger['parent_population_size'], size=(self.logger['offspring_population_size'],))
    
        return parent_ind1,parent_ind2
