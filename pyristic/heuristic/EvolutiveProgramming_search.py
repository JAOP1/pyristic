from pyristic.utils.helpers import function_type, ContinuosFixer, EvolutionaryProgrammingConfig
from pyristic.utils.operators import mutation,selection
from tqdm import tqdm
import numpy as np

__all__=['EvolutionaryProgramming']

class EvolutionaryProgramming:
    
    def __init__(self,  function: function_type,\
                        decision_variables: int,\
                        constraints:list=[],\
                        bounds: list=[],\
                        config = EvolutionaryProgrammingConfig()):
        """
        ------------------------------------------------------
        Description:
            Initializing every variable necessary to the search.
        Arguments: 
            - Function: Objective function to minimize.
            - Decision variables: Number of decision variables. 
            - Constraints: Constraints to be a feasible solution.
            - Bounds: Bounds for each variable. This should be a matrix 2 x N
            where N is the variables number. The first element is the lower limit, 
            and another one is the upper limit.  
            - Config: EvolutationaryProgrammingConfig, which changes the behavior of
            some operators.
        ------------------------------------------------------ 
        """

        self.f = function
        self.Constraints = constraints
        self.Bounds = bounds
        self.Decision_variables = decision_variables

        #Setting operators.
        self._mutation_operator           = config.mutation_op          if config.mutation_op != None          else mutation.sigma_mutator()
        self._survivor_selector           = config.survivor_selector    if config.survivor_selector != None    else selection.merge_selector()
        self._fixer                       = config.fixer                if config.fixer != None                else ContinuosFixer(self.Bounds)
        self._adaptive_mutation_operator  = config.adaptive_mutation_op if config.adaptive_mutation_op != None else mutation.sigma_ep_adaptive_mutator(self.Decision_variables, 0.5)

        #Global information.
        self.logger = {}
        self.logger['best_individual']              = None
        self.logger['best_f']                       = None
        self.logger['current_iter']                 = None
        self.logger['total_iter']                   = None
        self.logger['population_size']              = None 
        self.logger['parent_population_x']          = None
        self.logger['parent_population_sigma']      = None
        self.logger['parent_population_f']          = None   
        self.logger['offspring_population_x']       = None
        self.logger['offspring_population_sigma']   = None
        self.logger['offspring_population_f']       = None

    def __str__(self):
        printable = "Evolutive Programming search: \n f(X) = {} \n X = {} \n ".format(self.logger['best_f'], self.logger['best_individual'])
        first = True
        
        for i in range(len(self.Constraints)):
            if self.Constraints[i].__doc__ != None:
                
                if first:
                    first = False
                    printable += "Constraints: \n "
                    
                self.Constraints[i](self.logger['best_individual'])
                printable += "{} \n".format( self.Constraints[i].__doc__)
                    
        return printable

    def optimize(self,  generations: int ,\
                        size_population: int,\
                        verbose=True,\
                        **kwargs) ->None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using Evolutionary programming.
        Arguments:
            - Generations: Number of iterations.
            - Population size: Number of individuals. 
        ------------------------------------------------------
        """
        #Reset global solution.
        self.logger['current_iter']       = 0
        self.logger['total_iter']         = generations
        self.logger['population_size']    = size_population
        self.logger['best_individual']         = None
        self.logger['best_f']           = None  

        #Initial population.
        
        self.logger['parent_population_x']       = self.initialize_population(**kwargs)
        self.logger['parent_population_sigma']   = self.initialize_step_weights(**kwargs)
        self.logger['parent_population_f'] =  np.apply_along_axis(self.f , 1, self.logger['parent_population_x'])      



        try:
            for g in tqdm(range(generations), disable= not verbose):
                #Mutation.
                self.logger['offspring_population_sigma']  = self.adaptive_mutation(**kwargs)
                
                self.logger['offspring_population_x']      = self.mutation_operator(**kwargs)
                
                #Fixing solutions and getting aptitude.
                f_offspring = []
                for ind in range(len(self.logger['offspring_population_x'])):
                    if self.is_invalid(self.logger['offspring_population_x'][ind]):
                        self.logger['offspring_population_x'][ind] = self.fixer(ind)
                    f_offspring.append(self.f(self.logger['offspring_population_x'][ind]))
                self.logger['offspring_population_f'] = np.array(f_offspring)

                #Survivor selection.
                next_generation = self.survivor_selection(**kwargs)
                self.logger['parent_population_x']        = next_generation['parent_population_x']
                self.logger['parent_population_sigma']    = next_generation['parent_population_sigma']
                self.logger['parent_population_f']  = next_generation['parent_population_f']

                self.logger['current_iter']  += 1

        except KeyboardInterrupt:
            print("Interrupted, saving best solution found so far.")
        
        ind = np.argmin(self.logger['parent_population_f'])
        self.logger['best_individual'] = self.logger['parent_population_x'][ind]
        self.logger['best_f']   = self.logger['parent_population_f'][ind]
    
    def mutation_operator(self,**kwargs) -> None:
        """
        ------------------------------------------------------
        Description:
            A new population is created from the current population. 
            This function applies small changes to the decision variables 
            and the step sizes.
        ------------------------------------------------------
        """
        return self._mutation_operator(self.logger['parent_population_x'],\
                                       self.logger['parent_population_sigma'])

    def adaptive_mutation(self, **kwargs) -> None:
        return self._adaptive_mutation_operator(self.logger['parent_population_sigma'])

    def survivor_selection(self,**kwargs) -> dict:
        individuals = {}
        individuals['parent_population_x'] = [self.logger['parent_population_x'], self.logger['offspring_population_x']]
        individuals['parent_population_sigma']      = [self.logger['parent_population_sigma'], self.logger['offspring_population_sigma']]

        return self._survivor_selector( self.logger['parent_population_f'],\
                                        self.logger['offspring_population_f'],\
                                        individuals)

    def initialize_step_weights(self, **kwargs) -> np.ndarray :
        steps = np.random.uniform(0,1, size=(self.logger['population_size'] , self.Decision_variables))
        return np.maximum(steps,0.00001)

    def initialize_population(self, **kwargs) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
            This function creates a population using a uniform distribution.
            It returns a matrix of size (n x m), where n is the size population 
            and m is the number of variables.
        Arguments:
            -n: The population size. 
        ------------------------------------------------------
        """
        return np.random.uniform( self.Bounds[0], self.Bounds[1], size=(self.logger['population_size'],self.Decision_variables))

    def fixer(self, ind:int) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
        This function helps to move one solution to the feasible region.
        Arguments:
            - ind: Index of the individual.
        ------------------------------------------------------
        """
        return  self._fixer(self.logger['offspring_population_x'], ind)

    def is_invalid(self, x : np.ndarray) -> bool:
        """
        ------------------------------------------------------
        Description:
            This function checks if the current solution is unfeasible. 
        ------------------------------------------------------
        """
        for constraint in self.Constraints:
            if not constraint(x):
                return True
        return False