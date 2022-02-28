from pyristic.utils.helpers import function_type
from tqdm import tqdm
import numpy as np

__all__= ['Genetic']

class Genetic:

    def __init__(self,  function: function_type,\
                        decision_variables:int,\
                        constraints:list=[],\
                        bounds: list=[],\
                        config = None):
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
        #Information about problem.
        self.f = function
        self.Constraints = constraints
        self.Bounds = bounds
        self.Decision_variables = decision_variables # Decision variables.

        #Operators.
        if config != None:
            self._mutation_operator  = config.mutation_op
            self._crossover_operator = config.cross_op
            self._survivor_selector  = config.survivor_selector
            self._parent_selector    = config.parent_selector
            self._fixer              = config.fixer
        #Search information.
        self.logger = {}
        self.logger['best_individual']      = None
        self.logger['best_f']        = None
        self.logger['current_iter']    = None
        self.logger['total_iter']      = None
        self.logger['population_size'] = None

    def __str__(self):
        printable = "Genetic search: \n F_a(X) = {} \n X = {} \n ".format(self.logger['best_f'], self.logger['best_individual'])
        first = True

        for i in range(len(self.Constraints)):
            if self.Constraints[i].__doc__ != None:

                if first:
                    first = False
                    printable += "Constraints: \n "

                self.Constraints[i](self.logger['best_individual'])
                printable += "{} \n".format( self.Constraints[i].__doc__)

        return printable

    #-----------------------------------------------------
                    #Public functions.
    #-----------------------------------------------------
    def optimize(self,  generations:int ,\
                        size_population: int,\
                        cross_percentage: float = 1.0,\
                        mutation_percentage: float = 1.0,\
                        verbose:bool=True,\
                        **kwargs) -> None:
        """
        ------------------------------------------------------
        Description:
            The main function to find the best solution using tabu search.
        Arguments:
            -generations: number of iterations.
            -population: Number of new solutions.
        ------------------------------------------------------
        """
        assert 0 <= cross_percentage <= 1.0
        assert 0 <= mutation_percentage <= 1.0
        #Reset global information.
        self.logger['current_iter']         = 0
        self.logger['total_iter']           = generations
        self.logger['population_size']      = size_population
        self.logger['cross_percentage']     = cross_percentage
        self.logger['mutation_percentage']  = mutation_percentage
        self.logger['best_f']               = None
        self.logger['best_individual']      = None

        #Initial population.
        self.logger['parent_population_x']  = self.initialize_population(**kwargs)
        self.logger['parent_population_f']  = np.apply_along_axis(self.f , 1, self.logger['parent_population_x'])
        
        try:
            for g in tqdm(range(generations), disable = not verbose):
                #Parent selection.
                parent_ind = self.parent_selection(**kwargs)
                first_parent_indices, second_parent_indices = self.__get_pairs(parent_ind)
                #Cross.
                self.__cross_individuals(first_parent_indices, second_parent_indices)
                #mutate.
                self.__mutate_individuals()

                #Fixing solutions and getting aptitude.
                f_offspring = []
                for ind in range(len(self.logger['offspring_population_x'])):
                    if self.is_invalid(self.logger['offspring_population_x'][ind]):
                        self.logger['offspring_population_x'][ind] = self.fixer(ind)
                    f_offspring.append(self.f(self.logger['offspring_population_x'][ind]))
                self.logger['offspring_population_f'] = np.array(f_offspring)

                #Survivor selection.
                next_generation = self.survivor_selection(**kwargs)
                self.logger['parent_population_x']  = next_generation['population']
                self.logger['parent_population_f']  = next_generation['parent_population_f']

                self.logger['current_iter'] += 1

        except KeyboardInterrupt:
            print("Interrupted, saving best solution found so far.")

        ind = np.argmin(self.logger['parent_population_f'])
        self.logger['best_individual'] = self.logger['parent_population_x'][ind]
        self.logger['best_f']   = self.logger['parent_population_f'][ind]
    
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
            -size_: tuple with two integers (m,n)
            where m is the number generated of new solutions and
            n is the number of variables about the problem.
        ------------------------------------------------------
        """
        return np.random.uniform( self.Bounds[0], self.Bounds[1], size=(self.logger['population_size'], self.Decision_variables))
        
    def fixer(self, ind: int) -> np.ndarray:
        """
        ------------------------------------------------------
        Description:
        Function which helps to move solution in a valid region.
        Arguments:
            -X: numpy array, this one is the current population.
            -ind: index of individual.
        ------------------------------------------------------
        """
        if self._fixer == None:
            raise NotImplementedError
        return self._fixer(self.logger['offspring_population_x'], ind)

    def mutation_operator(self, indices,**kwargs):
        """
        ------------------------------------------------------
        Description:
            Apply the mutation operator selected by the configuration.
        Arguments:
            - Indices: the indices (or boolean array) of individuals in offspring_population_x 
            to mutate.
        ------------------------------------------------------
        """
        if self._mutation_operator == None:
            raise NotImplementedError
        return self._mutation_operator(self.logger['offspring_population_x'][indices])

    def crossover_operator(self,parent_ind1: np.ndarray,\
                                parent_ind2: np.ndarray,\
                                **kwargs):
        if self._crossover_operator == None:
            raise NotImplementedError
        return self._crossover_operator(self.logger['parent_population_x'], parent_ind1, parent_ind2)

    def survivor_selection(self,**kwargs) -> dict:
        if self._survivor_selector == None:
            raise NotImplementedError

        individuals = {}
        individuals['population'] = [self.logger['parent_population_x'], self.logger['offspring_population_x']]

        return self._survivor_selector( self.logger['parent_population_f'],\
                                        self.logger['offspring_population_f'],\
                                        individuals)

    def parent_selection(self,**kwargs) -> np.ndarray:
        if self._parent_selector == None:
            raise NotImplementedError
        return self._parent_selector(self.logger['parent_population_f'])

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

    def __get_pairs(self,parent_ind : np.ndarray):
        parent_ind1=[]
        parent_ind2=[]
        size_ = len(parent_ind)//2

        for a,b in np.random.randint(0,len(parent_ind),size=(size_,2)):
            parent_ind1.append(parent_ind[a])
            parent_ind2.append(parent_ind[b])

        return parent_ind1,parent_ind2

    def __cross_individuals(self, first_parents: np.ndarray, second_parents: np.ndarray) -> None:

        offspring = []
        for i in range(len(first_parents)):
            H1 = self.logger['parent_population_x'][first_parents[i]] 
            H2 = self.logger['parent_population_x'][second_parents[i]]
            if(np.random.rand() <= self.logger['cross_percentage']):
                H1,H2= self.crossover_operator([first_parents[i]], [second_parents[i]])
            offspring += [H1,H2]

        self.logger['offspring_population_x'] = np.array(offspring)


    def __mutate_individuals(self):
        individuals, variables = self.logger['offspring_population_x'].shape
        individuals_to_mutate = [np.random.rand() <= self.logger['mutation_percentage'] for i in range(individuals)]
        self.logger['offspring_population_x'][individuals_to_mutate] = self.mutation_operator(individuals_to_mutate)