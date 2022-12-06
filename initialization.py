import numpy as np
import random
from individual import Individual
import time
import pdb 

class Initialization:

    def __init__(self):
        pass

    @classmethod
    def binary_random(self, environment, population_size):

        population = [] 

        for i in range(0, population_size):

            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1

            individual = Individual(
                chromosome,
                environment
            )
            if individual.fitness > environment.best_sol:
                environment.best_sol = individual.fitness
            population.append(individual)

        return population

    @classmethod
    def binary_biased(self, environment, population_size, min_diff):
        min_diff = float(min_diff)
        population = [] 
        while len(population) < population_size:
            
            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
            
            found_close = False
            for individual in population:
                if np.mean(np.abs(individual.chromosome - chromosome)) < min_diff:
                    found_close = True
                    break
            if not found_close:
                individual = Individual(
                    chromosome,
                    environment
                )

                if individual.fitness > environment.best_sol:
                    environment.best_sol = individual.fitness
                population.append(individual)

        return population
    
    @classmethod
    def permutation_random(self, environment, population_size):
        
        population = [] 
        
        for i in range(0, population_size):

            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1

            chromosome = np.array([np.where(chromosome == 1)[0]]).T

            individual = Individual(
                chromosome,
                environment
            )
            if individual.fitness > environment.best_sol:
                environment.best_sol = individual.fitness
            population.append(individual)

        return population

    @classmethod
    def Gabriel_heuristic_local_search(self, environment, population_size):
        # TODO Add the option to produce initial solutions using 
        # Gabriel's heuristics and local search.
        pass

    @classmethod
    def initialize_population(self, environment, population_size):

        function_and_params = environment.initialization_method.split("_")
        function_name = function_and_params[0]
        params = function_and_params[1:] if len(function_and_params) > 0 else []
        function_name = environment.encoding + "_" + function_name
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(environment, population_size, *params)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))