from utils import Utils
import random
import numpy as np
from copy import deepcopy
import math
import pdb

class Selection:
    def __init__(self):
        pass

    @classmethod
    def nbest(self, population, n):
        sorted_population = Utils.sort_population(population)
        return sorted_population[:n]

    def nbestdifferent(self, population, n):
        return Utils.get_n_best_individuals_without_repetition(population, n)
        

    @classmethod
    def percent_population(self, population, perc, best=True):
        number_of_individuals = math.floor(len(population) * perc)
        if best:
            return population[:number_of_individuals]
        else:
            return population[(len(population) - number_of_individuals):]

    # Performs the class-selection method. The selection is done with replacement
    # - population -> A list containing the individuals compose the current population.
    #                 That list must be already sorted.
    # - expected_size -> Number of individuals to be retrieved
    # Returns:
    # - ? -> A list of "expected_size" selected individuals. 
    @classmethod
    def byclass(self, population, expected_size, percent_best, percent_worst):
        percent_best = float(percent_best)
        percent_worst = float(percent_worst)

        if (percent_best + percent_worst) > 1.0:
            # TODO add a log
            raise Exception("percent_best + percent_worst) > 1.0! \nIt must be 0 < (percent_best + percent_worst) <= 1.0")

        best_individuals= self.percent_population(population, percent_best, True)
        worst_individuals = self.percent_population(population, percent_worst, False)
        expected_remaining = expected_size - (len(best_individuals) + len(worst_individuals))
        random_individuals = random.choices(population, k=expected_remaining) # with replacement

        return best_individuals + worst_individuals + random_individuals

    @classmethod
    def fullyrandom(self, population, n):
        return random.choices(population, k=n) # with replacement

    # Performs the ranking-selection method. It attributes a probability of selection 
    # that is proportional to its fitness. The selection is done with replacement.
    # THIS IS NOT THE SAME AS THE "nbest".
    # Params:
    # - population -> A list containing the individuals compose the current population.
    #                 That list must be already sorted.
    # - n -> Number of individuals to be retrieved
    # Returns:
    # - ? -> A list of n selected individuals.
    @classmethod
    def ranking(self, population, n):
        weights = list(range(len(population), 0, -1))
        return random.choices(population, weights=weights, k=n)

    # Performs the rouletted-selection method. The selection is done with replacement
    # - population -> A list containing the individuals compose the current population.
    # - n -> Number of individuals to be retrieved
    # Returns:
    # - ? -> A list of n selected individuals.
    @classmethod
    def roulette(self, population, n):
        weights = np.array([individual.fitness for individual in population], dtype=float)
        return random.choices(population, weights=1/(1+np.exp(-weights)), k=n)
    
    @classmethod
    def tournament(self, population, n, k):
        k = int(k)
        winners = []
        for _ in range(n):
            i = random.choice(range(0, len(population)))
            ids_participants = [i] + random.choices(range(len(population)), k=k)
            winner = ids_participants[np.argmax([population[i].fitness for i in ids_participants])]
            winners.append(deepcopy(population[winner]))
        return winners

    @classmethod
    def select(self, individuals, n, function_name):
        function_and_params = function_name.split("_")
        function_name = function_and_params[0]
        params = function_and_params[1:] if len(function_and_params) > 0 else []

        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(individuals, n, *params)
        else:
            raise Exception(f"Method \"{function_name}\" not found.")

    #@classmethod
    #def select_parents(self, individuals, method):
    #    function_name = "parents" + "_" + method
    #    return select(individuals, function_name)
        
    #@classmethod
    #def select_nextgen(self, individuals, method):
    #    function_name = "nextgen" + "_" + method
    #    return select(individuals, function_name)