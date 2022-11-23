from select import select
import operator
import random
import numpy as np
import copy

class Selection:
    def __init__(self):
        pass

    @classmethod
    def nbest(self, population, n):
        ordered_population = sorted(population, key=operator.attrgetter("fitness"), reverse=True)
        return ordered_population[:n]

    @classmethod
    def roulette(self, population, n):
        weights = np.array([individual.fitness for individual in population])
        weights = weights / sum(weights)
        return random.choices(population, weights=weights, k=n)
    
    @classmethod
    def tournament_duo(self, population, n):
        winners = []
        for _ in range(n):
            i = random.choice(range(0, len(population)))
            rivals = [i] + random.choices(range(len(population)), k=2)
            winners.append(copy.deepcopy(population[rivals[np.argmax([population[i].fitness for i in rivals])]]))
        return winners

    @classmethod
    def select(self, individuals, function_name):
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(individuals)
        else:
            raise Exception(f"Method \"{function_name}\" not found.")

    @classmethod
    def select_parents(self, individuals, method):
        function_name = "parents" + "_" + method
        return select(individuals, function_name)
        
    @classmethod
    def select_nextgen(self, individuals, method):
        function_name = "nextgen" + "_" + method
        return select(individuals, function_name)