
import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
import numpy as np
from scipy.io import loadmat

class GeneticAlgoritm:
    def __init__(self, seed, max_generations, init_population_size, A, n, m, s):
        self.seed = seed
        self.max_generations = max_generations
        self.init_population_size = init_population_size
        self.n = n
        self.m = m
        self.s = s
        self.A = A

        self.check_params()

        random.seed(self.seed)


    def initialize_population(self):
        population = [] 

        for i in range(0, self.init_population_size):
            chromosome = np.zeros((self.n, 1))

            # random.choices will select a value with replacement
            # so, for now, the code tolerates sum(chromosome <= self.n)
            for position in random.choices(range(0, self.n), k = self.s): 
                chromosome[position] = 1

            individual = Individual(
                chromosome,
                encoding="binary",
                crossover_method = "",
                mutation_method = "",
                A = self.A, 
                n = self.n, 
                m = self.m, 
                s = self.s
            )

            population.append(individual)

        return population

    def check_params(self):
        pass

    def loop(self):
        # generates initial population 
        population = self.initialize_population()

        for generation in range(self.max_generations):
            # selects parents for the next generation
            parents = Selection.select_parents(population, self)
            
            # generates and mutates children 
            children = [parents[i].breed(parents[i + 1]) for i in range(0, len(parents), 2)]

            # updates the population

            # checks stopping criteria
            
            pass

    def log(self):
        pass





instance = loadmat("instances/Instance_40_1.mat")
A = instance["A"]
print(A.shape)
n = A.shape[0]
m = A.shape[1]
s = int(n/2)

d_opt = GeneticAlgoritm(0, 100, 5, A, n, m, s)
d_opt.loop()