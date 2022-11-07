
import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
import numpy as np
from scipy.io import loadmat

class GeneticAlgoritm:
    def __init__(self, seed, max_generations, population_size, A, n, m, s):
        self.seed = seed
        self.max_generations = max_generations
        self.population_size = population_size
        self.n = n
        self.m = m
        self.s = s
        self.A = A

        print(self.n, self.m)

        self.check_params()

        random.seed(self.seed)


    def initialize_population(self):
        population = [] 

        for i in range(0, self.population_size):
            chromosome = np.zeros((self.n, 1))

            for position in random.sample(range(0, self.n), k = self.s): 
                chromosome[position] = 1

            individual = Individual(
                chromosome,
                encoding="binary",
                crossover_method = "singlepoint",
                mutation_method = "singlepoint",
                A = self.A, 
                n = self.n, 
                m = self.m, 
                s = self.s,
                mutate=False
            )

            population.append(individual)

        return population

    def check_params(self):
        pass

    def loop(self):
        # generates initial population 
        population = self.initialize_population()
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

            # selects parents for the next generation
            parents = Selection.nbest(population, 2)
            self.log("Parents selected", parents)

            # generates and mutates children 
            children = []
            for i in range(0, len(parents), 2):
                offspring = parents[i].breed(parents[i + 1]) 
                children = children + list(offspring)

            self.log("Children produced", children)

            # updates the population
            population = population + children
            self.log("Candidate population", population)


            # selects individuals for the next generation
            population = Selection.nbest(population, self.population_size)
            self.log("Population after selection", population)

            # checks stopping criteria
            if generation >= self.max_generations:
                self.log("Maximum number of generations reached.")
                break
            
            break

    def log(self, message, additional_content = "", status = "INFO"):
        prepared_message = "{}" + message + "\033[0m"
        if status == "INFO":
            print(prepared_message.format("\033[96m"), additional_content)
        elif status == "ERROR":
            print(prepared_message.format("\033[91m"), additional_content)
        else: # warning
            print(prepared_message.format("\033[93m"), additional_content)
        





instance = loadmat("instances/Instance_40_1.mat")
A = instance["A"]
print(A.shape)
n = A.shape[0]
m = A.shape[1]
s = int(n/2)

d_opt = GeneticAlgoritm(0, 100, 5, A, n, m, s)
d_opt.loop()