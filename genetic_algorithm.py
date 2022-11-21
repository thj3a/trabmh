import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
import numpy as np
from scipy.io import loadmat
import mat73
class GeneticAlgoritm:
    def __init__(self, seed, max_generations, population_size, A, n, m, s):
        self.seed = seed
        self.max_generations = max_generations
        self.population_size = population_size
        self.n = n
        self.m = m
        self.s = s
        self.A = A
        self.encoding = "binary"
        self.selection_method = "tournament_duo"
        self.mutation_method = "singlepoint_interchange"
        self.crossover_method = "misc"
        self.p_mutation = 0.2
        self.p_crossover = 0.8
        self.assexual_crossover = False
        self.best_sol = -np.inf
        self.nbest_size = int(population_size*0.1)

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
                encoding= self.encoding,
                crossover_method = self.crossover_method,
                mutation_method = self.mutation_method,
                A = self.A, 
                n = self.n, 
                m = self.m, 
                s = self.s,
            )
            if individual.fitness > self.best_sol:
                self.best_sol = individual.fitness
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
            for i in range(0, len(parents)):
                if random.uniform(0,1) < self.p_crossover:
                    id_parents = []
                    if self.assexual_crossover:
                        id_parents = random.choices(range(0, len(population)), k=2)
                    else:
                        id_parents = random.choices(range(0, len(population)), k=2)
                        while id_parents[0] == id_parents[1]:
                            id_parents = random.choices(range(0, len(population)), k=2)

                    offspring = population[id_parents[0]].breed(population[id_parents[1]])
                    if max([i.fitness for i in offspring]) > self.best_sol:
                        self.best_sol = max([i.fitness for i in offspring])
                    children = children + list(offspring)
            for child in children:
                if random.uniform(0,1) < self.p_mutation:
                    child.mutate()
            self.log("Children produced", children)

            # updates the population
            population = population + children
            self.log("Candidate population", population)

            # selects individuals for the next generation
            population = Selection.nbest(population, self.nbest_size) + getattr(Selection, self.selection_method)(population, self.population_size - self.nbest_size)
            #population = Selection.nbest(population, self.population_size)
            self.log("Population after selection", population)

            # updates the results plotted
            self.plot_results(population)

            # checks stopping criteria
            if generation == self.max_generations - 1:
                self.log("Maximum number of generations reached.")
            
        return population

    def plot_results(self, population):
        pass

    def log(self, message, additional_content = "", status = "INFO"):
        prepared_message = "{}" + message + "\033[0m"
        if status == "INFO":
            print(prepared_message.format("\033[96m"), additional_content)
        elif status == "ERROR":
            print(prepared_message.format("\033[91m"), additional_content)
        else: # warning
            print(prepared_message.format("\033[93m"), additional_content)
        




instance = loadmat("D-Opt-files/Instances/Instance_40_1.mat")

A = instance["A"]
print(A.shape)
n = A.shape[0]
m = A.shape[1]
s = int(n/2)

result = mat73.loadmat("D-Opt-files/ResultsInstances/x_ls_40_1.mat")
best_r = np.linalg.slogdet(np.matmul(np.matmul(A.T, np.diagflat(result['x_ls'])), A))

d_opt = GeneticAlgoritm(0, 10000, 1000, A, n, m, s)
results = d_opt.loop()