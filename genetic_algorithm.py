import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
import math
import numpy as np

class GeneticAlgoritm:
    def __init__(self, experiment):
        self.experiment_id = experiment["experiment_id"]
        self.seed = experiment["seed"]
        self.max_generations = experiment["max_generations"]
        self.population_size = experiment["population_size"]
        self.n = experiment["n"]
        self.m = experiment["m"]
        self.s = experiment["s"]
        self.A = experiment["A"]
        self.encoding = experiment["encoding_method"]
        self.selection_method = experiment["selection_method"]
        self.mutation_method = experiment["mutation_method"]
        self.crossover_method = experiment["crossover_method"]
        self.mutation_probability = experiment["mutation_probability"]
        self.crossover_probability = experiment["crossover_probability"]
        self.perform_assexual_crossover = experiment["perform_assexual_crossover"]
        self.best_sol = -np.inf
        self.elite_size = math.ceil(self.population_size * experiment["elite_size"])

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

    def loop(self):
        # generates initial population 
        population = self.initialize_population()
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

            # selects parents for the next generation
            parents = Selection.nbest(population, 2)
            self.log("Parents selected", parents)

            # generates children 
            children = []
            for i in range(0, len(parents)):
                if random.uniform(0,1) < self.crossover_probability:
                    id_parents = []
                    if self.perform_assexual_crossover:
                        id_parents = random.choices(range(0, len(population)), k=2)
                    else:
                        id_parents = random.choices(range(0, len(population)), k=2)
                        while id_parents[0] == id_parents[1]:
                            id_parents = random.choices(range(0, len(population)), k=2)

                    offspring = population[id_parents[0]].breed(population[id_parents[1]])
                    if max([i.fitness for i in offspring]) > self.best_sol:
                        self.best_sol = max([i.fitness for i in offspring])
                    children = children + list(offspring)

            self.log("Children produced", children)

            # mutates children
            for child in children:
                if random.uniform(0,1) < self.mutation_probability:
                    child.mutate()

            # updates the population
            population = population + children
            self.log("Candidate population", population)

            # selects individuals for the next generation
            population = Selection.nbest(population, self.elite_size) + getattr(Selection, self.selection_method)(population, self.population_size - self.elite_size)
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
        




