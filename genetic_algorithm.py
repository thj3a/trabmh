import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
from utils import Utils
import math
import numpy as np
import pdb

class GeneticAlgoritm:
    def __init__(self, experiment):
        self.experiment_id = experiment["experiment_id"]
        self.seed = experiment["seed"]
        self.silent = experiment["silent"]
        self.max_generations = experiment["max_generations"]
        self.population_size = experiment["population_size"]
        self.n = experiment["n"]
        self.m = experiment["m"]
        self.s = experiment["s"]
        self.A = experiment["A"]
        self.encoding = experiment["encoding_method"]
        self.selection_method = experiment["selection_method"]
        self.mutation_method = experiment["mutation_method"]
        self.self_mutation = experiment["self_mutation"]
        self.crossover_method = experiment["crossover_method"]
        self.parent_selection_method = experiment["parent_selection_method"]
        self.mutation_probability = experiment["mutation_probability"]
        self.crossover_probability = experiment["crossover_probability"]
        self.perform_assexual_crossover = experiment["perform_assexual_crossover"]
        self.best_sol = -np.inf
        self.elite_size = math.ceil(self.population_size * experiment["elite_size"])
        # offspring_size is a limit on how many children are going going to be generated.
        # Ultimately, it is the crossover_probability that will control if a mating attempt
        # will be successful or not. For that reason, offspring_size shouldn't be confused 
        # with a target number.
        # In other words, if crossover_probability = 1.0, offspring_size will be the number of
        # individuals to be generated. However, if crossover_probability < 1.0, the number of
        # individuals generated will probably be less than offspring_size.
        self.offspring_size = math.floor(self.population_size * experiment["offspring_size"])
        self.commoners_size = self.population_size - self.elite_size

        random.seed(self.seed)


    def initialize_population(self):
        population = [] 

        # TODO Add the option to produce initial solutions using 
        # Gabriel's heuristics and local search.
        # The current selection method will be called "random"

        for i in range(0, self.population_size):
            chromosome = np.zeros((self.n, 1))

            for position in random.sample(range(0, self.n), k = self.s): 
                chromosome[position] = 1

            if self.encoding == "permutation":
                chromosome = np.array([np.where(chromosome == 1)[0]]).T

            individual = Individual(
                chromosome,
                self
            )
            if individual.fitness > self.best_sol:
                self.best_sol = individual.fitness
            population.append(individual)

        return population

    def loop(self):
        # generates initial population 
        population = self.initialize_population()
        elite = None
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

            # Sorts the population. It is required because some selection methods rely on it
            population = Utils.sort_population(population)

            # generates children 
            children = []
            children_generation_attempts = 0

            while children_generation_attempts < self.offspring_size:
                if random.uniform(0,1) < self.crossover_probability:
                    # Selects parents
                    parents = Selection.select(population, 2, self.parent_selection_method)
                    offspring = parents[0].breed(parents[1])
                    children = children + list(offspring)
                
                if self.encoding == "binary":
                    # Usually, each successful attempt generates 2 children. For that reason,
                    # each unsuccessful attempt will also count as 2 attempts.
                    children_generation_attempts += 2
                elif self.encoding == "permutation": 
                    # Permutations usually produce only one child.
                    children_generation_attempts += 1

            self.log("Children produced", children)

            # Children produced through mutation
            mutants = []

            # Mutates the population, generating children in an assexual way
            # individuals that generate a mutant are not modified or removed
            # in this step.
            # Children created right before this step are not mutated in the
            # current generation.
            for individual in population:
                if random.uniform(0,1) < self.mutation_probability:
                    mutant = individual.mutate(self.self_mutation)
                    if mutant is not None:
                        mutants.append(mutant)

            # Updates the population
            population = population + children + mutants
            self.log("Candidate population", population)

            # updates the value of the best solution
            self.best_sol = Utils.get_best_solution(population, self.best_sol)

            # Sorts the population. It is required because some selection methods rely on it
            population = Utils.sort_population(population)

            # selects individuals for the next generation
            elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
            commoners = Selection.select(commoners, self.commoners_size, self.selection_method)
            population = elite + commoners
            self.log("Population after elitism and selection", population)

            # updates the results plotted
            self.plot_results(population)

            # checks stopping criteria
            if generation == self.max_generations - 1:
                self.log("Maximum number of generations reached.")

        # Performs path relinking.

        # Updates the elite set.
            
        return elite

    def plot_results(self, population):
        pass

    def log(self, message, additional_content = "", status = "INFO"):
        if self.silent:
            return

        prepared_message = "{}" + message + "\033[0m"
        if status == "INFO":
            print(prepared_message.format("\033[96m"), additional_content)
        elif status == "ERROR":
            print(prepared_message.format("\033[91m"), additional_content)
        else: # warning
            print(prepared_message.format("\033[93m"), additional_content)
        




