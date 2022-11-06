from audioop import cross
import math
from crossover import Crossover
from mutation import Mutation

class Individual:
    def __init__(self, chromosome, encoding, crossover_method, mutation_method, A, n, m, s):
        self.chromosome = chromosome
        self.encoding = encoding
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.n = n
        self.m = m
        self.s = s
        self.A = A
        self.mutate()
        self.fitness = self.fitness_function(self.chromosome)

    def fitness_function(self, chromosome):
        objective_function = self.A.T * np.diagflat(chromosome) * A
        penalty_component = - 0
        return objective_function + penalty_component

    def mutate(self):
        self.chromosome = Mutation.mutate(
            self.chromosome, 
            self.encoding, 
            self.mutation_method
        )

    # crossover
    def breed(self, another_individual):
        new_individual = None
        new_chromosome = Crossover.crossover(
            self.chromosome, 
            another_individual.chromosome,
            self.encoding, 
            self.crossover_method
        )
        new_individual = Individual(
            new_chromosome,
            encoding=self.encoding,
            crossover_method = self.crossover_method,
            mutation_method = self.mutation_method,
            A = self.A, 
            n = self.n, 
            m = self.m, 
            s = self.s
        )
        return new_individual


    