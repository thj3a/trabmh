import math
import numpy as np
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
        self.objective_function = None # will be calculated during the fitness calculation
        self.penalty = None # will be calculated during the fitness calculation
        self.fitness = None # will be calculated during the call of fitness function
        self.fitness_function()

    def description(self):
        num_1s = int(sum(self.chromosome))
        return "\033[95mIndividual|fitness:{}|obj_func:{}|s:{}|num_1s:{}|chromosome:{}\033[0m".format(
            self.fitness, 
            self.objective_function, 
            self.s, 
            num_1s, 
            self.chromosome.T if self.chromosome.shape[0] <= 20 else "too long"
        )

    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.description()

    def fitness_function(self):
        sign, self.objective_function = np.linalg.slogdet(np.matmul(np.matmul(self.A.T, np.diagflat(self.chromosome)), self.A))

        if sign < 0:
            #self.objective_function = - math.inf
            self.objective_function = self.objective_function*2
        self.penalty = - abs(self.s - int(sum(self.chromosome))) # test

        self.fitness = self.objective_function + self.penalty

    def mutate(self):
        self.chromosome = Mutation.mutate(
            self.chromosome, 
            self.encoding, 
            self.mutation_method
        )

    # crossover
    def breed(self, another_individual):
        new_individual_1 = None
        new_individual_2 = None
        
        new_chromosome_1, new_chromosome_2 = Crossover.crossover(
            self.chromosome, 
            another_individual.chromosome,
            self.encoding, 
            self.crossover_method
        )
        
        new_individual_1 = Individual(
            new_chromosome_1,
            encoding=self.encoding,
            crossover_method = self.crossover_method,
            mutation_method = self.mutation_method,
            A = self.A, 
            n = self.n, 
            m = self.m, 
            s = self.s
        )

        new_individual_2 = Individual(
            new_chromosome_2,
            encoding=self.encoding,
            crossover_method = self.crossover_method,
            mutation_method = self.mutation_method,
            A = self.A, 
            n = self.n, 
            m = self.m, 
            s = self.s
        )
        return new_individual_1, new_individual_2


    