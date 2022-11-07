from audioop import cross
import math
import numpy as np
from crossover import Crossover
from mutation import Mutation

class Individual:
    def __init__(self, chromosome, encoding, crossover_method, mutation_method, A, n, m, s, mutate = True):
        self.chromosome = chromosome
        self.encoding = encoding
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.n = n
        self.m = m
        self.s = s
        self.A = A

        if mutate:
            self.mutate()

        self.fitness = self.fitness_function(self.chromosome)

    def fitness_function(self, chromosome):
        print(len(chromosome))
        print(self.A.T.shape, np.diagflat(chromosome).shape)
        print(np.matmul(self.A.T, np.diagflat(chromosome)))
        #print(np.matmul(np.matmul(self.A.T, np.diagflat(chromosome)), self.A))
        
        sign, objective_function = np.linalg.slogdet(np.matmul(np.matmul(self.A.T, np.diagflat(chromosome)), self.A))

        if sign < 0:
            objective_function = - math.inf

        penalty_component = - 0 # incomplete

        return objective_function + penalty_component

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
            new_chromosome_1,
            encoding=self.encoding,
            crossover_method = self.crossover_method,
            mutation_method = self.mutation_method,
            A = self.A, 
            n = self.n, 
            m = self.m, 
            s = self.s
        )
        return new_individual_1, new_individual_2


    