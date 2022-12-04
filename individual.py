import math
import numpy as np
from crossover import Crossover
from mutation import Mutation
import pdb

class Individual:
    def __init__(self, chromosome, environment): #encoding, crossover_method, mutation_method, A, n, m, s):
        self.environment = environment
        self.chromosome = chromosome
        self.binary_chromosome = self.generate_binary_chromosome()
        self.environment = environment
        self.objective_function = None # will be calculated during the fitness calculation
        self.penalty = None # will be calculated during the fitness calculation
        self.fitness = None # will be calculated during the call of fitness function
        self.individual_hash = None
        self.fitness_function()
        self.calculate_hash()

    def description(self):
        num_1s = int(sum(self.chromosome))
        return "\033[95mIndividual|fitness:{}|obj_func:{}|s:{}|num_1s:{}|chromosome:{}\033[0m".format(
            self.fitness, 
            self.objective_function, 
            self.environment.s, 
            num_1s, 
            self.chromosome.T if self.chromosome.shape[0] <= 20 else "too long"
        )

    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.description()

    def fitness_function(self):
        sign, self.objective_function = np.linalg.slogdet(
            np.matmul(
                np.matmul(
                    self.environment.A.T, 
                    np.diagflat(self.binary_chromosome)
                ), 
                self.environment.A
            )
        )

        if sign < 0:
            #self.objective_function = - math.inf
            self.objective_function = self.objective_function*2

        self.penalty = - abs(self.environment.s - int(sum(self.binary_chromosome))) 

        self.fitness = self.objective_function + self.penalty

    # Returns the binary representation of the chromosome.
    # If the internal representation is already binary, it just returns it.
    # Otherwise, it converts the chromosome to binary.
    def generate_binary_chromosome(self):
        if self.environment.encoding == "binary":
            return self.chromosome

        bin_chromosome = np.zeros((self.environment.n, 1))
        bin_chromosome[self.chromosome] = 1
        
        return bin_chromosome

    def calculate_hash(self):
        #bin_string = "".join(str(i) for i in self.chromosome)
        #self.individual_hash = int(bin_string, 2)
        self.individual_hash = int(self.binary_chromosome.T.dot(2**np.arange(self.binary_chromosome.T.size)[::-1]))

    def mutate(self, self_mutation = False):
        mutated_chromosome = Mutation.mutate(
            self.chromosome, 
            self.environment
        )

        if self_mutation:
            self.chromosome = mutated_chromosome
            self.binary_chromosome = self.generate_binary_chromosome()
            self.fitness_function()
            self.calculate_hash()
        else:
            return Individual(
                mutated_chromosome,
                self.environment
            )
            
        return None



    # crossover
    def breed(self, another_individual):
        new_individuals = []
        
        new_chromosome_1, new_chromosome_2 = Crossover.crossover(
            self.chromosome, 
            another_individual.chromosome,
            self.environment
        )
        
        new_individual_1 = Individual(
            new_chromosome_1,
            self.environment
        )

        new_individuals.append(new_individual_1)

        if new_chromosome_2 is not None:
            new_individual_2 = Individual(
                new_chromosome_2,
                self.environment
            )

            new_individuals.append(new_individual_2)
        
        return new_individuals


    