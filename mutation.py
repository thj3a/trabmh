import random
import numpy as np
import copy
class Mutation:
    def __init__(self):
        pass

    #must use fitness function similar to lagrangian function
    @classmethod
    def binary_singlepoint(self, chromosome, environment):
        point = random.choice(range(0, len(chromosome)))
        chromosome[point] = 1 - chromosome[point]
        return chromosome
    
    @classmethod
    def binary_singlepointinterchange(self, chromosome, environment):
        ones = np.where(chromosome == 1)[0]
        zeros = np.where(chromosome == 0)[0]
        #print(chromosome.T)
        if len(ones) > 0:
            flip_point_0, flip_point_1  = random.choice(ones), random.choice(zeros) 
            chromosome[flip_point_0] = 0
            chromosome[flip_point_1] = 1
        else:
            # Deals with the cas where the number of 1s in a solution is 0.
            # Such a case can arise if unfeasible solutions are allowed,
            # like whan a constraint unaware crossover method is used.
            chromosome[random.choice(range(len(chromosome)))] = 1


        return chromosome

    # Swaps a range element for an elemento that isn't present in 
    # the chromosome. This is valid and needed because the current
    # problem (d-optimality) doesn't care about ordering. For that
    # reason, mutation occurs when a new element is inserted in the
    # solution (chromosome). 
    @classmethod
    def permutation_singleexchange(self, chromosome, environment):
        # executing "list(range(0, environment.s))" every time this method is called probably
        # isn't very efficient. TODO maybe create an attribute with it in "environment".
        available_values = [value for value in list(range(0, environment.n)) if value not in chromosome]
        index_1 = random.choice(range(0, len(chromosome)))
        index_2 = random.choice(range(0, len(available_values)))

        chromosome[index_1] = available_values[index_2]
        
        return chromosome
        

    @classmethod
    def mutate(self, chromosome, environment):
        function_name = environment.encoding + "_" + environment.mutation_method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome, environment)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))