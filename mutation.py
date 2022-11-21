import random
import numpy as np
import copy
class Mutation:
    def __init__(self):
        pass

    #must use fitness function similar to lagrangian function
    @classmethod
    def binary_singlepoint(self, chromosome):
        point = random.choice(range(0, len(chromosome)))
        chromosome[point] = 1 - chromosome[point]
        return chromosome
    
    @classmethod
    def binary_singlepoint_interchange(self, chromosome):
        ones = np.where(chromosome == 1)[0]
        zeros = np.where(chromosome == 0)[0]
        flip_point_0, flip_point_1  = random.choice(ones), random.choice(zeros)
        chromosome[flip_point_0] = 0
        chromosome[flip_point_1] = 1
        return chromosome

    @classmethod
    def mutate(self, chromosome, encoding, method):
        function_name = encoding + "_" + method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))