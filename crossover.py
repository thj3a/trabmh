import random
import numpy as np
import copy

class Crossover:
    def __init__(self):
        pass
    
    # could be converted to k-point crossover later on
    @classmethod
    def binary_singlepoint(self, chromosome_1, chromosome_2):
        cut_position = random.choice(range(0, len(chromosome_1)))
        if cut_position == 0:
            offspring_1 = copy.deepcopy(chromosome_1)
            offspring_2 = copy.deepcopy(chromosome_2)
        else:
            offspring_1 = np.concatenate([chromosome_1[:cut_position], chromosome_2[cut_position:]])
            offspring_2 = np.concatenate([chromosome_2[:cut_position], chromosome_1[cut_position:]])
        return offspring_1, offspring_2

    # uniform crossover 
    @classmethod
    def binary_uniform(self, chromosome_1, chromosome_2):
        return None


    @classmethod
    def crossover(self, chromosome_1, chromosome_2, encoding, method):
        function_name = encoding + "_" + method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome_1, chromosome_2)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

