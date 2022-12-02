import random
import numpy as np
import copy

class Crossover:
    def __init__(self):
        pass
    
    # could be converted to k-point crossover later on
    @classmethod
    def binary_singlepoint(self, chromosome_1, chromosome_2, environment):
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
    def binary_uniform(self, chromosome_1, chromosome_2, environment):
        return None

    @classmethod
    def binary_misc(self, chromosome_1, chromosome_2, environment):
        ones = np.unique(np.concatenate([np.where(chromosome_1 == 1)[0], np.where(chromosome_2 == 1)[0]]))

        offspring_1 = np.zeros((len(chromosome_1), 1))
        offspring_2 = np.zeros((len(chromosome_1), 1))
        
        offspring_1[ones[random.sample(range(len(ones)), k=environment.s)]] = 1
        offspring_2[ones[random.sample(range(len(ones)), k=environment.s)]] = 1
    
        return offspring_1, offspring_2

    @classmethod
    def crossover(self, chromosome_1, chromosome_2, environment):
        function_name = environment.encoding + "_" + environment.crossover_method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome_1, chromosome_2, environment)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

