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
    def permutation_opx(self, chromosome_1, chromosome_2, environment):
        cut_position = random.choice(range(0, len(chromosome_1)))
        offspring = None

        if cut_position == 0:
            offspring = copy.deepcopy(chromosome_2)
        else:
            chromosome_1_part = chromosome_1[:cut_position]
            chromosome_2_part = []

            # if len(chromosome_1) = 5 and cut_position = 2, then
            # chromosome_1_part's range will be [0,1], while 
            # chromosome_2_part's range will be [2,3,4], whose size
            # is 3, that can be achieved by 5 - 2 = 3, that is calculated
            # by the following line.
            remaining_positions = len(chromosome_1) - cut_position

            for value in chromosome_2:
                if value not in chromosome_1_part:
                    chromosome_2_part.append(value)

                    remaining_positions -= 1
                    if remaining_positions == 0:
                        break
            #print(cut_position, chromosome_1_part, chromosome_2_part, np.array(chromosome_2_part))
            offspring = np.concatenate([chromosome_1_part, np.array(chromosome_2_part)])
        
        return offspring, None




    @classmethod
    def crossover(self, chromosome_1, chromosome_2, environment):
        chromosome_1 = copy.deepcopy(chromosome_1)
        chromosome_2 = copy.deepcopy(chromosome_2)
        function_name = environment.encoding + "_" + environment.crossover_method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome_1, chromosome_2, environment)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

