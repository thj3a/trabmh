import random
import numpy as np
import copy
import math

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

    # Variation of the singlepointinterchange that mutates several genes.
    # It requires the solution to be feasible (chromosome with "environment.s" ones).
    @classmethod
    def binary_percentchange(self, chromosome, environment, percent):
        ones = np.where(chromosome == 1)[0]
        zeros = np.where(chromosome == 0)[0]
        #print(ones, type(ones))

        percent = float(percent)
        num_changes = math.ceil(percent * environment.s) # uses ceil to guarantee at least one change
        ones_to_change = random.sample(list(ones), k = num_changes)
        zeros_to_change = random.sample(list(zeros), k = num_changes)

        chromosome[ones_to_change] = 0
        chromosome[zeros_to_change] = 1

        return chromosome

    # Requires the solution to be feasible (chromosome of size "environment.s" and without repetition of value).
    @classmethod
    def permutation_percentchange(self, chromosome, environment, percent):
        ones = chromosome.T[0]
        zeros = np.array([i for i in range(environment.n) if i not in ones])

        percent = float(percent)
        num_changes = math.ceil(percent * environment.s) # uses ceil to guarantee at least one change
        #print(num_changes, type(ones), list(ones))
        ones_to_change = random.sample(list(ones), k = num_changes)
        zeros_to_change = random.sample(list(zeros), k = num_changes)
        #print(ones, ones_to_change, zeros, zeros_to_change)

        changes = 0
        for i in range(0, len(chromosome)):
            if chromosome[i] in ones_to_change:
                chromosome[i] = zeros_to_change[changes]
                changes +=1

                if changes == len(zeros_to_change):
                    break

        return chromosome

    @classmethod
    def variable_percent_change(self, chromosome, environment, min_percent, max_percent):
        percent = None

        # TODO Add logic to allow the "environment" to control the distribution of the values
        # in a context of an adaptive approach
        # if environment.something == something_else:
        #     percent = a value between min_percent and max_percent according to a non uniform distribution
        # else
        percent = random.uniform(float(min_percent), float(max_percent))

        if environment.encoding == "binary":
            return self.binary_percentchange(chromosome, environment, percent)
        elif environment.encoding == "permutation":
            return self.permutation_percentchange(chromosome, environment, percent)

    @classmethod
    def binary_variablepercentchange(self, chromosome, environment, min_percent, max_percent):
        return self.variable_percent_change(chromosome, environment, min_percent, max_percent)

    @classmethod
    def permutation_variablepercentchange(self, chromosome, environment, min_percent, max_percent):
        return self.variable_percent_change(chromosome, environment, min_percent, max_percent)

    

    

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
        chromosome = copy.deepcopy(chromosome)
        method_and_params = environment.mutation_method.split("_")
        mutation_method = method_and_params[0]
        params = method_and_params[1:] if len(method_and_params) > 0 else []

        function_name = environment.encoding + "_" + mutation_method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome, environment, *params)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))