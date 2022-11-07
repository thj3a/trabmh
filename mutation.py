import random

class Mutation:
    def __init__(self):
        pass

    @classmethod
    def binary_singlepoint(self, chromosome):
        point = random.choice(range(0, len(chromosome)))
        chromosome[point] = 1 - chromosome[point]
        return chromosome
    
    @classmethod
    def mutate(self, chromosome, encoding, method):
        function_name = encoding + "_" + method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))