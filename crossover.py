class Crossover:
    def __init__(self):
        pass

    def crossover(self, chromosome_1, chromosome_2, encoding, method):
        function_name = encoding + "_" + method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome_1, chromosome_2, random_generator)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

