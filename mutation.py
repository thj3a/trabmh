class Mutation:
    def __init__(self):
        pass
    
    @classmethod
    def mutate(self, chromosome, encoding, method):
        function_name = encoding + "_" + method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            return func(chromosome, random_generator)
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))