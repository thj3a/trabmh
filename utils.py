import operator

class Utils:
    def __init__(self):
        pass

    @classmethod
    def sort_population(self, population):
        return sorted(population, key=operator.attrgetter("fitness"), reverse=True)

    # poplation must be sorted
    @classmethod
    def split_elite_commoners(self, population, elise_size):
        return population[:elise_size], population[elise_size:]