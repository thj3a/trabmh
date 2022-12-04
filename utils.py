import operator
import pdb

class Utils:
    def __init__(self):
        pass

    @classmethod
    def sort_population(self, population):
        return sorted(population, key=operator.attrgetter("fitness"), reverse=True)

    # population must be sorted
    @classmethod
    def split_elite_commoners(self, population, elite_size):
        return population[:elite_size], population[elite_size:]

    @classmethod
    def get_best_solution(self, population, best_sol):
        best_population_fitness = max([i.fitness for i in population])
        if best_population_fitness > best_sol:
            return best_population_fitness
        else:
            return best_sol