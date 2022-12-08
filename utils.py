import operator
import pdb
import numpy as np

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
    def get_path_relinking_candidates(self, population, number_of_individuals):
        hashes = []
        pr_list = []
        
        for individual in population:
            if individual.individual_hash not in hashes:
                pr_list.append(individual)
                hashes.append(individual.individual_hash)

            if len(pr_list) == number_of_individuals:
                break

        return pr_list

    @classmethod
    def get_best_solution(self, population, best_sol):
        best_population_fitness = max([i.fitness for i in population])
        if best_population_fitness > best_sol:
            return best_population_fitness
        else:
            return best_sol

    @classmethod
    def convert_binary_to_permutation(self, sequence):
        return np.array([np.where(sequence == 1)[0]]).T
    
    @classmethod
    def convert_chromosomes_from_binary_to_permutation(self, chromosomes):
        return [self.convert_binary_to_permutation(chromosome) for chromosome in chromosomes]

    @classmethod
    def convert_permutation_to_binary(self, sequence, expected_size):
        new_sequence = np.zeros((expected_size, 1))
        new_sequence[sequence] = 1
        return new_sequence
    
    @classmethod
    def convert_chromosomes_from_permutation_to_binary(self, chromosomes, expected_size):
        return [self.convert_permutation_to_binary(chromosome, expected_size) for chromosome in chromosomes]