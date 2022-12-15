import operator
import pdb
import numpy as np
import subprocess

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
    def remove_repeated_individuals(self, population):
        hashes = []
        individuals = []
        
        for individual in population:
            if individual.individual_hash not in hashes:
                individuals.append(individual)
                hashes.append(individual.individual_hash)
                
        return individuals

    @classmethod
    def get_n_best_individuals_without_repetition(self, population, number_of_individuals):
        population = Utils.sort_population(population)
        hashes = []
        individuals = []
        
        for individual in population:
            if individual.individual_hash not in hashes:
                individuals.append(individual)
                hashes.append(individual.individual_hash)

            if len(individuals) == number_of_individuals:
                break

        return individuals

    @classmethod
    def get_path_relinking_candidates(self, population, number_of_individuals):
        return self.get_n_best_individuals_without_repetition(population, number_of_individuals)
        

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

    # method based on the answer of the user "Yuji 'Tomita' Tomita" at 
    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    @classmethod
    def get_git_hash(self):
        hash_string = ""
        try:
            hash_string = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            hash_string = "Unable to retrieve the commit hash."
        return hash_string