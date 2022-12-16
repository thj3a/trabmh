import numpy as np
import random
from individual import Individual
from utils import Utils
import time
import pdb 
from copy import deepcopy

class Initialization:

    def __init__(self):
        pass

    @classmethod
    def binary_random(self, environment, population_size): # uniform random
        chromosomes = [] 

        for i in range(0, population_size):
            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
            chromosomes.append(chromosome)
        return chromosomes

    @classmethod
    def binary_biased(self, environment, population_size, min_diff):
        min_diff = float(min_diff)
        chromosomes = []

        while len(chromosomes) < population_size:    
            chromosome = np.zeros((environment.n, 1))
            chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
            
            found_close = False
            for existing_chromosome in chromosomes:
                if np.mean(np.abs(existing_chromosome - chromosome)) < min_diff:
                    found_close = True
                    break
            if not found_close:
                chromosomes.append(chromosome)
        return chromosomes

    @classmethod
    def binary_biasedweighted(self, environment, population_size): # can it be called uniform?
        chromosomes = []
        chromosome = np.zeros((environment.n, 1))
        chromosome[random.sample(range(0, environment.n), k = environment.s)] = 1
        chromosomes.append(chromosome)
        while len(chromosomes) < population_size:    
            sum = 1/(np.sum(chromosomes, axis=0)+1)
            p = sum.T[0]/sum.sum()
            chromosome = np.zeros((environment.n, 1))
            chromosome[np.random.choice(range(0, environment.n), p=p, size=environment.s, replace=False)] = 1
            chromosomes.append(chromosome)
        return chromosomes

    @classmethod
    def permutation_random(self, environment, population_size):
        chromosomes = self.binary_random(environment, population_size)
        return Utils.convert_chromosomes_from_binary_to_permutation(chromosomes)

    @classmethod
    def permutation_biased(self, environment, population_size, min_diff):
        chromosomes = self.binary_biased(environment, population_size, min_diff)
        return Utils.convert_chromosomes_from_binary_to_permutation(chromosomes)

    @classmethod
    def Gabriel_heuristic_local_search(self, environment, population_size):
        # TODO Add the option to produce initial solutions using 
        # Gabriel's heuristics and local search.
        pass

    def LSFI(self, A,n,x_init,z_lb): # Local Search First Improvement
        x = deepcopy(x_init)
        flag = True
        while flag:
            flag = False
            for i in range(n):
                if x[i] > 0:
                    x[i] = 0
                    for j in range(n):
                        if j != i and x[j] == 0:
                            x[j] = 1
                            z_lb_new = self.ldet_objval(A, x)
                            if z_lb_new > z_lb:
                                z_lb = z_lb_new
                                flag = True
                                break
                            else:
                                x[j] = 0
                    if flag:
                        break
                    else:
                        x[i] = 1
        return x, z_lb
    
    def LSFP(self, A,n,x_init,z_lb): # Local Search First Improvement Plus
        x = deepcopy(x_init)
        flag = True
        leave_x, enter_x = 0, 0
        while flag:
            flag = False
            for i in range(n):
                if x[i] > 0:
                    x[i] = 0
                    for j in range(n):
                        if j != i and x[j] == 0:
                            x[j] = 1
                            z_lb_new = self.ldet_objval(A, x)
                            if z_lb_new > z_lb:
                                leave_x, enter_x = i, j
                                z_lb = z_lb_new
                                flag = True
                            x[j] = 0
                    if flag:
                        break
                    else:
                        x[i] = 1
            if flag:
                x[leave_x] = 0
                x[enter_x] = 1
        return x, z_lb
    
    def LSBI(self, A,n,x_init,z_lb): # Local Search Best Improvement
        x = deepcopy(x_init)
        flag = True
        leave_x, enter_x = 0, 0
        while flag:
            flag = False
            for i in range(n):
                if x[i] > 0:
                    x[i] = 0
                    for j in range(n):
                        if j != i and x[j] == 0:
                            x[j] = 1
                            z_lb_new = self.ldet_objval(A, x)
                            if z_lb_new > z_lb:
                                leave_x, enter_x = i, j
                                z_lb = z_lb_new
                                flag = True
                            x[j] = 0
                    x[i] = 1
            if flag:
                x[leave_x] = 0
                x[enter_x] = 1
        return x, z_lb
    
    def init_binary(self, A,R,s,m,n):
        U, S, VH = np.linalg.svd(A, full_matrices=True)
        x = np.zeros((n,1))
        for j in range(n):
            for i in range(s):
                x[j] += (U[j,i]**2)
        x_save = deepcopy(x)
        x = np.zeros((n,1))
        for row in R:
            x[row-1] = 1
            x_save[row-1] = 0

        for i in range(s-m):
            max_indice = np.argmax(x_save)
            x[max_indice] = 1
            x_save[max_indice] = 0
        zlb = self.ldet_objval(A, x)
        xlb = x
        return xlb, zlb

    def init_greedy(self, A,R,s,m,n):
        U, S, VH = np.linalg.svd(A, full_matrices=True)
        x = np.zeros((n,1))
        k = min(s,m)
        for j in range(n):
            for i in range(k):
                x[j] += (S[i] * U[j,i]**2)
        x_save = deepcopy(x)
        x = np.zeros((n,1))
        for row in R:
            x[row-1] = 1
            x_save[row-1] = 0

        for i in range(s-m):
            max_indice = np.argmax(x_save)
            x[max_indice] = 1
            x_save[max_indice] = 0

        zlb = self.ldet_objval(A, x)
        xlb = x
        return xlb, zlb
    
    def ldet_objval(A, x):
        sign, value = np.linalg.slogdet(np.dot(np.dot(A.T, np.diag(x.T[0])), A))
        if sign > 0:
            return value
        else:
            return -np.inf

    @classmethod
    def initialize_population(self, environment, population_size):
        function_and_params = environment.initialization_method.split("_")
        function_name = function_and_params[0]
        params = function_and_params[1:] if len(function_and_params) > 0 else []
        function_name = environment.encoding + "_" + function_name
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            population_chromosomes = func(environment, population_size, *params)
            population = []
            best_sol = - np.inf

            for chromosome in population_chromosomes:
                individual = Individual(
                    chromosome,
                    environment
                )
                if individual.fitness > best_sol:
                    best_sol = individual.fitness
                population.append(individual)

            return population, best_sol
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))