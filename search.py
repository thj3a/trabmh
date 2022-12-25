from copy import deepcopy
import numpy as np
import time
from utils import Utils
from individual import Individual


class Search:
    def __init__(self,):
        pass

    @classmethod
    def LSFI(self, A,n,x_init,z_lb, max_time): # Local Search First Improvement
        start_time = time.time()
        x = deepcopy(x_init)
        flag, timeout = True, False
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
                            if time.time() - start_time > max_time:
                                timeout = True
                                print("LSFI time limit reached")
                                break
                            else:
                                x[j] = 0
                    if flag or timeout:
                        break
                    else:
                        x[i] = 1
            if timeout:
                break
        return x, z_lb
    
    @classmethod
    def LSFP(self, A,n,x_init,z_lb, max_time): # Local Search First Improvement Plus
        start_time = time.time()
        x = deepcopy(x_init)
        flag, timeout = True, False
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
                            if time.time() - start_time > max_time:
                                timeout = True
                                print("LSFP time limit reached")
                                break
                            x[j] = 0
                    if flag or timeout:
                        break
                    else:
                        x[i] = 1
            if flag:
                # x[leave_x] = 0  # is that it ?
                x[enter_x] = 1
            if timeout:
                break
        return x, z_lb
    
    @classmethod
    def LSBI(self, A,n,x_init,z_lb, max_time): # Local Search Best Improvement
        start_time = time.time()
        x = deepcopy(x_init)
        flag, timeout = True, False
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
                            if time.time() - start_time > max_time:
                                timeout = True
                                print("LSBI time limit reached")
                                break
                            x[j] = 0
                    if timeout:
                        break
                    x[i] = 1
            if flag:
                x[leave_x] = 0
                x[enter_x] = 1
            if timeout:
                break
        return x, z_lb
    
    @classmethod
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

    @classmethod
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
    
    @classmethod
    def ldet_objval(self, A, x):
        sign, value = np.linalg.slogdet(np.dot(np.dot(A.T, np.diag(x.T[0])), A))
        if sign > 0:
            return value
        else:
            return -np.inf

    @classmethod
    def heuristic_solutions(self, environment):
        init_binary_solution, _ = self.init_binary(environment.A, environment.R, environment.s, environment.m, environment.n)
        init_greedy_solution, _ = self.init_greedy(environment.A, environment.R, environment.s, environment.m, environment.n)
        return [init_binary_solution, init_greedy_solution]

    @classmethod
    def local_search(self, environment, individuals, max_time, best_sol, local_search_method):
        start_time = time.time()
        chromosomes = [deepcopy(ind.binary_chromosome) for ind in individuals]
        # x_g, sol_g = self.init_greedy(environment.A,environment.R,environment.s,environment.m,environment.n)
        # x_b, sol_b = self.init_binary(environment.A,environment.R,environment.s,environment.m,environment.n)
        # if max(sol_g, sol_b) > best_sol:
        #     best_sol = max(sol_g, sol_b)
        # if environment.encoding == "binary":
        #     chromosomes += [x_g, x_b]
        # elif environment.encoding == "permutation":
        #     chromosomes += [Utils.convert_binary_to_permutation(x_g), Utils.convert_binary_to_permutation(x_b)]

        function_name = local_search_method
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            new_chromosomes = []
            #new_solutions = []
            new_individuals = []
            remaining_time = max_time - (time.time() - start_time)
            for chromosome in chromosomes:
                new_chromosome, solution = func(environment.A, environment.n, chromosome, best_sol, remaining_time)
                new_chromosomes += [new_chromosome]
                #new_solutions += [solution]
                remaining_time = max_time - (time.time() - start_time)
                if remaining_time < 0:
                    break
            
            if environment.encoding == "permutation":
                new_chromosomes = [Utils.convert_binary_to_permutation(chromosome) for chromosome in new_chromosomes]

            for chromosome in new_chromosomes:
                new_individuals.append(Individual(chromosome, environment))

            return new_individuals#, new_solutions
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))

    