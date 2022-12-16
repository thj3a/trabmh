from copy import deepcopy
import numpy as np
import time
from utils import Utils


class Local_Search:
    def __init__(self,):
        pass

    @classmethod
    def binary_LSFI(self, A,n,x_init,z_lb, max_time): # Local Search First Improvement
        start_time = time.time()
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
                            if time.time() - start_time > max_time:
                                break
                            else:
                                x[j] = 0
                    if flag:
                        break
                    else:
                        x[i] = 1
        return x, z_lb
    
    @classmethod
    def binary_LSFP(self, A,n,x_init,z_lb, max_time): # Local Search First Improvement Plus
        start_time = time.time()
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
                            if time.time() - start_time > max_time: # check this 
                                flag = True
                                break
                            x[j] = 0
                    if flag:
                        break
                    else:
                        x[i] = 1
            if flag:
                x[leave_x] = 0
                x[enter_x] = 1
        return x, z_lb
    
    @classmethod
    def binary_LSBI(self, A,n,x_init,z_lb, max_time): # Local Search Best Improvement
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
    def search(self, environment, chromosomes, max_time, best_sol):
        start_time = time.time()
        chromosomes = deepcopy([ind.chromosome for ind in chromosomes])
        if environment.encoding == "binary":
            x, sol = self.init_greedy(environment.A,environment.R,environment.s,environment.m,environment.n)
            if sol > best_sol:
                best_sol = sol
            chromosomes += x
            chromosomes += [self.init_binary(environment.A,environment.R,environment.s,environment.m,environment.n)]
        elif environment.encoding == "permutation":
            chromosomes += [Utils.convert_binary_to_permutation(self.init_greedy(environment.A,environment.R,environment.s,environment.m,environment.n))]
            chromosomes += [Utils.convert_binary_to_permutation(self.init_binary(environment.A,environment.R,environment.s,environment.m,environment.n))]

        local_search_method = environment.local_search_method
        function_name = environment.encoding + "_" + local_search_method
        
        if hasattr(self, function_name) and callable(getattr(self, function_name)):
            func = getattr(self, function_name)
            new_chromosomes = []
            new_solutions = []
            best_chromosome = None
            remaining_time = max_time - (time.time() - start_time)
            for ind in chromosomes:
                chromosome, solution = func(environment.A, environment.n, ind, best_sol, remaining_time)
                if solution > best_sol:
                    best_sol = solution
                    best_chromosome = deepcopy(chromosome)
                new_chromosomes += chromosome
                new_solutions += solution
                remaining_time = max_time - (time.time() - start_time)
                if remaining_time < 0:
                    break
            return new_chromosomes, new_solutions, best_chromosome, best_sol
        else:
            raise Exception("Method \"{}\" not found.".format(function_name))