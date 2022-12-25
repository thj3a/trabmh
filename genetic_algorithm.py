import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
from initialization import Initialization
from search import Search
from plot import Plot
from utils import Utils
import math
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from datetime import datetime, date

class GeneticAlgoritm:
    def __init__(self, experiment):
        self.experiment = experiment
        self.start_time = None
        self.experiment_id = experiment["experiment_id"]
        self.execution_id = experiment["execution_id"]
        self.instance = experiment["instance"]
        self.seed = experiment["seed"]
        self.silent = experiment["silent"]
        self.generate_plots = experiment["generate_plots"]
        self.max_generations = experiment["max_generations"]
        self.max_generations_without_change = experiment["max_generations_without_change"]
        self.max_time = experiment["max_time"]
        self.max_time_without_change = experiment["max_time_without_change"]
        self.population_size = experiment["population_size"]
        self.n = experiment["n"]
        self.m = experiment["m"]
        self.s = experiment["s"]
        self.A = experiment["A"]
        self.R = experiment["R"]
        self.encoding = experiment["encoding_method"]
        self.initialization_method = experiment["initialization_method"]
        self.selection_method = experiment["selection_method"]
        self.mutation_method = experiment["mutation_method"]
        self.self_mutation = experiment["self_mutation"]
        self.crossover_method = experiment["crossover_method"]
        self.parent_selection_method = experiment["parent_selection_method"]
        self.mutation_probability = experiment["mutation_probability"]
        self.crossover_probability = experiment["crossover_probability"]
        self.perform_assexual_crossover = experiment["perform_assexual_crossover"]
        self.best_sol = -np.inf
        self.best_known_result = experiment["best_known_result"]
        self.elite_size = math.ceil(self.population_size * experiment["elite_size"])
        # offspring_size is a limit on how many children are going going to be generated.
        # Ultimately, it is the crossover_probability that will control if a mating attempt
        # will be successful or not. For that reason, offspring_size shouldn't be confused 
        # with a target number.
        # In other words, if crossover_probability = 1.0, offspring_size will be the number of
        # individuals to be generated. However, if crossover_probability < 1.0, the number of
        # individuals generated will probably be less than offspring_size.
        self.offspring_size = math.floor(self.population_size * experiment["offspring_size"])
        self.commoners_size = self.population_size - self.elite_size
        self.best_sol_tracking = []
        self.best_sol_changes = []
        self.best_sol_change_times = []
        self.best_sol_change_generations = []
        self.best_sol_individual = None
        self.plots_dir = experiment["plots_dir"]
        self.perform_path_relinking = experiment["perform_path_relinking"]
        self.stop_message = "Maximum number of iterations reached"
        self.generations_ran = 0
        # Avoids repeated individuals in the population.
        # This option can be set by the user and, in the future, 
        # might be employed by adaptive methods to bring more
        # diversification to the population.
        self.avoid_clones = experiment["avoid_clones"] 
        self.time_until_adapt = experiment["time_until_adapt"]
        self.generations_until_adapt = experiment["generations_until_adapt"]
        self.perform_lagrangian = experiment["perform_lagrangian"]
        self.perform_adaptation = experiment["perform_adaptation"]
        self.total_time = None
        self.time_since_last_change = None
        self.generations_since_last_change = None
        self.local_search_method = experiment["local_search_method"]
        self.max_time_local_search = experiment["max_time_local_search"]
        self.perform_local_search = experiment["perform_local_search"]
        
        if self.seed == 0:
            self.seed = random.randint(0, 1000000)

        random.seed(self.seed)
        np.random.seed(self.seed)

    def loop(self):
        self.start_time = time.time()

        # generates initial population 
        population, self.best_sol, self.best_sol_individual = Initialization.initialize_population(self, self.population_size)
        self.best_sol_tracking.append(self.best_sol)
        self.best_sol_changes.append(self.best_sol)
        self.best_sol_change_times.append(time.time())
        self.best_sol_change_generations.append(1)
        elite = None
        pr_candidates = None
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

            # complement population if necessary (mainly as a result of changes in the selection method)
            if len(population) != self.population_size:
                population, self.best_sol = self.complement_population(population, self.best_sol)

            # Sorts the population. It is required because some selection methods rely on it
            population = Utils.sort_population(population)

            # generates children 
            children = []
            children_generated = 0

            while children_generated < self.offspring_size:
                if random.uniform(0,1) < self.crossover_probability:
                    # Selects parents
                    parents = Selection.select(population, 2, self.parent_selection_method)
                    offspring = parents[0].breed(parents[1])
                    children = children + list(offspring)
                
                    if self.encoding == "binary":
                        # Usually, each successful attempt generates 2 children. For that reason,
                        # each unsuccessful attempt will also count as 2 attempts.
                        children_generated += 2
                    elif self.encoding == "permutation": 
                        # Permutations usually produce only one child.
                        children_generated += 1

                # Generates children in an asexual way, with probability self.mutation_probability 
                # of happening. The way it is executed right now might lead to offspring_size + 1 
                # individuals.
                if random.uniform(0,1) < self.mutation_probability:
                    an_individual = Selection.select(population, 1, self.parent_selection_method)
                    mutant = an_individual[0].mutate(self.self_mutation)
                    children = children + [mutant]
                    children_generated += 1

            self.log("Children produced", children)

            # Updates the population
            population = population + children # + mutants

            if self.avoid_clones:
                population = Utils.remove_repeated_individuals(population)

            self.log("Candidate population", population)

            # Sorts the population. It is required because some selection methods 
            # and some other things rely on it
            population = Utils.sort_population(population)

            # updates de best solution if necessary
            if population[0].fitness > self.best_sol:
                self.best_sol = population[0].fitness
                self.best_sol_individual = population[0]
                self.best_sol_changes.append(self.best_sol)
                self.best_sol_change_times.append(time.time())
                self.best_sol_change_generations.append(generation + 1)
            # Track the best solution found so far
            self.best_sol_tracking.append(self.best_sol)

            # selects individuals for the next generation
            elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
            commoners = Selection.select(commoners, self.commoners_size, self.selection_method)
            population = elite + commoners
            self.log("Population after elitism and selection", population)
            
            # Calculates atrributes related to stopping criteria and adaptation
            self.compute_stop_and_adaptation_attributes(generation)

            if self.perform_adaptation: 
                self.adapt()

            self.generations_ran += 1

            # Checks stopping criteria and stops accordingly
            if self.stop_execution(generation):
                # TODO Add call to reset the experiment parameters if adaptation is active.
                break

        # ===========================================================================================
        
        # main loop solution
        finish_time = time.time()
        ga_best_individual = self.best_sol_individual
        ga_best_individual_gap = Utils.calculate_gap(self.best_sol, self.best_known_result) # here best_sol still holds the 1st elite fitness

        # stuff relevant for post processing
        improvement_candidates = Utils.get_improvement_candidates(population, self.elite_size)
        improvement_candidates = Utils.sort_population(improvement_candidates) # possibly unnecessary, but used to ensure the ordering.

        results = self.refine(improvement_candidates, generation)

        best_sol_gap = Utils.calculate_gap(self.best_sol, self.best_known_result)
        print(f"Experiment {self.experiment_id} finished - Instance name: {self.instance} - Result {self.best_sol} - Best Known Result {self.best_known_result} - Gap {best_sol_gap} - Time {time.time() - self.start_time}.")

        results.update({
            # misc stuff
            "num_generations": self.generations_ran,
            "seed": self.seed,
            # sol changes stuff
            "best_sol_changes": self.best_sol_changes,
            "best_sol_change_times": [str(element - self.start_time) for element in self.best_sol_change_times],
            "best_sol_change_generations": self.best_sol_change_generations,
            "best_solution_found": self.best_sol,
            "gap": best_sol_gap,
            "best_solution_hash": Utils.convert_individuals_hashes_to_string([self.best_sol_individual]),
            "raw_best_solution": Utils.convert_individuals_binary_chromosomes_to_string([self.best_sol_individual]),
            
            # main loop stuff
            "ga_best_sol": ga_best_individual.fitness,
            "ga_best_sol_gap": ga_best_individual_gap,
            "loop_start_time": self.start_time,
            "loop_finish_time": finish_time, 
            "ga_total_time": finish_time - self.start_time,
            "elite_fitness": [individual.fitness for individual in elite],
            "elite_hashes": Utils.convert_individuals_hashes_to_string(elite)
            
        })


        # Draw the plots.
        Plot.draw_plots(self)
        self.save_solution_times()

        # Updates the elite set.
        return results, self.stop_message #sol_changes, self.start_time, improvement_candidates, pr_individuals, pr_sol_values, ls_individuals, ls_sol_values


    def refine(self, improvement_candidates, current_generation):
        pr_individuals = []
        pr_sol_values = []
        pr_best_sol = ""
        pr_best_sol_gap = ""
        improved_by_pr = False
        pr_total_time = 0.0
        ls_individuals = []
        ls_sol_values = []
        ls_best_sol = ""
        ls_best_sol_gap = ""
        ls_total_time = 0.0
        improved_by_ls = False
        

        # + 1 to account for the fact that the counting starts from 0
        refinement_generation = current_generation + 1

        # Performs path relinking and local search
        if self.perform_path_relinking:
            start_time = time.time()
            pr_individuals = self.path_relinking(improvement_candidates)
            pr_total_time = time.time() - start_time

            pr_individuals = Utils.sort_population(pr_individuals)
            pr_sol_values = [individual.fitness for individual in pr_individuals]

            if len(pr_individuals) > 0:
                if pr_individuals[0].fitness > self.best_sol:
                    improved_by_pr = True
                    refinement_generation += 1

                    self.best_sol = pr_individuals[0].fitness
                    self.best_sol_individual = pr_individuals[0]
                    self.best_sol_changes.append(self.best_sol)
                    self.best_sol_change_times.append(time.time())
                    self.best_sol_change_generations.append(refinement_generation)

                pr_best_sol = pr_individuals[0].fitness
                pr_best_sol_gap = Utils.calculate_gap(pr_best_sol, self.best_known_result)
        
        if self.perform_local_search:
            start_time = time.time()
            ls_individuals = Search.local_search(
                self, 
                improvement_candidates, 
                self.max_time_local_search, 
                self.best_sol, 
                self.local_search_method
            )
            ls_total_time = time.time() - start_time

            ls_individuals = Utils.sort_population(ls_individuals)
            ls_sol_values = [individual.fitness for individual in ls_individuals]

            if len(ls_individuals) > 0:
                if ls_individuals[0].fitness > self.best_sol:
                    improved_by_ls = True
                    refinement_generation += 1

                    self.best_sol = ls_individuals[0].fitness
                    self.best_sol_individual = ls_individuals[0]
                    self.best_sol_changes.append(self.best_sol)
                    self.best_sol_change_times.append(time.time())
                    self.best_sol_change_generations.append(refinement_generation)

                ls_best_sol = ls_individuals[0].fitness
                ls_best_sol_gap = Utils.calculate_gap(ls_best_sol, self.best_known_result)

        extra_results = {
            # refinement stuff 
            #"pr_individuals": pr_individuals, 
            "pr_sol_values": pr_sol_values, 
            "pr_improved_solution": improved_by_pr,
            "pr_best_sol": pr_best_sol,
            "pr_best_sol_gap": pr_best_sol_gap,
            "pr_total_time": pr_total_time,
            #"ls_individuals": ls_individuals, 
            "ls_sol_values": ls_sol_values,
            "ls_improved_solution": improved_by_ls,
            "ls_best_sol": ls_best_sol,
            "ls_best_sol_gap": ls_best_sol_gap,
            "ls_total_time": ls_total_time,
            "improvement_candidates_hashes": Utils.convert_individuals_hashes_to_string(improvement_candidates),
            "pr_individuals_hashes": Utils.convert_individuals_hashes_to_string(pr_individuals),
            "ls_individuals_hashes": Utils.convert_individuals_hashes_to_string(ls_individuals)
        }

        return extra_results


    def save_solution_times(self):
        if not self.generate_plots:
            return 

        file = os.path.join(self.plots_dir, "solution_times.csv")
        gaps = None

        if math.isinf(self.best_known_result):
            gaps = ["" for i in range(len(self.best_sol_changes))]
        else:
            gaps = [(-self.best_sol_changes[i] + self.best_known_result) / abs(self.best_known_result) for i in range(len(self.best_sol_changes))]

        df = pd.DataFrame({
            "best_sol_changes": self.best_sol_changes,
            "best_sol_change_times": self.best_sol_change_times,
            "best_sol_change_generations": self.best_sol_change_generations,
            "gaps": gaps
        })
        df["best_sol_change_times"] = df["best_sol_change_times"] - self.start_time
        df.to_csv(file, sep=";")     

    def compute_stop_and_adaptation_attributes(self, current_generation):
        self.total_time = time.time() - self.start_time 
        self.time_since_last_change = time.time() - self.best_sol_change_times[-1]
        self.generations_since_last_change = current_generation + 1 - self.best_sol_change_generations[-1]

    def stop_execution(self, current_generation):

        if current_generation == self.max_generations - 1:
            self.stop_message ="Maximum number of generations reached."

        if self.total_time >= self.max_time and self.max_time > 0:
            self.stop_message = "Maximum time reached"
            return True

        if self.time_since_last_change >= self.max_time_without_change and self.max_time_without_change > 0:
            self.stop_message = "Maximum time without improvement reached."
            return True

        if self.generations_since_last_change >= self.max_generations_without_change and self.max_generations_without_change > 0:
            self.stop_message = "Maximum number of generations without improvement reached"
            return True
        
            
        return False

    def adapt(self):
        if self.time_since_last_change > self.time_until_adapt or self.generations_since_last_change > self.generations_until_adapt:
                self.adapt_parameters()

        elif len(self.best_sol_tracking) > 3 and self.best_sol_tracking[-1] > self.best_sol_tracking[-2] and self.best_sol_tracking[-2] == self.best_sol_tracking[-3]:
            self.reset_parameters()

    def complement_population(self, population, best_sol):  
        if len(population) != self.population_size:
            if len(population) < self.population_size:
                new_individuals, highest_new_sol, _ = Initialization.initialize_population(self, self.population_size - len(population))
                if highest_new_sol > best_sol:
                    best_sol = highest_new_sol
                population = population + new_individuals
        return population, best_sol

    def adapt_parameters(self,):
        # parameters to be adapted in that order:
        # n individuos elite - done
        # alteração do tamanho da população?
        # parâmetros dos métodos de seleção/crossover/mutação
        # alteração dos próprios métodos para métodos mais aleatórios/diversificadores
        # em ultimo caso métodos de reinicio de população
        if self.elite_size > math.ceil(self.population_size * self.experiment["elite_size"]/3):
            self.elite_size -= 1
        elif self.population_size < 2*self.experiment["population_size"]:
            if self.population_size == self.experiment["population_size"]:
                self.population_size += math.floor(self.experiment["population_size"] * 0.1)
                self.elite_size = math.ceil(self.population_size * self.experiment["elite_size"])
                self.offspring_size = math.ceil(self.population_size * self.experiment["offspring_size"])
                self.commoners_size = self.population_size - self.elite_size
            else:
                self.population_size += math.floor(self.experiment["population_size"] * 0.1)
        elif self.selection_method != "nbestdifferent":
            self.selection_method = "nbestdifferent"

    def reset_parameters(self,):
        self.population_size = self.experiment["population_size"]
        self.elite_size = math.ceil(self.population_size * self.experiment["elite_size"])
        self.offspring_size = math.floor(self.population_size * self.experiment["offspring_size"])
        self.commoners_size = self.population_size - self.elite_size
        self.selection_method = self.experiment["selection_method"]
        self.mutation_method = self.experiment["mutation_method"]
        self.self_mutation = self.experiment["self_mutation"]
        self.crossover_method = self.experiment["crossover_method"]
        self.parent_selection_method = self.experiment["parent_selection_method"]
        self.mutation_probability = self.experiment["mutation_probability"]
        self.crossover_probability = self.experiment["crossover_probability"]

    def path_relinking(self, population):
        if len(population) < 2:
            return [], []

        new_individuals = []
        unique_individuals , indices = np.unique([ind.individual_hash for ind in population], return_index=True)
        unique_individuals = [population[index] for index in indices]
        for individual in unique_individuals:
            for other_individual in unique_individuals:
                diff = np.where(individual.binary_chromosome != other_individual.binary_chromosome)[0]
                if len(diff) > 0:
                    for i in range(0, len(diff)):
                        # forward path relinking
                        new_chromosome = individual.binary_chromosome.copy()
                        new_chromosome[diff[i]] = other_individual.binary_chromosome[diff[i]]
                        new_individual = Individual(new_chromosome, individual.environment)
                        print(sum(new_individual.binary_chromosome.T[0]), new_individual.binary_chromosome.T[0])
                        if new_individual.fitness >= self.best_sol:
                            new_individuals.append(new_individual)
        return new_individuals
    
    def log(self, message, additional_content = "", status = "INFO"):
        if self.silent:
            return

        prepared_message = "{}" + message + "\033[0m"
        if status == "INFO":
            print(prepared_message.format("\033[96m"), additional_content)
        elif status == "ERROR":
            print(prepared_message.format("\033[91m"), additional_content)
        else: # warning
            print(prepared_message.format("\033[93m"), additional_content)