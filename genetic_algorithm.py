import random
from re import S
import time
import os
from selection import Selection
from individual import Individual
from initialization import Initialization
from utils import Utils
import math
import numpy as np
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
        self.best_sol_change_times = []
        self.best_sol_change_generations = []
        self.plots_dir = experiment["plots_dir"]
        self.perform_path_relinking = experiment["perform_path_relinking"]
        self.stop_message = "Maximum number of iterations reached"
        self.generations_ran = 0
        # Avoids repeated individuals in the population.
        # This option can be set by the user and, in the future, 
        # might be employed by adaptive methods to bring more
        # diversification to the population.
        self.avoid_clones = experiment["avoid_clones"] 
        self.max_time_to_adapt = experiment["max_time_to_adapt"]
        self.max_generations_to_adapt = experiment["max_generations_to_adapt"]
        self.perform_lagrangian = experiment["perform_lagrangian"]
        random.seed(self.seed)

    def loop(self):
        self.start_time = time.time()

        # generates initial population 
        population, self.best_sol = Initialization.initialize_population(self, self.population_size)
        self.best_sol_tracking.append(self.best_sol)
        self.best_sol_change_times.append(time.time())
        self.best_sol_change_generations.append(1)
        elite = None
        pr_candidates = None
        self.log("Population initialized", population)

        for generation in range(self.max_generations):
            self.log("Generation", generation + 1)

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
                    mutant = Selection.select(population, 1, self.parent_selection_method)
                    children = children + list(mutant)
                    children_generated += 1

            self.log("Children produced", children)

            # Children produced through mutation
            #mutants = []

            # Mutates the population, generating children in an assexual way
            # individuals that generate a mutant are not modified or removed
            # in this step.
            # Children created right before this step are not mutated in the
            # current generation.
            #for individual in population:
            #    if random.uniform(0,1) < self.mutation_probability:
            #        mutant = individual.mutate(self.self_mutation)
            #        if mutant is not None:
            #            mutants.append(mutant)

            # Updates the population
            population = population + children # + mutants

            if self.avoid_clones:
                population = Utils.remove_repeated_individuals(population)

            self.log("Candidate population", population)

            # updates the value of the best solution
            #self.best_sol = Utils.get_best_solution(population, self.best_sol)

            # Sorts the population. It is required because some selection methods 
            # and some other things rely on it
            population = Utils.sort_population(population)

            # updates de best solution if necessary
            if population[0].fitness > self.best_sol:
                self.best_sol = population[0].fitness
                self.best_sol_change_times.append(time.time())
                self.best_sol_change_generations.append(generation + 1)

            # selects individuals for the next generation
            elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
            commoners = Selection.select(commoners, self.commoners_size, self.selection_method)
            population = elite + commoners
            self.log("Population after elitism and selection", population)

            if self.perform_path_relinking:
                pr_candidates = Utils.get_path_relinking_candidates(population, self.elite_size)

            # Track the best solution found so far
            self.best_sol_tracking.append(self.best_sol)

            self.generations_ran += 1

            # checks stopping criteria
            if generation == self.max_generations - 1:
                self.log("Maximum number of generations reached.")

            # Checks stopping criteria and stops accordingly
            if self.stop_execution_or_adapt(generation):
                break

        # Performs path relinking.
        if self.perform_path_relinking:
            path_relinking_results = self.path_relinking(pr_candidates)
            if len(path_relinking_results) > 0:
                population += path_relinking_results
                elite, commoners = Utils.split_elite_commoners(population, self.elite_size)
        
        # Draw the plots.
        self.draw_plots()

        print(f"Experiment {self.experiment_id} finished.")

        # Updates the elite set.
        return elite, self.generations_ran, self.stop_message

    # TODO add a stop criterion based on the optimality gap.
    def stop_execution_or_adapt(self, current_generation):
        total_time = time.time() - self.start_time 
        time_since_last_change = time.time() - self.best_sol_change_times[-1]
        generations_since_last_change = current_generation + 1 - self.best_sol_change_generations[-1]

        if total_time >= self.max_time and self.max_time > 0:
            self.stop_message = "Maximum time reached"
            return True

        if time_since_last_change >= self.max_time_without_change and self.max_time_without_change > 0:
            self.stop_message = "Maximum time without improvement reached."
            return True

        if generations_since_last_change >= self.max_generations_without_change and self.max_generations_without_change > 0:
            self.stop_message = "Maximum number of generations without improvement reached"
            return True
        
        if time_since_last_change > self.max_time_to_adapt or generations_since_last_change > self.max_generations_to_adapt:
            sig_time = 1/(1+np.exp(-self.time_since_last_change))
            sig_gen = 1/(1+np.exp(-self.generations_since_last_change))
            self.adapt_parameters(max(sig_gen, sig_time))
        elif self.best_sol_tracking[-1] > self.best_sol_tracking[-2] and self.best_sol_tracking[-2] == self.best_sol_tracking[-3]:
            self.reset_parameters()
        return False

    def adapt_parameters(self, percent):
        if self.elite_size > 1:
            self.elite_size -= 1
        # parameters to be adapted in that order:
        # n individuos elite - done
        # alteração do tamanho da população?
        # parâmetros dos métodos de seleção/crossover/mutação
        # alteração dos próprios métodos para métodos mais aleatórios/diversificadores
        # em ultimo caso métodos de reinicio de população

    def reset_parameters(self,):
        self.elite_size = math.ceil(self.population_size * self.experiment["elite_size"])
        self.offspring_size = math.floor(self.population_size * self.experiment["offspring_size"])
        self.population_size = self.experiment["population_size"]
        self.commoners_size = self.population_size - self.elite_size
        self.selection_method = self.experiment["selection_method"]
        self.mutation_method = self.experiment["mutation_method"]
        self.self_mutation = self.experiment["self_mutation"]
        self.crossover_method = self.experiment["crossover_method"]
        self.parent_selection_method = self.experiment["parent_selection_method"]
        self.mutation_probability = self.experiment["mutation_probability"]
        self.crossover_probability = self.experiment["crossover_probability"]


    def draw_plots(self):
        if not self.generate_plots:
            return

        textstr = '\n'.join((
            r'Encoding method= %s' %(self.encoding,),
            r'Selection method= %s' %(self.selection_method,),
            r'Mutation method= %s' %(self.mutation_method,),
            r'Crossover method= %s' %(self.crossover_method,)))

        self.plot_results(textstr)
        self.plot_time_to_best_sol(textstr)
        self.plot_best_sol_tracking(textstr)

    def plot_results(self, textstr):
        return
    
    def plot_time_to_best_sol(self, textstr):
        times= np.array(self.best_sol_change_times) - self.start_time
        sols = np.unique(self.best_sol_tracking)
        plt.plot(times, sols, color='tab:blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Generation")
        plt.title("Exp. {} - Time to best solution found".format(str(self.experiment_id)))  
        plt.axhline(y=self.best_known_result, color='tab:red', linestyle='-')
        
        # place a text box in bottom right in axes coords
        ax = plt.gca()
        ax.text(0.4, 0.05, textstr, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', alpha=0.4))

        file_name = "{}_time_to_best_sol_.png".format(str(self.experiment_id))
        plots_file = os.path.join(self.plots_dir, file_name)
        plt.savefig(plots_file)
        plt.close()

    def plot_best_sol_tracking(self, textstr):
        plt.plot([i for i in range(len(self.best_sol_tracking))], self.best_sol_tracking, color='tab:blue')
        plt.xlabel("Generation")
        plt.ylabel("Best solution")
        plt.title("Exp. {} - Evolution of Best solution found so far".format(str(self.experiment_id)))  
        plt.axhline(y=self.best_known_result, color='tab:red', linestyle='-')
        
        # place a text box in bottom right in axes coords
        ax = plt.gca()
        ax.text(0.4, 0.05, textstr, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', alpha=0.4))


        file_name = "{}_best_sol_tracking_.png".format(str(self.experiment_id))
        plots_file = os.path.join(self.plots_dir, file_name)
        plt.savefig(plots_file)
        plt.close()

    def path_relinking(self, population):
        if len(population) <= 1:
            return []

        intersting_sol_found = []
        uniques , indices = np.unique([individual.binary_chromosome for individual in population], return_index=True)
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
                        if new_individual.fitness >= self.best_sol:
                            intersting_sol_found.append(new_individual)
        return intersting_sol_found
    
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