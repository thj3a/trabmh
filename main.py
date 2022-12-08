import mat73
import numpy as np
import pandas as pd
import sys
import os
from scipy.io import loadmat
import json
import time
from datetime import datetime
from genetic_algorithm import GeneticAlgoritm
import pdb
from multiprocessing import Process, Pool, freeze_support, Lock, cpu_count
from itertools import repeat
from copy import deepcopy

mutex = Lock()

# Checks if the param combination is valid.
# It focuses only on params that can be incompatible 
# between themselves.
def validade_experiment_params(params):
    # TODO implement param validation

    selection_methods = ["roulette", "ranking", "byclass", "fullyrandom"]
    binary_crossover_methods = ["misc", "singlepoint"]
    binary_mutation_methods = ["singlepoint", "singlepointinterchange"]
    permutation_crossover_methods = ["opx"]
    permutation_mutation_methods = ["singleexchange"]

    encoding_method = params["encoding_method"]
    crossover_methods = None
    mutation_methods = None
    
    if encoding_method == "binary": 
        crossover_methods = binary_crossover_methods
        mutation_methods = binary_mutation_methods
    elif encoding_method == "permutation": 
        crossover_methods = permutation_crossover_methods
        mutation_methods = permutation_mutation_methods
    else:
        return False, "Unknown encoding method {}.".format(encoding_method)

    crossover_method = params["crossover_method"].split("_")[0]
    if crossover_method not in crossover_methods:
        return False, "Unknown crossover method {} or it is incompatible with the encoding method {}.".format(crossover_method, encoding_method)

    mutation_method = params["mutation_method"].split("_")[0]
    if mutation_method not in mutation_methods:
        return False, "Unknown mutation method {} or it is incompatible with the encoding method {}.".format(mutation_method, encoding_method)
    
    # checks if the genetic operators exist
    selection_method = params["selection_method"].split("_")[0]
    if selection_method not in selection_methods:
        return False, "Unknown selection method {}.".format(selection_method)

    return True, ""

def build_experiments(experiment_setup):
    experiments = []
    experiment_count = 0

    instance_paths = os.listdir(experiment_setup["instance_dir"])

    for instance_path in instance_paths:
        instance_name = os.path.basename(instance_path)
        execution_id = str(int(time.time()))

        # If len(experiment_setup["instances"]) == 0, all instances are considered.
        # If different, only those in experiment_setup["instances"] will be considered.
        # Obs.: instance names are the full file name, including the extension. In other
        # words, "Instance_40_1.mat" is correct, while "Instance_40_1" is wrong.
        if len(experiment_setup["instances"]) > 0 and instance_name not in experiment_setup["instances"]:
            # TODO insert a log call here.
            continue

        print(instance_name)
        instance = loadmat(os.path.join(experiment_setup["instance_dir"], instance_path))
        
        A = instance["A"]
        n = A.shape[0]
        m = A.shape[1]
        s = int(n/2)

        best_known_result = - np.inf

        try:
            tmp = instance_name.replace("Instance_", "")
            known_results_file = os.path.join(experiment_setup["known_results_dir"], "x_ls_" + tmp)
            known_result = mat73.loadmat(known_results_file)
            best_known_result = np.linalg.slogdet(np.matmul(np.matmul(A.T, np.diagflat(known_result['x_ls'])), A))[1]

            # creates folder to store plots.
            plots_dir = os.path.join(experiment_setup["plots_dir"], str(execution_id))
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)

        except Exception as e:
            print(e)

        print(best_known_result)

        for initialization in experiment_setup["initialization_method"]:
            for encoding in experiment_setup["encoding_method"]:
                for max_generations in experiment_setup["max_generations"]:
                    for population_size in experiment_setup["population_size"]:
                        for selection in experiment_setup["selection_method"]:
                            for mutation in experiment_setup["mutation_method"]:
                                for self_mutation in experiment_setup["self_mutation"]:
                                    for crossover in experiment_setup["crossover_method"]:
                                        for parent_selection in experiment_setup["parent_selection_method"]:
                                            for mutation_probability in experiment_setup["mutation_probability"]:
                                                for crossover_probability in experiment_setup["crossover_probability"]:
                                                    for assexual_crossover in experiment_setup["perform_assexual_crossover"]:
                                                        for elite_size in experiment_setup["elite_size"]:
                                                            for offspring_size in experiment_setup["offspring_size"]:
                                                                for path_relinking in experiment_setup["perform_path_relinking"]:
                                                                    experiment = {
                                                                        "experiment_id": str(experiment_count),
                                                                        "execution_id": execution_id,
                                                                        "seed": int(experiment_setup["seed"]),
                                                                        "silent": bool(experiment_setup["silent"]),
                                                                        "instance": instance_name,
                                                                        "instance_path": instance_path,
                                                                        "plots_dir": plots_dir,
                                                                        "A": instance["A"],
                                                                        "n": n,
                                                                        "m": m,
                                                                        "s": s,
                                                                        "best_known_result": float(best_known_result),
                                                                        "generate_plots": bool(experiment_setup["generate_plots"]),
                                                                        "max_generations": int(max_generations),
                                                                        "max_generations_without_change": int(experiment_setup["max_generations_without_change"]),
                                                                        "max_time": int(experiment_setup["max_time"]),
                                                                        "max_time_without_change": int(experiment_setup["max_time_without_change"]),
                                                                        "population_size": int(population_size),
                                                                        "encoding_method": encoding,
                                                                        "initialization_method": initialization,
                                                                        "selection_method": selection,
                                                                        "mutation_method": mutation,
                                                                        "self_mutation": bool(self_mutation),
                                                                        "crossover_method": crossover,
                                                                        "parent_selection_method": parent_selection,
                                                                        "mutation_probability": float(mutation_probability),
                                                                        "crossover_probability": float(crossover_probability),
                                                                        "perform_assexual_crossover": True if assexual_crossover == "true" else False,
                                                                        "elite_size": float(elite_size),
                                                                        "offspring_size": float(offspring_size),
                                                                        "perform_path_relinking": bool(path_relinking),
                                                                    }

                                                                    experiments.append(experiment)

                                                                    experiment_count += 1

    return experiments

def run_experiment(experiment_setup, experiment):
    results = []
    validated, message = validade_experiment_params(experiment)

    start_time = time.time()
    
    if validated:
        #print(experiment)
        d_opt = GeneticAlgoritm(experiment)
        results, num_generations, message = d_opt.loop()

    finish_time = time.time()
    
    with mutex:
        # Saves results to disk.
        save_results(experiment_setup, experiment, results, start_time, finish_time, validated, num_generations, message)

def main(experiment_setup_file):
    experiment_setup = json.load(open(experiment_setup_file))
    experiments = build_experiments(experiment_setup)
    print(f"Number of experiments: {len(experiments)}")
    pool = Pool(experiment_setup["num_processes"])
    # startmap is synchronous and, as such, will hold the execution of this called thread untill all maped jobs are executed.
    pool.starmap(run_experiment, zip(repeat(experiment_setup), experiments), chunksize=1) 
    pool.close()

    print("Finished.")

def save_results(experiment_setup, experiment, results, start_time, finish_time, validated, num_generations, message):
    fields = ["experiment_id",
        "execution_id",
        "seed",
        "instance",
        "n",
        "m",
        "s",
        "best_known_result",
        "max_generations",
        "max_generations_without_change",
        "max_time",
        "max_time_without_change",
        "population_size",
        "encoding_method",
        "initialization_method",
        "selection_method",
        "mutation_method",
        "self_mutation",
        "crossover_method",
        "parent_selection_method",
        "mutation_probability",
        "crossover_probability",
        "elite_size",
        "offspring_size",
        "perform_path_relinking",
        "generate_plots", # important to include because it affects the total time of an experiment.
    ]

    results_fields = [
        "start_time",
        "finish_time",
        "total_time_ms",
        "best_solution_found",
        "best_solution_hash",
        "gap",
        "elite_fitness",
        "elite_hash",
        "num_generations",
        "message"
    ]
    
    results_file = os.path.join(experiment_setup["results_dir"], "results.csv")

    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            header = ";".join(fields + results_fields)
            file.write(header + "\n")

    with open(results_file, "a") as file:
            result_line = [experiment[field] for field in fields]
            result_line = ";".join([str(value) for value in result_line])

            elite_fitness = ",".join([str(individual.fitness) for individual in results])
            elite_hash = ",".join([str(individual.individual_hash) for individual in results])

            best_fitness = results[0].fitness if validated else ""
            best_hash = results[0].individual_hash if validated else ""
            gap = ""
            #print(best_fitness, type(best_fitness))
            if type(best_fitness) != str:
                gap = (best_fitness - experiment["best_known_result"]) / experiment["best_known_result"]


            file.write(
                result_line + ";" + 
                str(datetime.fromtimestamp(start_time)) + ";" + 
                str(datetime.fromtimestamp(finish_time)) + ";" + 
                str(finish_time - start_time) + ";" + 
                str(best_fitness) + ";" + 
                str(best_hash) + ";" + 
                str(gap) + ";" + 
                elite_fitness + ";" +
                #str(",".join([str(individual.chromosome.T) for individual in results])) + ";" + 
                elite_hash + ";" +
                str(num_generations) + ";" +
                message + "\n"
            )
    
if __name__ == "__main__":
    experiment_setup_file = sys.argv[1]
    main(experiment_setup_file)
