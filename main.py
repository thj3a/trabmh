import mat73
import numpy as np
import pandas as pd
import sys
import os
from scipy.io import loadmat
import json
import time
from genetic_algorithm import GeneticAlgoritm
import pdb
from multiprocessing import Process, Pool, freeze_support, Lock, cpu_count
from itertools import repeat
from copy import deepcopy

mutex = Lock()

# Checks if the param combination is valid
def validade_experiment_params(params):
    # TODO implement param validation
    return True

def build_experiments(experiment_setup):
    experiments = []
    experiment_count = 0

    instance_paths = os.listdir(experiment_setup["instance_dir"])

    for instance_path in instance_paths:
        instance_name = os.path.basename(instance_path)

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
        except Exception as e:
            print(e)

        print(best_known_result)

        for encoding in experiment_setup["encoding_method"]:
            for max_generations in experiment_setup["max_generations"]:
                for population_size in experiment_setup["population_size"]:
                    for selection in experiment_setup["selection_method"]:
                        for mutation in experiment_setup["mutation_method"]:
                            for crossover in experiment_setup["crossover_method"]:
                                for parent_selection in experiment_setup["parent_selection_method"]:
                                    for mutation_probability in experiment_setup["mutation_probability"]:
                                        for crossover_probability in experiment_setup["crossover_probability"]:
                                            for assexual_crossover in experiment_setup["perform_assexual_crossover"]:
                                                for elite_size in experiment_setup["elite_size"]:
                                                    for offspring_size in experiment_setup["offspring_size"]:
                                                        experiment = {
                                                            "experiment_id": str(experiment_count),
                                                            "seed": int(experiment_setup["seed"]),
                                                            "silent": bool(experiment_setup["silent"]),
                                                            "instance": instance_name,
                                                            "instance_path": instance_path,
                                                            "A": instance["A"],
                                                            "n": n,
                                                            "m": m,
                                                            "s": s,
                                                            "best_known_result": float(best_known_result),
                                                            "max_generations": int(max_generations),
                                                            "population_size": int(population_size),
                                                            "encoding_method": encoding,
                                                            "selection_method": selection,
                                                            "mutation_method": mutation,
                                                            "crossover_method": crossover,
                                                            "parent_selection_method": parent_selection,
                                                            "mutation_probability": float(mutation_probability),
                                                            "crossover_probability": float(crossover_probability),
                                                            "perform_assexual_crossover": True if assexual_crossover == "true" else False,
                                                            "elite_size": float(elite_size),
                                                            "offspring_size": float(offspring_size)
                                                        }

                                                        if not validade_experiment_params(experiment):
                                                            # TODO add log
                                                            continue

                                                        experiments.append(experiment)

                                                        experiment_count += 1

    return experiments

def run_experiment(d_opt: GeneticAlgoritm, experiment_setup, experiment):
    start_time = time.time()
    results = d_opt.loop()
    total_time = time.time() - start_time
    with mutex:
        save_results(experiment_setup, experiment, results, total_time)

def main(experiment_setup_file):
    experiment_setup = json.load(open(experiment_setup_file))
    experiments = build_experiments(experiment_setup)
    GA = [GeneticAlgoritm(experiment) for experiment in experiments]
    

    if experiment_setup["parallel"][0]:
        pool = Pool(cpu_count()-1)
        pool.starmap(run_experiment, zip(GA, repeat(experiment_setup), experiments))
        pool.close()
    else:
        for i, d_opt in enumerate(GA):
            if experiment_setup["parallel"][0]:
                pool = Pool(cpu_count()-1)
            else:
                run_experiment(d_opt, experiment_setup, experiments[i])


def save_results(experiment_setup, experiment, results, total_time):
    fields = ["experiment_id",
        "seed",
        "instance",
        "n",
        "m",
        "s",
        "best_known_result",
        "max_generations",
        "population_size",
        "encoding_method",
        "selection_method",
        "mutation_method",
        "crossover_method",
        "parent_selection_method",
        "mutation_probability",
        "crossover_probability",
        "elite_size",
        "offspring_size"
    ]
    results_fields =[
        "total_time",
        "best_solution_found",
        "best_solution_hash",
        "elite_fitness",
        "elite_hash"
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

            file.write(
                result_line + ";" + 
                str(total_time) + ";" + 
                str(results[0].fitness) + ";" + 
                str(results[0].individual_hash) + ";" + 
                elite_fitness + ";" +
                #str(",".join([str(individual.chromosome.T) for individual in results])) + ";" + 
                elite_hash + "\n"
            )
    







if __name__ == "__main__":
    experiment_setup_file = sys.argv[1]
    main(experiment_setup_file)