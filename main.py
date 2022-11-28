import mat73
import numpy as np
import pandas as pd
import sys
import os
from scipy.io import loadmat
import json
from genetic_algorithm import GeneticAlgoritm

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

                                                        experiments.append(experiment)

                                                        experiment_count += 1

    return experiments

def main(experiment_setup_file):
    experiment_setup = json.load(open(experiment_setup_file))
    experiments = build_experiments(experiment_setup)

    for experiment in experiments:
        #print(experiment["experiment_id"])
        d_opt = GeneticAlgoritm(experiment)
        results = d_opt.loop()

        # Saves results to disk.
        save_results(experiment_setup, experiment, results)

def save_results(experiment_setup, experiment, results):
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
        "elite_fitness"
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

            file.write(result_line + ";" + elite_fitness + "\n")
    


    


experiment_setup_file = sys.argv[1]

main(experiment_setup_file)