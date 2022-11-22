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
                                for mutation_probability in experiment_setup["mutation_probability"]:
                                    for crossover_probability in experiment_setup["crossover_probability"]:
                                        for assexual_crossover in experiment_setup["perform_assexual_crossover"]:
                                            for elite_size in experiment_setup["elite_size"]:
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
                                                    "mutation_probability": float(mutation_probability),
                                                    "crossover_probability": float(crossover_probability),
                                                    "perform_assexual_crossover": True if assexual_crossover == "true" else False,
                                                    "elite_size": float(elite_size)
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

experiment_setup_file = sys.argv[1]

main(experiment_setup_file)