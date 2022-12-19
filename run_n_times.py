import os
import sys
import json
import copy
import pandas as pd


experiment_setup_file = sys.argv[1]
number_of_runs = int(sys.argv[2])
desired_gap = float(sys.argv[3])
output_dir = "ttts"

try:
    output_dir = sys.argv[4]
except:
    print("No outputdir provided. Using '{}'.".format(output_dir))

experiment_setup = json.load(open(experiment_setup_file))

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for instance in experiment_setup["instances"]:
    instance_tmp_setup_file = instance + "_" + experiment_setup_file
    instance_tmp_setup_file = os.path.join(output_dir, instance_tmp_setup_file)
    tmp_setup_json = copy.deepcopy(experiment_setup)
    tmp_setup_json["instances"] = [instance]

    with open(instance_tmp_setup_file, "w") as file:
        json.dump(tmp_setup_json, file)

    #print(instance_tmp_setup_file, tmp_setup_json)

    for i in range(number_of_runs):
        print(instance, "Running", i+1)
        os.system("python main.py {}".format(instance_tmp_setup_file))


    results = pd.read_csv(os.path.join(experiment_setup["results_dir"], "results.csv"), sep=";")
    results = results[results["instance"] == instance]

    ttts = []

    for execution_id in results["execution_id"]:
        sols = pd.read_csv(os.path.join(experiment_setup["plots_dir"], os.path.join(str(execution_id), "solution_times.csv")), sep=";")
        sols = sols[sols["gaps"] <= desired_gap]
        ttts.append(sorted(sols["best_sol_change_times"])[0])

    print(ttts)
    
    ttts_file = os.path.join(output_dir, "{}_gap_{}_ttts.txt".format(instance, str(desired_gap)))
    ttts_dat_file = ttts_file.replace(".txt", ".dat")
    
    with open(ttts_file, "w") as file:
        file.write("\n".join(str(ttt) for ttt in ttts))

    with open(ttts_dat_file, "w") as file:
        file.write("\n".join(str(ttt) for ttt in sorted(ttts)))

    