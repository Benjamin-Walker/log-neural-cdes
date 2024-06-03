import os

import numpy as np


results_dirs = [
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_final_repeats_not_presplit/'
    # 'outputs_ppg_random_repeats/'
    # 'outputs_ppg_repeats/'
    "outputs_EW_repeats/"
]

for results_dir in results_dirs:
    for model in os.listdir(results_dir):
        if not model.startswith("."):
            model += "/"
            for dataset in sorted(os.listdir(results_dir + model)):
                # if model != "ssm/" or dataset != "EigenWorms":
                # if model != "nrde/" or dataset != "MotorImagery":
                dataset += "/"
                test_acc = []
                times = []
                for exp in os.listdir(results_dir + model + dataset):
                    test_acc.append(
                        np.load(results_dir + model + dataset + exp + "/test_acc.npy")
                    )
                    times.append(
                        np.load(results_dir + model + dataset + exp + "/all_time.npy")
                    )

                # print(np.mean([len(x) for x in times]))
                test_acc = np.array(test_acc)
                # print(100*test_acc)
                print(
                    f"{model[:-1]} {dataset[:-1]} {100*np.mean(test_acc)} {100*np.std(test_acc)}"
                )
