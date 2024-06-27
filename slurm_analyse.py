import os

import numpy as np


results_dirs = [
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_Oct10/',
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_Oct19/',
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_hypopt/',
    "/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_ssm_lru/",
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_EW/',
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_nrde/',
    # '/data/math-datasig/shug6778/Log-Neural-CDEs/outputs_ppg/',
    # "outputs_ppg/",
]

results = {}

for results_dir in results_dirs:
    for model in ["lru"]:
        if not model.startswith("."):
            model += "/"
            for dataset in sorted(os.listdir(results_dir + model)):
                if dataset in ["SelfRegulationSCP2"]:
                    dataset += "/"
                    all_all_train_acc = []
                    train_acc = []
                    val_acc = []
                    val_index = []
                    all_times = []
                    test_acc = []
                    exps = []
                    accepted_exps = []
                    rejected_exps = []
                    steps = []
                    print(len(os.listdir(results_dir + model + dataset)))
                    for exp in os.listdir(results_dir + model + dataset):
                        steps = np.load(
                            results_dir + model + dataset + exp + "/steps.npy"
                        )
                        all_train_acc = np.load(
                            results_dir + model + dataset + exp + "/all_train_acc.npy"
                        )
                        all_val_acc = np.load(
                            results_dir + model + dataset + exp + "/all_val_acc.npy"
                        )
                        idx = np.argmax(all_val_acc)
                        test_acc_ = np.load(
                            results_dir + model + dataset + exp + "/test_acc.npy"
                        )
                        all_time = np.load(
                            results_dir + model + dataset + exp + "/all_time.npy"
                        )
                        if all_train_acc[np.argmax(all_val_acc)] >= max(all_val_acc):
                            accepted_exps.append(
                                [exp.split("_")[13], exp.split("_")[22]]
                            )
                            exps.append(exp)
                            all_all_train_acc.append(all_train_acc)
                            train_acc.append(all_train_acc[np.argmax(all_val_acc)])
                            val_acc.append(np.max(all_val_acc))
                            val_index.append(np.argmax(all_val_acc))
                            test_acc.append(
                                np.load(
                                    results_dir
                                    + model
                                    + dataset
                                    + exp
                                    + "/test_acc.npy"
                                )
                            )
                            all_times.append(all_time)
                        # except:
                        #     rejected_exps.append([exp.split("_")[13], exp.split("_")[22]])
                        #     pass
                        #     # print(exp.split("_")[13], exp.split("_")[22])
                        #     # import sys
                        #     # sys.exit()
                        #     # pass

                    # print(len(rejected_exps))
                    # list_ = [el for el in rejected_exps if el in accepted_exps]
                    # new_list = []

                    # for item in list_:
                    #     if item not in new_list:
                    #         new_list.append(item)
                    # print(new_list)
                    val_acc = np.array(val_acc)
                    print(len(val_acc))
                    test_acc = np.array(test_acc)
                    if len(val_acc) > 0:
                        results[
                            model[:-1] + "_" + dataset[:-1] + "_" + results_dir[-2:]
                        ] = [max(val_acc), test_acc[np.argmax(val_acc)].item()]
                        all_idxs = np.argwhere(val_acc == np.max(val_acc))
                        train_accs = []
                        for idx in all_idxs:
                            idx = idx[0]
                            train_accs.append(train_acc[idx])
                        all_train_idxs = np.argwhere(train_accs == np.max(train_accs))
                        for idx in all_train_idxs:
                            idx = all_idxs[idx[0]][0]
                            print(
                                f"{model[:-1]} {dataset[:-1]} "
                                + " ".join(exps[idx].split("_"))
                                + f" {train_acc[idx]}, {val_acc[idx]}, {test_acc[idx]},"
                            )
