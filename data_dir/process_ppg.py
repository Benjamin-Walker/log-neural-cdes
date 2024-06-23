"""
This script processes the PPG_FieldStudy dataset and saves the processed data in the data_dir/processed directory.
"""

import os
import pickle
import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv


all_train_input = []
all_val_input = []
all_test_input = []
all_train_output = []
all_val_output = []
all_test_output = []

for i in range(1, 16):

    print(i)

    with open(f"data_dir/raw/PPG_FieldStudy/S{i}/S{i}.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ACC = np.repeat(data["signal"]["wrist"]["ACC"], 2, axis=0)
    BVP = data["signal"]["wrist"]["BVP"]
    EDA = np.repeat(data["signal"]["wrist"]["EDA"], 16, axis=0)
    TEMP = np.repeat(data["signal"]["wrist"]["TEMP"], 16, axis=0)

    ACC = 2 * (ACC - np.min(ACC)) / (np.max(ACC) - np.min(ACC)) - 1
    BVP = 2 * (BVP - np.min(BVP)) / (np.max(BVP) - np.min(BVP)) - 1
    EDA = 2 * (EDA - np.min(EDA)) / (np.max(EDA) - np.min(EDA)) - 1
    TEMP = 2 * (TEMP - np.min(TEMP)) / (np.max(TEMP) - np.min(TEMP)) - 1

    input = np.concatenate([ACC, BVP, EDA, TEMP], axis=1)
    output = data["label"]
    output = np.concatenate([[output[0]], [output[0]], [output[0]], output], axis=0)
    # output = np.concatenate([output, [output[-1]]], axis=0)

    output = 2 * (output - np.min(output)) / (np.max(output) - np.min(output)) - 1

    variant = random.randint(0, 5)
    if variant == 0:
        train_input = input[: int(0.7 * len(input))]
        train_output = output[: int(0.7 * len(output))]
        val_input = input[int(0.7 * len(input)) : int(0.85 * len(input))]
        val_output = output[int(0.7 * len(output)) : int(0.85 * len(output))]
        test_input = input[int(0.85 * len(input)) :]
        test_output = output[int(0.85 * len(output)) :]
    elif variant == 1:
        train_input = input[: int(0.7 * len(input))]
        train_output = output[: int(0.7 * len(output))]
        val_input = input[int(0.85 * len(input)) :]
        val_output = output[int(0.85 * len(output)) :]
        test_input = input[int(0.7 * len(input)) : int(0.85 * len(input))]
        test_output = output[int(0.7 * len(output)) : int(0.85 * len(output))]
    elif variant == 2:
        train_input = input[int(0.15 * len(input)) : int(0.85 * len(input))]
        train_output = output[int(0.15 * len(output)) : int(0.85 * len(output))]
        val_input = input[: int(0.15 * len(input))]
        val_output = output[: int(0.15 * len(output))]
        test_input = input[int(0.85 * len(input)) :]
        test_output = output[int(0.85 * len(output)) :]
    elif variant == 3:
        train_input = input[int(0.15 * len(input)) : int(0.85 * len(input))]
        train_output = output[int(0.15 * len(output)) : int(0.85 * len(output))]
        val_input = input[int(0.85 * len(input)) :]
        val_output = output[int(0.85 * len(output)) :]
        test_input = input[: int(0.15 * len(input))]
        test_output = output[: int(0.15 * len(output))]
    elif variant == 4:
        train_input = input[int(0.30 * len(input)) :]
        train_output = output[int(0.30 * len(output)) :]
        val_input = input[: int(0.15 * len(input))]
        val_output = output[: int(0.15 * len(output))]
        test_input = input[int(0.15 * len(input)) : int(0.30 * len(input))]
        test_output = output[int(0.15 * len(output)) : int(0.30 * len(output))]
    elif variant == 5:
        train_input = input[int(0.30 * len(input)) :]
        train_output = output[int(0.30 * len(output)) :]
        val_input = input[int(0.15 * len(input)) : int(0.30 * len(input))]
        val_output = output[int(0.15 * len(output)) : int(0.30 * len(output))]
        test_input = input[: int(0.15 * len(input))]
        test_output = output[: int(0.15 * len(output))]

    train_input = np.swapaxes(swv(train_input, 49920, 0)[::4992], 1, 2)
    val_input = np.swapaxes(swv(val_input, 49920, 0)[::4992], 1, 2)
    test_input = np.swapaxes(swv(test_input, 49920, 0)[::4992], 1, 2)

    train_output = swv(train_output, 390, 0)[::39]
    val_output = swv(val_output, 390, 0)[::39]
    test_output = swv(test_output, 390, 0)[::39]

    all_train_input.append(train_input)
    all_val_input.append(val_input)
    all_test_input.append(test_input)

    all_train_output.append(train_output)
    all_val_output.append(val_output)
    all_test_output.append(test_output)

train_input = np.concatenate(all_train_input, axis=0)
val_input = np.concatenate(all_val_input, axis=0)
test_input = np.concatenate(all_test_input, axis=0)

train_output = np.concatenate(all_train_output, axis=0)
val_output = np.concatenate(all_val_output, axis=0)
test_output = np.concatenate(all_test_output, axis=0)

os.makedirs("data_dir/processed/PPG/ppg", exist_ok=True)

with open("data_dir/processed/PPG/ppg/X_train.pkl", "wb") as f:
    pickle.dump(train_input, f)
with open("data_dir/processed/PPG/ppg/y_train.pkl", "wb") as f:
    pickle.dump(train_output, f)
with open("data_dir/processed/PPG/ppg/X_val.pkl", "wb") as f:
    pickle.dump(val_input, f)
with open("data_dir/processed/PPG/ppg/y_val.pkl", "wb") as f:
    pickle.dump(val_output, f)
with open("data_dir/processed/PPG/ppg/X_test.pkl", "wb") as f:
    pickle.dump(test_input, f)
with open("data_dir/processed/PPG/ppg/y_test.pkl", "wb") as f:
    pickle.dump(test_output, f)
