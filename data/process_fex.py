import numpy as np
import pandas as pd
from process_uea import save_pickle


years = [
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
]

leap_years = ["2008", "2012", "2016", "2020"]

all_4_data = []

for fex in ["AUDJPY", "EURCHF", "GBPCAD", "NZDUSD"]:
    print(fex)
    all_data = []
    for year in years:
        filename = f"data/raw/FEX/{fex.lower()}/DAT_ASCII_{fex}_M1_{year}/DAT_ASCII_{fex}_M1_{year}.csv"
        df = pd.read_csv(filename, delimiter=";", header=None)
        df.drop(columns=[2, 3, 4, 5], inplace=True)

        df[0] = pd.to_datetime(df[0], format="%Y%m%d %H%M%S%f")
        base_time = pd.to_datetime(f"{year}0101 170000", format="%Y%m%d %H%M%S%f")

        df[0] = (df[0] - base_time).dt.total_seconds().div(60)

        idxs = np.array(df[0], dtype=int)
        values = np.array(df[1], dtype=float)

        if year in leap_years:
            data = np.empty((527040 - 1440,))
        else:
            data = np.empty((525600 - 1440,))
        data[:] = np.nan

        data[idxs] = values

        if np.isnan(data[0]):
            first_idx = np.where(~np.isnan(data))[0][0]
            data[0] = data[first_idx]

        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        data[mask] = data[idx[mask]]
        all_data.append(data)

    all_data = np.hstack(all_data)
    all_4_data.append(all_data)

all_4_data = np.stack([x[:, None] for x in all_4_data], axis=1).squeeze()
np.save("data/processed/FEX/all_data.npy", all_4_data)

data = np.load("data/processed/FEX/all_data.npy")

for i in range(4):
    data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

day = 60 * 24

idxs = [0, 1, 2, 5, 6]
idxs_copy = idxs.copy()
for i in range(1, 52):
    idxs = idxs + [x + 7 * i for x in idxs_copy]
idxs_copy = idxs.copy()
for i in range(1, 14):
    idxs = idxs + [x + 363 * i for x in idxs_copy]

choices = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

labels = []

length = 1

save_data = np.zeros((len(idxs[: -2 * length]) * len(choices), length * day, 4))

i = 0
for choice in choices:
    for idx in idxs[: -2 * length]:
        if np.random.rand() < 0.5:
            data1 = data[idx * day : (idx + length) * day, choice[0]]
            data2 = data[idx * day : (idx + length) * day, choice[1]]
            save_data[i, :, choice[0]] = data1
            save_data[i, :, choice[1]] = data2
            labels.append(0)
        else:
            if np.random.rand() < 0.5:
                data1 = data[idx * day : (idx + length) * day, choice[0]]
                data2 = data[(idx + length) * day : (idx + 2 * length) * day, choice[1]]
                save_data[i, :, choice[0]] = data1
                save_data[i, :, choice[1]] = data2
            else:
                data1 = data[(idx + length) * day : (idx + 2 * length) * day, choice[0]]
                data2 = data[idx * day : (idx + length) * day, choice[1]]
                save_data[i, :, choice[0]] = data1
                save_data[i, :, choice[1]] = data2
            labels.append(1)
        i += 1

save_labels = np.array(labels)
rng_state = np.random.get_state()
np.random.shuffle(save_labels)
np.random.set_state(rng_state)
np.random.shuffle(save_data)

save_pickle(save_data, "data/processed/FEX/daily/data.pkl")
save_pickle(save_labels, "data/processed/FEX/daily/labels.pkl")
