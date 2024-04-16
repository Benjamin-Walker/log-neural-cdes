import os

import numpy as np
from scipy.io.wavfile import read


data = np.zeros((2452, 305908, 2))
labels = np.zeros((2452, 8))
i = 0
for type in zip(["speech", "song"], ["01", "02"]):
    for actor in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    ]:
        for emotion in zip(
            ["01", "02", "03", "04", "05", "06", "07", "08"], [1, 2, 3, 4, 5, 6, 7, 8]
        ):
            for intensity in ["01", "02"]:
                for statement in ["01", "02"]:
                    for repetition in ["01", "02"]:
                        file = (
                            f"data/raw/speech/{type[0]}/Actor_{actor}/"
                            f"03-{type[1]}-{emotion[0]}-{intensity}-{statement}-{repetition}-{actor}.wav"
                        )
                        if os.path.isfile(file):
                            a = read(file)
                            new_data = np.array(a[1], dtype=float)
                            if len(new_data.shape) > 1:
                                new_data = new_data[:, 0]
                            data[i, 1 : len(new_data) + 1, 0] = new_data
                            data[i, 2 : len(new_data) + 2, 1] = new_data
                            labels[i, emotion[1] - 1] = 1
                            i += 1

data = data / 32768
for i in range(10):
    np.save(f"data/processed/speech/data_{i}.npy", data[i * 246 : (i + 1) * 246])
    np.save(f"data/processed/speech/labels_{i}.npy", labels[i * 246 : (i + 1) * 246])
# save_pickle(data, "data/processed/speech/data.pkl")
# save_pickle(labels, "data/processed/speech/labels.pkl")
