import os

import iisignature as iisig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font = {"size": 20}
matplotlib.rc("font", **font)
plt.rcParams["text.usetex"] = True

vs = np.arange(1, 16)
length1 = [iisig.logsiglength(v, 1) for v in vs]
length2 = [iisig.logsiglength(v, 2) for v in vs]
plt.figure(figsize=(8, 4))
plt.plot(vs, vs, label="$v$", color="#377eb8", zorder=0, linewidth=3)
plt.scatter(vs, length1, label="$N=1$", color="#ff7f00", s=100, zorder=1)
plt.scatter(vs, length2, label="$N=2$", color="#4daf4a", s=100, zorder=2)
plt.xlabel("Time Series Dimension, $v$")
plt.xticks(range(1, 16))
plt.ylabel("Log-Signature Dimension, $\\beta(v,N)$")
plt.title("Log-Signature Dimension vs Time Series Dimension")
plt.legend()
os.makedirs("results/images", exist_ok=True)
plt.savefig("results/images/logsig_length.png", dpi=300, bbox_inches="tight")
plt.show()
