"""Plot the losses of the DRE during the training."""

import matplotlib.pyplot as plt
import numpy as np

history = np.loadtxt("results/dre_losses.txt")
epochs = history[:, 0]

plt.figure()
plt.plot(epochs, history[:, 1], "--", color="tab:blue", label="training - reconstruction")
plt.plot(epochs, history[:, 3], "-", color="tab:blue", label="validation - reconstruction")
plt.plot(epochs, history[:, 2], "--", color="tab:orange", label="training - dynamics")
plt.plot(epochs, history[:, 4], "-", color="tab:orange", label="validation - dynamics")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("results/dre_losses.pdf", bbox_inches="tight")
