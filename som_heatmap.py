import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as mpatches

data = np.load("reliability_vectors.npy")
som = pickle.load(open("models/som_model.pkl", "rb"))

grid = np.zeros((10, 10))
counts = np.zeros((10, 10))

for x in data:
    node = som.winner(x)
    grid[node] += x[-1] 
    counts[node] += 1

for i in range(10):
    for j in range(10):
        if counts[i, j] > 0:
            grid[i, j] /= counts[i, j]


plt.figure(figsize=(7, 6))

heatmap = plt.imshow(
    grid,
    cmap="RdYlGn",
    origin="lower",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

plt.xticks([])
plt.yticks([])

cbar = plt.colorbar(heatmap)
cbar.set_label("Average Reliability Score", fontsize=11)

plt.title("Global Reliability Landscape via Self-Organizing Map", fontsize=12)

safe_patch = mpatches.Patch(color="#1a9641", label="Safe (≥ 0.75)")
moderate_patch = mpatches.Patch(color="#fdae61", label="Moderate (0.50–0.75)")
high_patch = mpatches.Patch(color="#d7191c", label="High Risk (< 0.50)")

plt.legend(
    handles=[safe_patch, moderate_patch, high_patch],
    loc="upper right",
    fontsize=9,
    frameon=False
)

plt.tight_layout()

plt.savefig("som_reliability_heatmap.png", dpi=600)
plt.savefig("som_reliability_heatmap.pdf")

plt.show()