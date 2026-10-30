import numpy as np
import matplotlib.pyplot as plt

data = np.load("result/threshold/simulation_threshold.npz", allow_pickle=True)

plt.figure(figsize=(8, 6))
markers = {'ER': 'o-', 'BA': 's-', 'WS': '^-'}
colors = {'exp': 'r', 'logn': 'g', 'pl': 'b'}
labels = {'exp': 'Exponential', 'logn': 'Lognormal', 'pl': 'Power-law'}

for net_type in data.files:
    for dist_key in data[net_type].item():
        k_vals = data[net_type].item()[dist_key]['k']
        tau_vals = data[net_type].item()[dist_key]['tau']
        plt.plot(k_vals, tau_vals, markers[net_type] + colors[dist_key], label=f"{net_type}-{labels[dist_key]}")

plt.plot(k_vals, 1/np.array(k_vals))
plt.xlabel("Average Degree <k>")
plt.ylabel("Threshold τ")
plt.legend()
plt.title("SIS Epidemic Threshold in Different Networks")
plt.grid()
plt.show()
