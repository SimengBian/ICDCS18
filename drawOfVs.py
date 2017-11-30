import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
gamma = systemInformation['gamma']
Vs = systemInformation['Vs']
lenOfVs = len(Vs)

metric = 3  # 0: queue backlogs; 1: energy costs; 2: comm. costs; 3: energy costs + gamma * partition costs.

queues = np.load("resultsONRA/timeAverageOfQueueBacklogs.npy")
energies = np.load("resultsONRA/timeAverageOfEnergyCosts.npy")
partitions = np.load("resultsONRA/timeAverageOfPartitionCosts.npy")

x = Vs
ys = [[], [], [], []]
for i in range(lenOfVs):
    ys[0].append(queues[i, maxTime-1])
    ys[1].append(energies[i, maxTime-1])
    ys[2].append(partitions[i, maxTime - 1])
    ys[3].append(energies[i, maxTime-1] + gamma * partitions[i, maxTime - 1])

# print(ys[1])

ylabels = ["Time-Avg Queue Backlog Size $(\\times 10^4)$",
          "Time-Avg Energy Cost",
          "Time-Avg Communication Cost",
          "Time-Avg Total Cost"]
colors = ['ro-', 'go-', 'bo-', 'mo-']

plt.figure(figsize=(10, 8))
plt.xlabel("V", fontsize=30)
plt.ylabel(ylabels[metric], fontsize=30)
plt.plot(x, ys[metric], colors[metric], linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if metric == 0:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


plt.show()
