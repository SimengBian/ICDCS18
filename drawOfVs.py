import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
gamma = systemInformation['gamma']
Vs = systemInformation['Vs']
lenOfVs = len(Vs)

index = 3  # 0: queue backlogs; 1: energy costs; 2: partition costs; 3: energy costs + gamma * partition costs.

queues = np.load("results/timeAverageOfQueueBacklogs.npy")
energies = np.load("results/timeAverageOfEnergyCosts.npy")
partitions = np.load("results/timeAverageOfPartitionCosts.npy")

x = Vs
ys = [[], [], [], []]
for i in range(lenOfVs):
    ys[0].append(queues[i, maxTime-1])
    ys[1].append(energies[i, maxTime-1])
    ys[2].append(partitions[i, maxTime - 1])
    ys[3].append(energies[i, maxTime-1] + gamma * partitions[i, maxTime - 1])

labels = ["Time-Average Queue Backlogs at Time-slot " + str(maxTime),
          "Time-Average Energy Cost at Time-slot " + str(maxTime),
          "Time-Average Partition Cost at Time-slot " + str(maxTime),
          "Time-Average Energy and Partition Cost at Time-slot " + str(maxTime)]
colors = ['r', 'g', 'b', 'y']

plt.xlabel("V")
plt.ylabel(labels[index])

plt.plot(x, ys[index], colors[index])
plt.show()

print("test")