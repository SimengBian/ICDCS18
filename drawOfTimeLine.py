import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

i = 2
print("V: ", Vs[i])
index = 0  # 0: queue backlogs; 1: energy costs; 2: partition costs; 3: energy costs + gamma * partition costs.

queues = np.load("results/timeAverageOfQueueBacklogs.npy")
energies = np.load("results/timeAverageOfEnergyCosts.npy")
partitions = np.load("results/timeAverageOfPartitionCosts.npy")

x = list(range(maxTime))

ys = [queues[i], energies[i], partitions[i], np.array(energies[i]) + gamma * np.array(partitions[i])]
labels = ["Time-Average Queue Backlogs",
          "Time-Average Energy Cost",
          "Time-Average Partition Cost",
          "Time-Average Energy and Partition Cost"]
colors = ['r', 'g', 'b', 'y']

plt.xlabel("Time-slot (V=" + str(Vs[i]) + ")")
plt.ylabel(labels[index])

plt.plot(x, ys[index], colors[index])
plt.show()

print("test")
