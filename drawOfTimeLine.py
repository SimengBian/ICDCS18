import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

i = 39
print("V: ", Vs[i])
metric = 0  # 0: queue backlogs; 1: energy costs; 2: partition costs; 3: energy costs + gamma * partition costs.
scheme = 3  # 0: FF; 1: FFD; 2: NAH; 3: ONRA

schemes = ["FF", "FFD", "NAH", "ONRA", ]

queues = np.load("results%s/timeAverageOfQueueBacklogs.npy" % schemes[scheme])
energies = np.load("results%s/timeAverageOfEnergyCosts.npy" % schemes[scheme])
partitions = np.load("results%s/timeAverageOfPartitionCosts.npy" % schemes[scheme])

x = list(range(maxTime))


if scheme == 3:
    ys = [queues[i], energies[i], partitions[i], np.array(energies[i]) + gamma * np.array(partitions[i])]
    plt.xlabel("Time-slot (V=" + str(Vs[i]) + ")")
else:
    ys = [queues, energies, partitions, np.array(energies) + gamma * np.array(partitions)]
    plt.xlabel("Time-slot")

labels = ["Time-Average Queue Backlogs",
          "Time-Average Energy Cost",
          "Time-Average Partition Cost",
          "Time-Average Energy and Partition Cost"]
colors = ['r', 'g', 'b', 'y']

plt.ylabel(labels[metric])
plt.title(schemes[scheme])

plt.plot(x, ys[metric], colors[metric])
plt.show()

print("test")
