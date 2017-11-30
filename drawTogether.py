import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

i = 9
print("V: ", Vs[i])

metric = 0  # 0: queue backlogs; 1: energy costs; 2: partition costs; 3: energy costs + gamma * partition costs.

scheme = ["FF", "FFD", "NAH", "ONRA", ]

queuesFF = np.load("resultsFF/timeAverageOfQueueBacklogs.npy")
energiesFF = np.load("resultsFF/timeAverageOfEnergyCosts.npy")
partitionsFF = np.load("resultsFF/timeAverageOfPartitionCosts.npy")

queuesFFD = np.load("resultsFFD/timeAverageOfQueueBacklogs.npy")
energiesFFD = np.load("resultsFFD/timeAverageOfEnergyCosts.npy")
partitionsFFD = np.load("resultsFFD/timeAverageOfPartitionCosts.npy")

queuesNAH = np.load("resultsNAH/timeAverageOfQueueBacklogs.npy")
energiesNAH = np.load("resultsNAH/timeAverageOfEnergyCosts.npy")
partitionsNAH = np.load("resultsNAH/timeAverageOfPartitionCosts.npy")

queuesONRA = np.load("resultsONRA/timeAverageOfQueueBacklogs.npy")
energiesONRA = np.load("resultsONRA/timeAverageOfEnergyCosts.npy")
partitionsONRA = np.load("resultsONRA/timeAverageOfPartitionCosts.npy")

x = list(range(maxTime))
ys = [[queuesFF, queuesFFD, queuesNAH, queuesONRA[i]],
      [energiesFF, energiesFFD, energiesNAH, energiesONRA[i]],
      [partitionsFF, partitionsFFD, partitionsNAH, partitionsONRA[i]],
      [energiesFF+partitionsFF, energiesFFD+partitionsFFD, energiesNAH+partitionsNAH,energiesONRA[i]+partitionsONRA[i]]]

plt.figure(figsize=(10, 8))
plt.plot(x, ys[metric][0], 'r-', label=scheme[0], linewidth=2)
plt.plot(x, ys[metric][1], 'g--', label=scheme[1],  linewidth=2)
plt.plot(x, ys[metric][2], 'b-', label=scheme[2], linewidth=2)
plt.plot(x, ys[metric][3], 'y-', label=scheme[3] + ' (V=' + str(Vs[i]) + ')', linewidth=2)

ylabels = ["Time-Avg Queue Backlog Size $(\\times 10^4)$",
          "Time-Avg Energy Cost",
          "Time-Avg Communication Cost",
          "Time-Avg Total Cost"]

plt.xlabel("Time-slot", fontsize=30)
plt.ylabel(ylabels[metric], fontsize=30)
plt.legend(loc=0, fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if metric == 0:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


plt.show()

print("test")
