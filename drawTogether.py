import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

i = 9
print("V: ", Vs[i])

metric = 0  # 0: queue backlogs; 1: energy costs; 2: communication costs; 3: energy costs + gamma * communication costs.

scheme = ["FF", "FFD", "NAH", "ONRA", ]

queuesFF = np.load("resultsFF/timeAverageOfQueueBacklogs.npy")
energiesFF = np.load("resultsFF/timeAverageOfEnergyCosts.npy")
communicationsFF = np.load("resultsFF/timeAverageOfCommunicationCosts.npy")

queuesFFD = np.load("resultsFFD/timeAverageOfQueueBacklogs.npy")
energiesFFD = np.load("resultsFFD/timeAverageOfEnergyCosts.npy")
communicationsFFD = np.load("resultsFFD/timeAverageOfCommunicationCosts.npy")

queuesNAH = np.load("resultsNAH/timeAverageOfQueueBacklogs.npy")
energiesNAH = np.load("resultsNAH/timeAverageOfEnergyCosts.npy")
communicationsNAH = np.load("resultsNAH/timeAverageOfCommunicationCosts.npy")

queuesONRA = np.load("resultsONRA/timeAverageOfQueueBacklogs.npy")
energiesONRA = np.load("resultsONRA/timeAverageOfEnergyCosts.npy")
communicationsONRA = np.load("resultsONRA/timeAverageOfCommunicationCosts.npy")

x = list(range(maxTime))
ys = [[queuesFF, queuesFFD, queuesNAH, queuesONRA[i]],
      [energiesFF, energiesFFD, energiesNAH, energiesONRA[i]],
      [communicationsFF, communicationsFFD, communicationsNAH, communicationsONRA[i]],
      [energiesFF+communicationsFF, energiesFFD+communicationsFFD, energiesNAH+communicationsNAH,energiesONRA[i]+communicationsONRA[i]]]

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
