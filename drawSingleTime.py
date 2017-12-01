import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
alpha = systemInformation['alpha']
lenOfVs = len(Vs)

i = 0
print("V: ", Vs[i])
metric = 0  # 0: queue backlogs; 1: energy costs; 2: communication costs; 3: energy costs + gamma * communication costs.
scheme = 3  # 0: FF; 1: FFD; 2: NAH; 3: ONRA; 4: P-ONRA
windowSize = 1

schemes = ["FF", "FFD", "NAH", "ONRA", "P-ONRA"]

if scheme != 4:
    queues = np.load("results%s/timeAverageOfQueueBacklogs.npy" % schemes[scheme])
    energies = np.load("results%s/timeAverageOfEnergyCosts.npy" % schemes[scheme])
    communications = np.load("results%s/timeAverageOfCommunicationCosts.npy" % schemes[scheme])
else:
    queues = np.load("resultsP-ONRA/wsize%s/timeAverageOfQueueBacklogs.npy" % windowSize)
    energies = np.load("resultsP-ONRA/wsize%s/timeAverageOfEnergyCosts.npy" % windowSize)
    communications = np.load("resultsP-ONRA/wsize%s/timeAverageOfCommunicationCosts.npy" % windowSize)

x = list(range(maxTime))


if scheme in [3, 4]:
    ys = [queues[i], energies[i], communications[i], np.array(energies[i]) + gamma * np.array(communications[i])]
    plt.xlabel("Time-slot (V=" + str(Vs[i]) + ")")
else:
    ys = [queues, energies, communications, np.array(energies) + gamma * np.array(communications)]
    plt.xlabel("Time-slot")

labels = ["Time-Average Queue Backlogs",
          "Time-Average Energy Cost",
          "Time-Average Communication Cost",
          "Time-Average Energy and Communication Cost"]
colors = ['r', 'g', 'b', 'y']

# plt.axis([0, 100000, 4, 5])
plt.ylabel(labels[metric])
if scheme != 4:
    plt.title(schemes[scheme])
else:
    plt.title("P-ONRA (wsize = %s)" % windowSize)

plt.plot(x, ys[metric], colors[metric])
plt.show()

print("test")
