import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
lenOfVs = len(Vs)

i = 100
print("V: ", Vs[i])
queues = np.load("results/timeAverageOfQueueBacklogs.npy")
energies = np.load("results/timeAverageOfEnergyCosts.npy")

x = list(range(maxTime))
xx = x[100000:]
# y = queues[i]
y = energies[i]
yy = y[100000:]


# plt.ylabel("Time-Average Queue Backlogs")
plt.ylabel("Time-Average Energy Cost")

plt.xlabel("Time-slot (V=" + str(Vs[i]) + ")")

plt.plot(xx, yy, 'r-')
plt.show()

print("test")