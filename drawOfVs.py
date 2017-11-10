import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
lenOfVs = len(Vs)

queues = np.load("results/timeAverageOfQueueBacklogs.npy")
energies = np.load("results/timeAverageOfEnergyCosts.npy")

x = Vs
y = []
for i in range(lenOfVs):
    y.append(queues[i, maxTime-1])
    # y.append(energies[i, maxTime-1])

plt.ylabel("Time-Average Queue Backlogs")
# plt.ylabel("Time-Average Energy Cost")
plt.xlabel("V")

plt.plot(x, y, 'b-')
plt.show()

# print(x)
# print(y)

print("test")