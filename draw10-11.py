import numpy as np
import matplotlib.pyplot as plt

filename = 'config3/'

systemInformation = np.load(filename + "System Information.npz")
# maxTime = systemInformation['maxTime']
maxTime = int(8e4)
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

snInformation = np.load(filename + "SN Information.npz")
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']
print(numOfServer, serverCapacities)

temp = ""
metric = 0  # 0: queue backlogs; 1: energy costs; 2: comm. costs; 3: energy costs + gamma * communication costs.
stride = 2

queues_OSCAS = np.load(temp + "resultsOSCAS/" + "timeAverageOfQueueBacklogs.npy")
energies_OSCAS = np.load(temp + "resultsOSCAS/" + "timeAverageOfEnergyCosts.npy")
communications_OSCAS = np.load(temp + "resultsOSCAS/" + "timeAverageOfCommunicationCosts.npy")

queues_FF = np.load(temp + "resultsFF/" + "timeAverageOfQueueBacklogs.npy")
energies_FF = np.load(temp + "resultsFF/" + "timeAverageOfEnergyCosts.npy")
communications_FF = np.load(temp + "resultsFF/" + "timeAverageOfCommunicationCosts.npy")

queues_FFD = np.load(temp + "resultsFFD/" + "timeAverageOfQueueBacklogs.npy")
energies_FFD = np.load(temp + "resultsFFD/" + "timeAverageOfEnergyCosts.npy")
communications_FFD = np.load(temp + "resultsFFD/" + "timeAverageOfCommunicationCosts.npy")

queues_NAH = np.load(temp + "resultsNAH/" + "timeAverageOfQueueBacklogs.npy")
energies_NAH = np.load(temp + "resultsNAH/" + "timeAverageOfEnergyCosts.npy")
communications_NAH = np.load(temp + "resultsNAH/" + "timeAverageOfCommunicationCosts.npy")

x = Vs
ys_OSCAS = [[], [], [], []]
ys_FF = [[queues_FF[maxTime - 1] for V in Vs], [energies_FF[maxTime - 1] for V in Vs],
         [communications_FF[maxTime - 1] for V in Vs], [energies_FF[maxTime - 1] + gamma * communications_FF[maxTime - 1] for V in Vs]]
ys_FFD = [[queues_FFD[maxTime - 1] for V in Vs], [energies_FFD[maxTime - 1] for V in Vs],
         [communications_FFD[maxTime - 1] for V in Vs], [energies_FFD[maxTime - 1] + gamma * communications_FFD[maxTime - 1] for V in Vs]]
ys_NAH = [[queues_NAH[maxTime - 1] for V in Vs], [energies_NAH[maxTime - 1] for V in Vs],
         [communications_NAH[maxTime - 1] for V in Vs], [energies_NAH[maxTime - 1] + gamma * communications_NAH[maxTime - 1] for V in Vs]]

for i in range(lenOfVs):
    ys_OSCAS[0].append(queues_OSCAS[i, maxTime-1])  #+ 200 * abs(10-i)
    ys_OSCAS[1].append(energies_OSCAS[i, maxTime-1])
    ys_OSCAS[2].append(communications_OSCAS[i, maxTime - 1])
    ys_OSCAS[3].append(energies_OSCAS[i, maxTime-1] + gamma * communications_OSCAS[i, maxTime - 1])

# print(ys[1])

ylabels = ["Time-Avg Queue Backlog Size $(\\times 10^5)$",
          "Time-Avg Energy Cost",
          "Time-Avg Communication Cost",
          "Time-Avg Total Cost"]
colors = ['ro-', 'go-', 'bo-', 'mo-']

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel("V", fontsize=30)
plt.ylabel(ylabels[metric], fontsize=30)

ax.set_xlim(4, 100)
# ax.set_ylim(200, 800)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

if metric == 0:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# t1 = x[0:lenOfVs:2]
# t2 = [ys_FFD[metric][0]] + ys_FFD[metric][1:lenOfVs:2] + [ys_FFD[metric][lenOfVs-1]]
# print(t1)
# print(t2)
ax.plot(list(x[0:lenOfVs:stride]) + [Vs[lenOfVs-1]], list(ys_FF[metric][0:lenOfVs:stride]) + [ys_FF[metric][lenOfVs-1]], 'rs-', linewidth=2, label='FF', markersize=10)
ax.plot([x[0]] + list(x[1:lenOfVs:stride]), [ys_FFD[metric][0]] + list(ys_FFD[metric][1:lenOfVs:stride]), 'gv-', linewidth=2, label='FFD', markersize=10)
ax.plot(x[0:lenOfVs], ys_NAH[metric][0:lenOfVs], 'b^-', linewidth=2, label='NAH', markersize=10)
ax.plot(x[0:lenOfVs], ys_OSCAS[metric][0:lenOfVs], 'mo-', linewidth=2, label='OSCAS', markersize=10)


ax.legend(numpoints=2, prop={'size': 25}, loc=0)

# plt.show()
plt.savefig("/Users/biansimeng/Desktop/figures/V-queue-OSCAS.eps")

plt.close()