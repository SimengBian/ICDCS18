import numpy as np
import matplotlib.pyplot as plt

systemInformation = np.load("config/System Information.npz")
maxTime = systemInformation['maxTime']
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
lenOfVs = len(Vs)

metric = 0  # 0: queue backlogs; 1: energy costs; 2: communication costs; 3: energy costs + gamma * communication costs.

schemes = ["FF", "FFD", "NAH", "ONRA", "P-ONRA"]
# windowSizes = [0, 1, 5, 10, 100]

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

queuesPONRA0 = np.load("resultsP-ONRA/wsize0/timeAverageOfQueueBacklogs.npy")
energiesPONRA0 = np.load("resultsP-ONRA/wsize0/timeAverageOfEnergyCosts.npy")
communicationsPONRA0 = np.load("resultsP-ONRA/wsize0/timeAverageOfCommunicationCosts.npy")

queuesPONRA1 = np.load("resultsP-ONRA/wsize1/timeAverageOfQueueBacklogs.npy")
energiesPONRA1 = np.load("resultsP-ONRA/wsize1/timeAverageOfEnergyCosts.npy")
communicationsPONRA1 = np.load("resultsP-ONRA/wsize1/timeAverageOfCommunicationCosts.npy")

queuesPONRA5 = np.load("resultsP-ONRA/wsize5/timeAverageOfQueueBacklogs.npy")
energiesPONRA5 = np.load("resultsP-ONRA/wsize5/timeAverageOfEnergyCosts.npy")
communicationsPONRA5 = np.load("resultsP-ONRA/wsize5/timeAverageOfCommunicationCosts.npy")

queuesPONRA10 = np.load("resultsP-ONRA/wsize10/timeAverageOfQueueBacklogs.npy")
energiesPONRA10 = np.load("resultsP-ONRA/wsize10/timeAverageOfEnergyCosts.npy")
communicationsPONRA10 = np.load("resultsP-ONRA/wsize10/timeAverageOfCommunicationCosts.npy")

queuesPONRA100 = np.load("resultsP-ONRA/wsize100/timeAverageOfQueueBacklogs.npy")
energiesPONRA100 = np.load("resultsP-ONRA/wsize100/timeAverageOfEnergyCosts.npy")
communicationsPONRA100 = np.load("resultsP-ONRA/wsize100/timeAverageOfCommunicationCosts.npy")

queues = [queuesFF, queuesFFD, queuesNAH, queuesONRA,
          queuesPONRA0, queuesPONRA1, queuesPONRA5, queuesPONRA10, queuesPONRA100]
energies = [energiesFF, energiesFFD, energiesNAH, energiesONRA,
            energiesPONRA0, energiesPONRA1, energiesPONRA5, energiesPONRA10, energiesPONRA100]
communications = [communicationsFF, communicationsFFD, communicationsNAH, communicationsONRA,
                  communicationsPONRA0, communicationsPONRA1, communicationsPONRA5, communicationsPONRA10, communicationsPONRA100]
weightedSums = [energiesFF+communicationsFF, energiesFFD+communicationsFFD, energiesNAH+communicationsNAH,
               energiesONRA+communicationsONRA,
               energiesPONRA0+communicationsPONRA0, energiesPONRA1+communicationsPONRA1,
               energiesPONRA5+communicationsPONRA5, energiesPONRA10+communicationsPONRA10,
               energiesPONRA100+communicationsPONRA100]

x = Vs
# ys = [queues, energies, communications, weightedSums]
ys = [[[] for i in range(9)], [[] for j in range(9)], [[] for k in range(9)], [[] for l in range(9)]]  # ys[metric][0], ys[metric][7]
for i in range(lenOfVs):
    ys[0][0].append(queues[0][maxTime-1])
    ys[0][1].append(queues[1][maxTime - 1])
    ys[0][2].append(queues[2][maxTime - 1])
    ys[0][3].append(queues[3][i, maxTime - 1])
    ys[0][4].append(queues[4][i, maxTime - 1])
    ys[0][5].append(queues[5][i, maxTime - 1])
    ys[0][6].append(queues[6][i, maxTime - 1])
    ys[0][7].append(queues[7][i, maxTime - 1])
    ys[0][8].append(queues[8][i, maxTime - 1])

    ys[1][0].append(energies[0][maxTime-1])
    ys[1][1].append(energies[1][maxTime - 1])
    ys[1][2].append(energies[2][maxTime - 1])
    ys[1][3].append(energies[3][i, maxTime - 1])
    ys[1][4].append(energies[4][i, maxTime - 1])
    ys[1][5].append(energies[5][i, maxTime - 1])
    ys[1][6].append(energies[6][i, maxTime - 1])
    ys[1][7].append(energies[7][i, maxTime - 1])
    ys[1][8].append(energies[8][i, maxTime - 1])

    ys[2][0].append(communications[0][maxTime-1])
    ys[2][1].append(communications[1][maxTime - 1])
    ys[2][2].append(communications[2][maxTime - 1])
    ys[2][3].append(communications[3][i, maxTime - 1])
    ys[2][4].append(communications[4][i, maxTime - 1])
    ys[2][5].append(communications[5][i, maxTime - 1])
    ys[2][6].append(communications[6][i, maxTime - 1])
    ys[2][7].append(communications[7][i, maxTime - 1])
    ys[2][8].append(communications[8][i, maxTime - 1])

    ys[3][0].append(weightedSums[0][maxTime-1])
    ys[3][1].append(weightedSums[1][maxTime - 1])
    ys[3][2].append(weightedSums[2][maxTime - 1])
    ys[3][3].append(weightedSums[3][i, maxTime - 1])
    ys[3][4].append(weightedSums[4][i, maxTime - 1])
    ys[3][5].append(weightedSums[5][i, maxTime - 1])
    ys[3][6].append(weightedSums[6][i, maxTime - 1])
    ys[3][7].append(weightedSums[7][i, maxTime - 1])
    ys[3][8].append(weightedSums[8][i, maxTime - 1])

plt.figure(figsize=(10, 8))
plt.plot(x, ys[metric][0], 'r-', label=schemes[0], linewidth=2)
plt.plot(x, ys[metric][1], 'g--', label=schemes[1],  linewidth=2)
plt.plot(x, ys[metric][2], 'b-', label=schemes[2], linewidth=2)
plt.plot(x, ys[metric][3], 'y-', label=schemes[3], linewidth=2)
plt.plot(x, ys[metric][4], 'm-', label="P-ONRA (wsize = 0)", linewidth=2)
plt.plot(x, ys[metric][5], 'm--', label="P-ONRA (wsize = 1)", linewidth=2)
plt.plot(x, ys[metric][6], 'm.-', label="P-ONRA (wsize = 5)", linewidth=2)
plt.plot(x, ys[metric][7], 'mo-', label="P-ONRA (wsize = 10)", linewidth=2)
plt.plot(x, ys[metric][8], 'm*-', label="P-ONRA (wsize = 1000)", linewidth=2)

ylabels = ["Time-Avg Queue Backlog Size $(\\times 10^4)$",
          "Time-Avg Energy Cost",
          "Time-Avg Communication Cost",
          "Time-Avg Total Cost"]

plt.xlabel("V", fontsize=30)
plt.ylabel(ylabels[metric], fontsize=30)
plt.legend(loc=0, fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if metric == 0:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.show()
