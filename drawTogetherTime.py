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

queuesPONRA = np.load("resultsP-ONRA/timeAverageOfQueueBacklogs.npy")
energiesPONRA = np.load("resultsP-ONRA/timeAverageOfEnergyCosts.npy")
communicationsPONRA = np.load("resultsP-ONRA/timeAverageOfCommunicationCosts.npy")

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

queues = [queuesFF, queuesFFD, queuesNAH, queuesONRA[i], queuesPONRA[i],
          queuesPONRA0[i], queuesPONRA1[i], queuesPONRA5[i], queuesPONRA10[i], queuesPONRA100[i]]
energies = [energiesFF, energiesFFD, energiesNAH, energiesONRA[i], energiesPONRA[i],
            energiesPONRA0[i], energiesPONRA1[i], energiesPONRA5[i], energiesPONRA10[i], energiesPONRA100[i]]
communications = [communicationsFF, communicationsFFD, communicationsNAH, communicationsONRA[i], communicationsPONRA[i],
                  communicationsPONRA0[i], communicationsPONRA1[i], communicationsPONRA5[i], communicationsPONRA10[i], communicationsPONRA100[i]]
weightedSums = [energiesFF+communicationsFF, energiesFFD+communicationsFFD, energiesNAH+communicationsNAH,
               energiesONRA[i]+communicationsONRA[i], energiesPONRA[i]+communicationsPONRA[i],
               energiesPONRA0[i]+communicationsPONRA0[i], energiesPONRA1[i]+communicationsPONRA1[i],
               energiesPONRA5[i]+communicationsPONRA5[i], energiesPONRA10[i]+communicationsPONRA10[i],
               energiesPONRA100[i]+communicationsPONRA100[i]]

x = list(range(maxTime))
ys = [queues, energies, communications, weightedSums]

plt.figure(figsize=(10, 8))
plt.plot(x, ys[metric][0], 'r-', label=schemes[0], linewidth=2)
plt.plot(x, ys[metric][1], 'g--', label=schemes[1],  linewidth=2)
plt.plot(x, ys[metric][2], 'b-', label=schemes[2], linewidth=2)
plt.plot(x, ys[metric][3], 'y-', label=schemes[3] + ' (V=' + str(Vs[i]) + ')', linewidth=2)
plt.plot(x, ys[metric][4], 'm-', label="P-ONRA (wsize = 0)", linewidth=2)
plt.plot(x, ys[metric][5], 'm--', label="P-ONRA (wsize = 1)", linewidth=2)
plt.plot(x, ys[metric][6], 'm.-', label="P-ONRA (wsize = 5)", linewidth=2)
plt.plot(x, ys[metric][7], 'mo-', label="P-ONRA (wsize = 10)", linewidth=2)
plt.plot(x, ys[metric][8], 'm*-', label="P-ONRA (wsize = 1000)", linewidth=2)

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

