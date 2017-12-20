import numpy as np
import matplotlib.pyplot as plt

filename = 'config3/'

systemInformation = np.load(filename + "System Information.npz")
# maxTime = systemInformation['maxTime']
maxTime = int(8e4)
Vs = systemInformation['Vs']
gamma = systemInformation['gamma']
# gamma = 100
lenOfVs = len(Vs)

snInformation = np.load(filename + "SN Information.npz")
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']
print(numOfServer, serverCapacities)

i = 7
print("The chosen V is: ", Vs[i])

metric = 3  # 0: queue backlogs; 1: energy costs; 2: communication costs; 3: energy costs + gamma * communication costs.
schemes = ["FF", "FFD", "NAH", ("OSCAS $(V=%s)$" % Vs[i])]

temp = ''
stride = 4000
half_stride = int(stride / 2)

queuesFF = np.load(temp + "resultsFF/timeAverageOfQueueBacklogs.npy")
energiesFF = np.load(temp + "resultsFF/timeAverageOfEnergyCosts.npy")
communicationsFF = np.load(temp + "resultsFF/timeAverageOfCommunicationCosts.npy")

queuesFFD = np.load(temp + "resultsFFD/timeAverageOfQueueBacklogs.npy")
energiesFFD = np.load(temp + "resultsFFD/timeAverageOfEnergyCosts.npy")
communicationsFFD = np.load(temp + "resultsFFD/timeAverageOfCommunicationCosts.npy")

queuesNAH = np.load(temp + "resultsNAH/timeAverageOfQueueBacklogs.npy")
energiesNAH = np.load(temp + "resultsNAH/timeAverageOfEnergyCosts.npy")
communicationsNAH = np.load(temp + "resultsNAH/timeAverageOfCommunicationCosts.npy")

queuesOSCAS = np.load(temp + "resultsOSCAS/timeAverageOfQueueBacklogs.npy")
energiesOSCAS = np.load(temp + "resultsOSCAS/timeAverageOfEnergyCosts.npy")
communicationsOSCAS = np.load(temp + "resultsOSCAS/timeAverageOfCommunicationCosts.npy")

queues = [queuesFF, queuesFFD, queuesNAH, queuesOSCAS[i]]
energies = [energiesFF, energiesFFD, energiesNAH, energiesOSCAS[i]]
communications = [communicationsFF, communicationsFFD, communicationsNAH, communicationsOSCAS[i]]
weightedSums = [energiesFF + gamma * communicationsFF, energiesFFD + gamma * communicationsFFD,
                energiesNAH + gamma * communicationsNAH,
                energiesOSCAS[i] + gamma * communicationsOSCAS[i]]

x = list(range(maxTime))
ys = [queues, energies, communications, weightedSums]

# print(len([x[0]] + x[250: new_time+1 : 500]))
# print(len([ys[metric][1][0]] + list(ys[metric][1][250: new_time+1 : 500])))

plt.figure(figsize=(12, 8))
xFF = list(x[0: maxTime: stride]) + [x[maxTime-1]]
yFF = list(ys[metric][0][0: maxTime: stride]) + [ys[metric][0][maxTime-1]]
plt.plot(xFF, yFF, 'rs-', label=schemes[0], linewidth=2, markersize=10)

xFFD = [x[0]] + x[half_stride: maxTime: stride] + [x[maxTime-1]]
yFFD = [ys[metric][1][0]] + list(ys[metric][1][half_stride: maxTime : stride]) + [ys[metric][1][maxTime-1]]
plt.plot(xFFD, yFFD, 'gv-', label=schemes[1],  linewidth=2, markersize=10)

xNAH = [x[0]] + x[half_stride: maxTime: stride] + [x[maxTime-1]]
yNAH = [ys[metric][2][0]] + list(ys[metric][2][half_stride: maxTime : stride]) + [ys[metric][2][maxTime-1]]
plt.plot(xNAH, yNAH, 'b^-', label=schemes[2], linewidth=2, markersize=10)

xOSCAS = list(x[0: maxTime: stride]) + [x[maxTime-1]]
yOSCAS = list(ys[metric][3][0: maxTime: stride]) + [ys[metric][3][maxTime-1]]
plt.plot(xOSCAS, yOSCAS, 'mo-', label=schemes[3], linewidth=2, markersize=10)

ylabels = ["Time-Avg Queue Backlog Size $(\\times 10^5)$",
          "Time-Avg Energy Cost",
          "Time-Avg Communication Cost",
          "Time-Avg Total Cost"]

plt.xlabel("Time slot $(\\times 10^4)$", fontsize=30)
plt.ylabel(ylabels[metric], fontsize=30)
plt.legend(loc=0, fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
if metric == 0:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# plt.show()
plt.savefig("/Users/biansimeng/Desktop/figures/T-cost-together.eps")

plt.close()