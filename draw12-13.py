import numpy as np
import matplotlib.pyplot as plt

Vs = [i*5 for i in range(1, 21)]
gamma = 100
maxTime = int(1e2)
metric = 0  # 0: queue backlogs; 1: energy costs; 2: comm. costs; 3: energy costs + gamma * communication costs.
accuracies = [100, 90, 50, 10]

data = [[] for index in range(len(accuracies))]
for index in range(len(accuracies)):
    if metric == 0:
        data[index] = np.load(("resultsPOSCAS_error_V/accuracy%s/timeAverageOfQueueBacklogs.npy" % accuracies[index]))
    elif metric == 1:
        data[index] = np.load(("resultsPOSCAS_error_V/accuracy%s/timeAverageOfEnergyCosts.npy" % accuracies[index]))
    elif metric == 2:
        data[index] = np.load(("resultsPOSCAS_error_V/accuracy%s/timeAverageOfCommunicationCosts.npy" % accuracies[index]))
    elif metric == 3:
        term1 = np.load(("resultsPOSCAS_error_V/accuracy%s/timeAverageOfEnergyCosts.npy" % accuracies[index]))
        term2 = np.load(("resultsPOSCAS_error_V/accuracy%s/timeAverageOfCommunicationCosts.npy" % accuracies[index]))
        data[index] = term1 + gamma * term2
    else:
        raise Exception("Illegal metric.")

ys = [[] for index in range(len(accuracies))]
for index in range(len(accuracies)):
    for i in range(len(Vs)):
        ys[index].append(data[index][i, maxTime - 1])

colors = ['ro-', 'gs-', 'bv-', 'm*-']

print(len(Vs))
print(len(ys[0]))

fig, ax = plt.subplots(figsize=(12, 8))

plt.xlabel("Window Size", fontsize=30)
plt.ylabel("Average Delay (time slot)", fontsize=30)

# ax.set_xlim(-1, 20)
# ax.set_ylim(0, 9)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for i in range(len(accuracies)):
    ax.plot(Vs, ys[i], colors[i], linewidth=2, label=("accuracy: %s" % accuracies[i]), markersize=8)

ax.legend(numpoints=2, prop={'size': 26})
plt.show()
# plt.savefig("/Users/biansimeng/Desktop/figures/wsize-delay.eps")

plt.close()