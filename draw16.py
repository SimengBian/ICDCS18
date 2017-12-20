import numpy as np
import matplotlib.pyplot as plt

windowSizes = [i*2 for i in range(0, 6)] + [j*5 for j in range(4, 21)]

accuracies = [100, 90, 50, 10]

# avgDelays = {accuracy: [] for accuracy in accuracies}
avgDelays = [[] for index in range(len(accuracies))]
for index in range(len(accuracies)):
    avgDelays[index] = np.load(("resultsPOSCAS_error_W/accuracy%s/avgDelays.npy" % accuracies[index]))

colors = ['ro-', 'gs-', 'bv-', 'm*-']

print(len(windowSizes))
print(len(avgDelays[0]))

fig, ax = plt.subplots(figsize=(12, 8))

plt.xlabel("Window Size", fontsize=30)
plt.ylabel("Average Delay (time slot)", fontsize=30)

# ax.set_xlim(-1, 20)
# ax.set_ylim(0, 9)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

for i in range(len(accuracies)):
    ax.plot(windowSizes, avgDelays[i], colors[i], linewidth=2, label=("accuracy: %s" % accuracies[i]), markersize=8)

ax.legend(numpoints=2, prop={'size': 26})
plt.show()
# plt.savefig("/Users/biansimeng/Desktop/figures/wsize-delay.eps")

plt.close()