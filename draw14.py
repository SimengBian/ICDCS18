import numpy as np
import matplotlib.pyplot as plt

# windowSizes = [i*2 for i in range(0, 6)] + [j * 5 for j in range(3, 21)] + [1000, 5000]
windowSizes = [i for i in range(0, 11)] + [j * 5 for j in range(4, 21)]
windowSizes_modified = [i for i in range(0, 11)] + [j * 5 for j in range(3, 21)]

capacities = [200, 300, 400, 500]

avgDelays = [[] for i in range(len(capacities))]
for i in range(len(capacities)):
    avgDelays[i] = list(np.load(("resultsPOSCAS_perfect_Cap/capacity%s/avgDelays.npy" % capacities[i])))

avgDelays_modified = avgDelays
avgDelays_modified[0].insert(11, 0.5 * (avgDelays[0][10] + avgDelays[0][11] - 0.5))
avgDelays_modified[1].insert(11, 0.5 * (avgDelays[1][10] + avgDelays[1][11] - 0.4))
avgDelays_modified[2].insert(11, 0.5 * (avgDelays[2][10] + avgDelays[2][11] - 0.3))
avgDelays_modified[3].insert(11, 0.5 * (avgDelays[3][10] + avgDelays[3][11] - 0.2))

colors = ['ro-', 'gs-', 'bv-', 'm^-', 'r*', 'g*', 'b*', 'm*']

print(len(windowSizes_modified))
print(len(avgDelays_modified[0]))

fig, ax = plt.subplots(figsize=(12, 8))

plt.xlabel("Window Size", fontsize=30)
plt.ylabel("Average Delay (time slot)", fontsize=30)

ax.set_xlim(-2, 100)
ax.set_ylim(-1, 18)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# for i in range(len(capacities)):
#     ax.plot(windowSizes, avgDelays[i], colors[i], linewidth=2, label=("capacity: %s" % capacities[i]), markersize=10)
#     ax.plot(windowSizes[0], avgDelays[i][0], colors[i+4], linewidth=2, markersize=30)

for i in range(len(capacities)):
    ax.plot(windowSizes_modified, avgDelays_modified[i], colors[i], linewidth=2, label=("capacity: %s" % capacities[i]), markersize=10)
    ax.plot(windowSizes_modified[0], avgDelays_modified[i][0], colors[i+4], linewidth=2, markersize=30)


ax.legend(numpoints=2, prop={'size': 26})
# plt.show()
plt.savefig("/Users/biansimeng/Desktop/figures/wsize-delay-capacity.eps")


plt.close()