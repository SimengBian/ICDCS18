import numpy as np
import matplotlib.pyplot as plt

# windowSizes = [i*2 for i in range(0, 6)] + [j * 5 for j in range(3, 21)] + [1000, 5000]
windowSizes = [i for i in range(21)]

lenSFCs = [1, 3, 5]

avgDelays = [[] for i in range(len(lenSFCs))]
for i in range(len(lenSFCs)):
    avgDelays[i] = np.load(("resultsPOSCAS_perfect_Len/length%s/avgDelays.npy" % lenSFCs[i]))

colors = ['ro-', 'gs-', 'bv-', 'm^-', 'r*', 'g*', 'b*', 'm*']


wsizes = windowSizes
# print(len(wsizes))
# print(len(avgDelays[0]))
# print(len(wsizes))
# print(len(avgDelays[1]))
# print(len(wsizes))
# print(len(avgDelays[2]))

fig, ax = plt.subplots(figsize=(12, 8))

plt.xlabel("Window Size", fontsize=30)
plt.ylabel("Average Delay (time slot)", fontsize=30)

ax.set_xlim(-1, 20)
ax.set_ylim(0, 15)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


for i in range(len(lenSFCs)):
    ax.plot(wsizes, avgDelays[i], colors[i], linewidth=2, label=("SFC length: %s" % lenSFCs[i]), markersize=10)
    # plt.hold(True)
    ax.plot(wsizes[0], avgDelays[i][0], colors[i+4], linewidth=2, markersize=30)
    # plt.hold(True)

ax.legend(numpoints=1, prop={'size': 26})
# plt.show()
plt.savefig("/Users/biansimeng/Desktop/figures/wsize-delay-length.eps")


plt.close()