import random
import numpy as np

filename = 'config1/'

nfInformation = np.load(filename + "NF Information.npz")
numOfNF = int(nfInformation['numOfNF'])
'''
Service Chain (SC)
'''
numOfSC = 5  # number of Service Chain types
pOfSC = np.array(2 * [0.05] + 3 * [0.3])
# pOfSC = np.array(10 * [0.01] + 10 * [0.09])
# pOfSC = np.array(8 * [0.05] + 2 * [0.3])

lengthOfSC = 1

#  Here the service chains are generated randomly
serviceChains = {0: [15], 1: [25], 2: [0], 3: [1], 4: [2]}
# serviceChains = {0: [0, 1, 15], 1: [25, 3, 2], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
# serviceChains = {0: [15, 3, 2, 1, 0], 1: [25, 7, 6, 5, 4], 2: [0, 1, 2, 3, 4], 3: [5, 6, 7, 8, 9], 4: [10, 11, 12, 13, 14]}

#  here we must use dictionary to verify if the generated chain is new
# serviceChains = {c: [] for c in range(numOfSC)}
# c = 0
# while True:
#     if c >= numOfSC:
#         break
#
#     NFs = list(range(numOfNF))  # the networks function {0,1,2,...,F-1}
#     random.shuffle(NFs)
#     chain = NFs[0:lengthOfSC]  # the chosen service chain
#     if chain not in serviceChains.values():  # if it is new
#         serviceChains[c] = chain  # added it to the dictionary
#         c += 1

#  serviceChainsNew[c, i] is the i-th NF of SC type c
serviceChainsNew = (-1) * np.ones((numOfSC, lengthOfSC), dtype=int)
for c in range(numOfSC):
    for i in range(lengthOfSC):
        serviceChainsNew[c, i] = serviceChains[c][i]

print(serviceChainsNew)
np.savez(filename + "SC Information.npz", numOfSC=numOfSC, lengthOfSC=lengthOfSC, serviceChains=serviceChainsNew, pOfSC=pOfSC,)
