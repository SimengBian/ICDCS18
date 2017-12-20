import numpy as np

filename = 'configTest/'
'''
Network Function (NF)
'''
numOfNF = 3  # number of NF types

# processingCosts = np.array([5, 2, 1])
processingCosts = np.array([1, 1, 1])

'''
Service Chain (SC)
'''
numOfSC = 2  # number of Service Chain types
pOfSC = np.array(2 * [0.05] + 3 * [0.3])
# pOfSC = np.array(10 * [0.01] + 10 * [0.09])
# pOfSC = np.array(8 * [0.05] + 2 * [0.3])

lengthOfSC = 3

#  Here the service chains are generated randomly
serviceChains = {0: [0, 1, 2], 1: [2, 0, 1]}

#  serviceChainsNew[c, i] is the i-th NF of SC type c
serviceChainsNew = (-1) * np.ones((numOfSC, lengthOfSC), dtype=int)
for c in range(numOfSC):
    for i in range(lengthOfSC):
        serviceChainsNew[c, i] = serviceChains[c][i]

print(serviceChainsNew)

'''
Substrate Network (SN)
'''
numOfServer = 2  # number of servers

serverCapacities = np.zeros(numOfServer)
for c in range(numOfServer):
    serverCapacities[c] = 100

idleEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    idleEnergies[c] = 0

maxEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    maxEnergies[c] = 10

'''
System Information
'''
arrivalRate = 5.88
maxTime = int(10)
Vs = [1]
# Vs = [i*5 for i in range(1, 21)]
alpha = 1
gamma = 100
unitCommCost = 1
maxWindowSize = 20

#  arrivals[c, t] is the number of arrival requests of SC type c at time-slot t
arrivals = np.array([maxTime * [2], maxTime * [5]])
arrivals_error = np.array([maxTime * [3], maxTime * [4]])

print("Save")
np.savez(filename + "System Information.npz", unitCommCost=unitCommCost, maxTime=maxTime, arrivals=arrivals, Vs=Vs, gamma=gamma, alpha=alpha, maxWindowSize=maxWindowSize)
np.savez(filename + "SN Information.npz", numOfServer=numOfServer, serverCapacities=serverCapacities, idleEnergies=idleEnergies, maxEnergies=maxEnergies)
np.savez(filename + "SC Information.npz", numOfSC=numOfSC, lengthOfSC=lengthOfSC, serviceChains=serviceChainsNew, pOfSC=pOfSC,)
np.savez(filename + "NF Information.npz", numOfNF=numOfNF, processingCosts=processingCosts)
np.savez(filename + "Arrival_error.npz", arrivals_error=arrivals_error)
