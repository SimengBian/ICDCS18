import numpy as np

filename = "config1/"
'''
Substrate Network (SN)
'''
numOfServer = 6  # number of servers

serverCapacities = np.zeros(numOfServer)
for c in range(numOfServer):
    serverCapacities[c] = 16

idleEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    idleEnergies[c] = 0.805

maxEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    maxEnergies[c] = 27.35

np.savez(filename + "SN Information.npz", numOfServer=numOfServer, serverCapacities=serverCapacities, idleEnergies=idleEnergies, maxEnergies=maxEnergies)
