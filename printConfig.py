import numpy as np
'''
configurations
'''
scInformation = np.load("config/SC Information.npz")
nfInformation = np.load("config/NF Information.npz")
snInformation = np.load("config/SN Information.npz")
systemInformation = np.load("config/System Information.npz")

# System Information
maxTime = systemInformation['maxTime']
gamma = systemInformation['gamma']
Vs = systemInformation['Vs']
lenOfVs = len(Vs)
arrivals = systemInformation['arrivals']  # arrivals[c, t]
pCost = systemInformation['pCost']

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCost = nfInformation['processingCost']  # processingCost[f]

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']  # serviceChains[c, i]

# Substrate Network Information (mainly about the servers)
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']  # serverCapacities[s]
idleEnergies = snInformation['idleEnergies']  # idleEnergies[s]
maxEnergies = snInformation['maxEnergies']  # maxEnergies[s]

print(numOfNF)
print(processingCost)

print(numOfSC)
print(lengthOfSC)

print(numOfServer)
print(serverCapacities)

print(maxTime)
print(Vs)
