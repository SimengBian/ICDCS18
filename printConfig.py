import numpy as np

fileIndex = 1

filenames = {1: 'config1/',
             3: 'config3/',
             5: 'config5/',
             10: 'configAccuracy10/',
             50: 'configAccuracy50/',
             90: 'configAccuracy90/',
             100: 'configAccuracy100/',
            -1: 'configTest/'}

"""
configurations
"""
scInformation = np.load(filenames[fileIndex] + "SC Information.npz")
nfInformation = np.load(filenames[fileIndex] + "NF Information.npz")
snInformation = np.load(filenames[fileIndex]+ "SN Information.npz")
systemInformation = np.load(filenames[fileIndex] + "System Information.npz")

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
print("Number of NFs: ", numOfNF)
processingCosts = nfInformation['processingCosts']
print("Processing Costs: ", processingCosts)

# print("**********************************************")

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
print("Number of SFCs: ", numOfSC)
lengthOfSC = int(scInformation['lengthOfSC'])
print("Length of SFC: ", lengthOfSC)
serviceChains = scInformation['serviceChains']
print("Service Function Chains: ")
print(serviceChains)

# print("**********************************************")

# Substrate Network Information
numOfServer = int(snInformation['numOfServer'])
print("Number of servers: ", numOfServer)
serverCapacities = snInformation['serverCapacities']
print("Server Capacities: ", serverCapacities)
idleEnergies = snInformation['idleEnergies']
maxEnergies = snInformation['maxEnergies']

# print("**********************************************")

# System Information
maxTime = int(systemInformation['maxTime'])
print("Number of time slots: ", maxTime)
gamma = int(systemInformation['gamma'])
print("Gamma: ", gamma)
Vs = systemInformation['Vs']
print("Vs: ", Vs)
arrivals = systemInformation['arrivals']
print(arrivals.shape)
# print("Actual arrivals of SFC 2: ", arrivals[2, 0:100])
unitCommCost = systemInformation['unitCommCost']

# print("**********************************************")

if fileIndex not in [1, 3, 5]:
    observed_arrivals = np.load(filenames[fileIndex] + "Arrival_error.npz")["arrivals_error"]
    # print("Observed arrivals of SFC 2: ", observed_arrivals[2, 0:100])
    print("Errors: ", observed_arrivals[2, 0:100] - arrivals[2, 0:100])

# print("Ends.")