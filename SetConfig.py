import random
import numpy as np

'''
Network Function (NF)
'''
numOfNF = 3  # number of NF types

processingCost = [5, 2, 1]
# processingCost = np.zeros(numOfNF)
# for f in range(numOfNF):
#     processingCost[f] = 1

'''
Service Chain (SC)
'''
numOfSC = 2  # number of Service Chain types
lengthOfSC = 3

#  Here the service chains are generated randomly
serviceChains = {0: [2, 0, 1], 1: [0, 1, 2]}  # here we must use dictionary to verify if the generated chain is new
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

#  Here each service chain may have different arrival rates
arrivalRates = [1, 2]
# arrivalRates = np.zeros(numOfSC)
# for c in range(numOfSC):
#     arrivalRates[c] = 2


'''
Substrate Network (SN)
'''
numOfServer = 2  # number of servers

serverCapacities = np.zeros(numOfServer)
for c in range(numOfServer):
    serverCapacities[c] = 20

idleEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    idleEnergies[c] = 10

maxEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    maxEnergies[c] = 50

'''
System Information
'''
maxTime = 200000
# Vs = [1]
Vs = [i for i in range(101)]
gamma = 1

#  arrivals[c, t] is the number of arrival requests of SC type c at time-slot t
arrivals = np.zeros((numOfSC, maxTime))
procs = {
    'exp': (lambda lam: random.expovariate(lam)),
    'pareto': (lambda alpha: random.paretovariate(0.999*alpha+1)-1),
    'uni': (lambda r: random.uniform(0, 2.0/r)),
    'normal': (lambda r: max(0, random.normalvariate(1.0/r, 0.005))),
    'constant': (lambda r: 1.0/r),
    # 'trace': interval_generator
}


def generate(maxTime, arrRate, arrProc):
    totalArrivals = np.zeros(maxTime)
    currentTime = 0
    arrivalTimePoints = []
    while currentTime < maxTime:
        interval = arrProc(arrRate)
        currentTime += interval
        arrivalTimePoints.append(currentTime)

    t = 1
    for p in arrivalTimePoints:
        if p <= t:
            totalArrivals[t-1] += 1
        else:
            while t < p:
                t += 1
            if t > maxTime:
                break
            totalArrivals[t-1] += 1

    return totalArrivals

for c in range(numOfSC):
        arrivals[c, :] = generate(maxTime, arrivalRates[c], procs['exp'])

np.savez("config/NF Information.npz", numOfNF=numOfNF, processingCost=processingCost)
np.savez("config/SC Information.npz", numOfSC=numOfSC, lengthOfSC=lengthOfSC, serviceChains=serviceChainsNew, arrivalRates=arrivalRates)
np.savez("config/SN Information.npz", numOfServer=numOfServer, serverCapacities=serverCapacities, idleEnergies=idleEnergies, maxEnergies=maxEnergies)
np.savez("config/System Information", maxTime=maxTime, arrivals=arrivals, Vs=Vs, gamma=gamma)