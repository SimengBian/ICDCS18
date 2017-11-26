import random
import numpy as np
td = [0, 0.08514851485148515, 0.15247524752475247, 0.19603960396039605, 0.2297029702970297, 0.2514851485148515, 0.26732673267326734, 0.28316831683168314, 0.2891089108910891, 0.297029702970297, 0.299009900990099, 0.30297029702970296, 0.3069306930693069, 0.41386138613861384, 0.49504950495049505, 0.592079207920792, 0.7346534653465346, 0.8138613861386138, 0.9603960396039604, 1.0]
xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 250, 400, 450, 650, 1000, 1600, 8000, 25000]  # us

'''
Network Function (NF)
'''
numOfNF = 30  # number of NF types

processingCost = np.array(15 * [1] + 10 * [2] + 5 * [5])
# processingCost = np.zeros(numOfNF)
# for f in range(numOfNF):
#     processingCost[f] = 1

'''
Service Chain (SC)
'''
numOfSC = 10  # number of Service Chain types
lengthOfSC = 3

#  Here the service chains are generated randomly
# serviceChains = {0: [2, 0, 1], 1: [0, 1, 2]}  # here we must use dictionary to verify if the generated chain is new
serviceChains = {c: [] for c in range(numOfSC)}
c = 0
while True:
    if c >= numOfSC:
        break

    NFs = list(range(numOfNF))  # the networks function {0,1,2,...,F-1}
    random.shuffle(NFs)
    chain = NFs[0:lengthOfSC]  # the chosen service chain
    if chain not in serviceChains.values():  # if it is new
        serviceChains[c] = chain  # added it to the dictionary
        c += 1

#  serviceChainsNew[c, i] is the i-th NF of SC type c
serviceChainsNew = (-1) * np.ones((numOfSC, lengthOfSC), dtype=int)
for c in range(numOfSC):
    for i in range(lengthOfSC):
        serviceChainsNew[c, i] = serviceChains[c][i]

arrivalRate = 5.88

'''
Substrate Network (SN)
'''
numOfServer = 10  # number of servers

serverCapacities = np.zeros(numOfServer)
for c in range(numOfServer):
    serverCapacities[c] = 16

idleEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    idleEnergies[c] = 80.5

maxEnergies = np.zeros(numOfServer)
for c in range(numOfServer):
    maxEnergies[c] = 2735

'''
System Information
'''
maxTime = 20
Vs = [1]
# Vs = [i*10 for i in range(51)]
gamma = 1
pCost = 1


def interval_generator(r):
    rand = random.random()
    for i in range(1, len(td)):
        if rand <= td[i]:
            return 0.5*(xs[i] + xs[i-1])/1e4  # 10ms
        else:
            continue
    raise Exception("It shouldn't reach here..")


procs = {
    'exp': (lambda lam: random.expovariate(lam)),
    'pareto': (lambda alpha: random.paretovariate(0.999*alpha+1)-1),
    'uni': (lambda r: random.uniform(0, 2.0/r)),
    'normal': (lambda r: max(0, random.normalvariate(1.0/r, 0.005))),
    'constant': (lambda r: 1.0/r),
    'trace': interval_generator
}


def generate(maxTime, arrRate, arrProc):
    totalArrivals = np.zeros((numOfSC, maxTime))
    currentTime = 0
    arrivalTimePoints = []
    while currentTime < maxTime:
        interval = arrProc(arrRate)
        currentTime += interval
        SCtype = random.choice(range(numOfSC))
        arrivalTimePoints.append((SCtype, currentTime))

    t = 1
    for item in arrivalTimePoints:
        SCtype = item[0]
        timePoint = item[1]
        if timePoint <= t:
            totalArrivals[SCtype, t-1] += 1
        else:
            while t < timePoint:
                t += 1
            if t > maxTime:
                break
            totalArrivals[SCtype, t-1] += 1

    return totalArrivals


#  arrivals[c, t] is the number of arrival requests of SC type c at time-slot t
arrivals = generate(maxTime, arrivalRate, procs['exp'])

np.savez("config/NF Information.npz", numOfNF=numOfNF, processingCost=processingCost)
np.savez("config/SC Information.npz", numOfSC=numOfSC, lengthOfSC=lengthOfSC, serviceChains=serviceChainsNew, arrivalRates=arrivalRates)
np.savez("config/SN Information.npz", numOfServer=numOfServer, serverCapacities=serverCapacities, idleEnergies=idleEnergies, maxEnergies=maxEnergies)
np.savez("config/System Information", pCost=pCost, maxTime=maxTime, arrivals=arrivals, Vs=Vs, gamma=gamma)
