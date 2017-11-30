import numpy as np
from time import time

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
unitCommCost = systemInformation['unitCommCost']
alpha = systemInformation['alpha']

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCosts = nfInformation['processingCosts']  # processingCosts[f]

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']  # serviceChains[c, i]
windowSizes = scInformation['windowSizes']
maxWindowSize = np.max(windowSizes)

# Substrate Network Information (mainly about the servers)
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']  # serverCapacities[s]
idleEnergies = snInformation['idleEnergies']  # idleEnergies[s]
maxEnergies = snInformation['maxEnergies']  # maxEnergies[s]

'''
the observed states at the beginning of time-slot t, we maintain states for each trade-off parameter V
'''
# A[V][c, 0], A[V][c, 1], ...
predictionQueueBacklogs = {V: np.zeros((numOfSC, maxWindowSize+1)) for V in Vs}

# mu[V][c] is the number of requests of SC c that been sent into the system.
predictionServiceCapacities = {V: np.zeros(numOfSC) for V in Vs}

# U. queueBacklogs[V][c, f, s] saves the queue backlogs of server s, VM f, type c.
queueBacklogs = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# X. placements[V][(c, f)] = s means the NF f of service chain c is placed on server s.
placements = {V: {} for V in Vs}

# R. resourceAllocations[V][c, f, s] denotes how many resources is allocated.
resourceAllocations = {V: np.zeros((numOfSC, numOfNF, numOfServer)) for V in Vs}

# H. actualServices[V][c, f, s] denotes actually how many requests are finished.
actualServices = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# E. energyCosts[V][s] denotes the energy consumption on server s.
energyCosts = {V: np.zeros(numOfServer, dtype=int) for V in Vs}

# P. communicationCosts[V][c] denotes the communication cost of chain c.
communicationCosts = {V: np.zeros(numOfSC, dtype=int) for V in Vs}

'''
records of states over time
'''
varOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
cumulativeQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}

cumulativeEnergyCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfEnergyCosts = {V: np.zeros(maxTime) for V in Vs}

cumulativeCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}


def VNFPlacement(t, V, predictionQueues, queues, services):
    #  placement[(c,f)] = s means VNF f of SC type c is placed on server s
    placement = {}
    predictionServices = np.zeros(numOfSC)

    #  For the ingress VNF (f = serviceChains[c, 0])
    for c in range(numOfSC):
        ingressF = serviceChains[c, 0]
        chosenServer = np.argmin(queues[c, ingressF, :])
        placement[(c, ingressF)] = chosenServer
        Asum = np.sum(predictionQueues[c, :])
        if queues[c, ingressF, chosenServer] < alpha * Asum:
            predictionServices[c] = Asum
        else:
            predictionServices[c] = predictionQueues[c, 0]

    #  mValue[c, f] denotes the arrivals of VNF f of SC type c
    mValue = np.zeros((numOfSC, numOfNF))
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            #  If f is the ingress of SC type c,
            if i == 0:
                mValue[c, f] += predictionServices[c]
            #  Otherwise,
            else:
                fPre = serviceChains[c, i-1]
                for s in range(numOfServer):
                    mValue[c, f] += services[c, fPre, s]

    #  For the non-ingress VNF (f != serviceChains[c, 0]):
    for c in range(numOfSC):
        for i in range(1, lengthOfSC):
            f = serviceChains[c, i]
            pair = (c, f)
            weights = np.zeros(numOfServer)
            for s in range(numOfServer):
                fPre = serviceChains[c, i - 1]
                weights[s] = queues[c, f, s] * mValue[c, f] - gamma * V * services[c, fPre, s]

            chosenServer = weights.argmin()
            placement[pair] = chosenServer

    return placement, predictionServices, mValue


def ResourceAllocation(t, V, queues, placement, mValue):
    #  allocation[c,f,s] denotes the number of resources allocated to queue (c,f,s)
    allocation = np.zeros((numOfSC, numOfNF, numOfServer))

    for s in range(numOfServer):
        term1 = V * (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s])
        weights = term1 * np.ones((numOfSC, numOfNF))
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                weights[c, f] -= queues[c, f, s] / float(processingCosts[f])

        restCapacity = serverCapacities[s]
        while True:
            (chosenType, chosenVM) = np.unravel_index(weights.argmin(), weights.shape)
            if weights[chosenType, chosenVM] >= 0:
                break

            neededResource = queues[chosenType, chosenVM, s] * processingCosts[chosenVM]
            if (chosenType, chosenVM) in placement.keys() and placement[(chosenType, chosenVM)] == s:
                neededResource += mValue[chosenType, chosenVM] * processingCosts[chosenVM]

            if restCapacity >= processingCosts[chosenVM]:
                allocation[chosenType, chosenVM, s] = min(restCapacity, neededResource)
                restCapacity -= allocation[chosenType, chosenVM, s]
            weights[chosenType, chosenVM] = float('inf')

            if restCapacity == 0:
                break

    return allocation


def QueueUpdate(t, V, queues, servicesPre, predictionServices, placement, resources):

    #  updatedQueues[c, f, s] is queue length after updating
    updatedQueues = queues.copy()
    #  communications[c] is the communication cost of SC c
    communications = np.zeros(numOfSC)
    #  services[c, f, s] denotes the number of finished requests of queue (c,f,s) of current time-slot
    services = np.zeros((numOfSC, numOfNF, numOfServer))
    #  energies[s] denotes the energy consumption of server s
    energies = np.array(idleEnergies)

    # Appending arrivals
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            chosenServer = placement[(c, f)]

            if i == 0:
                updatedQueues[c, f, chosenServer] += predictionServices[c]
            else:
                fPre = serviceChains[c, i - 1]
                for s in range(numOfServer):
                    updatedQueues[c, f, chosenServer] += servicesPre[c, fPre, s]
                    if s != chosenServer:
                        communications[c] += servicesPre[c, fPre, s] * unitCommCost

    # Service process
    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                services[c, f, s] = resources[c, f, s] / float(processingCosts[f])
                energies[s] += (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s]) * resources[c, f, s]
                updatedQueues[c, f, s] -= services[c, f, s]

    return updatedQueues, communications, services, energies


def PredictionQueueUpdate(t, V, predictionQueues, predictionServices):
    updatedPredictionQueues = np.zeros((numOfSC, maxWindowSize + 1))
    for c in range(numOfSC):
        if t + windowSizes[c] + 1 < maxTime:
            updatedPredictionQueues[c, windowSizes[c]] = arrivals[c, t + windowSizes[c] + 1]
        else:
            updatedPredictionQueues[c, windowSizes[c]] = 0

        restCapacity = predictionServices[c]
        restCapacity -= predictionQueues[c, 0]
        for d in range(windowSizes[c]):
            if restCapacity > 0:
                service = min(predictionQueues[c, d+1], restCapacity)
                updatedPredictionQueues[c, d] = predictionQueues[c, d+1] - service
                restCapacity -= service
            else:
                break

    return updatedPredictionQueues


def VNFGreedy(t, V):
    '''
    :param t: current time-slot t
    :param V: current trade-off parameter V
    '''
    global predictionQueueBacklogs, predictionServiceCapacities, queueBacklogs, placements, resourceAllocations, \
        actualServices, energyCosts, communicationCosts

    placements[V], predictionServiceCapacities[V], mValue = VNFPlacement(t, V, predictionQueueBacklogs[V], queueBacklogs[V], actualServices[V])

    resourceAllocations[V] = ResourceAllocation(t, V, queueBacklogs[V], placements[V], mValue)

    queueBacklogs[V], communicationCosts[V], actualServices[V], energyCosts[V] = \
        QueueUpdate(t, V, queueBacklogs[V], actualServices[V], predictionServiceCapacities[V], placements[V], resourceAllocations[V])

    predictionQueueBacklogs[V] = PredictionQueueUpdate(t, V, predictionQueueBacklogs[V], predictionServiceCapacities[V])


if __name__ == "__main__":

    start_time = time()

    for V in Vs:
        for c in range(numOfSC):
            for d in range(windowSizes[c] + 1):
                predictionQueueBacklogs[V][c, d] = arrivals[c, d]

    for V in Vs:
        # print("Now V is %s" % (V, ))
        for t in range(maxTime):
            print("Now V is %s and time slot is %s" % (V, t))
            VNFGreedy(t, V)

            cumulativeQueueBacklogs[V][t] += cumulativeQueueBacklogs[V][t - 1] + np.sum(queueBacklogs[V])
            timeAverageOfQueueBacklogs[V][t] = cumulativeQueueBacklogs[V][t] / float(t + 1)

            cumulativeEnergyCosts[V][t] += cumulativeEnergyCosts[V][t - 1] + np.sum(energyCosts[V])
            timeAverageOfEnergyCosts[V][t] = cumulativeEnergyCosts[V][t] / float(t + 1)

            cumulativeCommunicationCosts[V][t] += cumulativeCommunicationCosts[V][t - 1] + np.sum(communicationCosts[V])
            timeAverageOfCommunicationCosts[V][t] = cumulativeCommunicationCosts[V][t] / float(t + 1)

            varOfQueueBacklogs[V][t] = np.var(queueBacklogs[V])

    end_time = time()

    # Be careful of the Vs mapping to i. [1, 10, 20, 50, 100] --> [0, 1, 2, 3, 4]
    timeAverageOfQueueBacklogsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfEnergyCostsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfCommunicationCostsNew = np.zeros((lenOfVs, maxTime))

    for i in range(lenOfVs):
        timeAverageOfQueueBacklogsNew[i, :] = np.array(timeAverageOfQueueBacklogs[Vs[i]])
        timeAverageOfEnergyCostsNew[i, :] = np.array(timeAverageOfEnergyCosts[Vs[i]])
        timeAverageOfCommunicationCostsNew[i, :] = np.array(timeAverageOfCommunicationCosts[Vs[i]])

    np.save("resultsP-ONRA/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogsNew)
    np.save("resultsP-ONRA/timeAverageOfEnergyCosts.npy", timeAverageOfEnergyCostsNew)
    np.save("resultsP-ONRA/timeAverageOfCommunicationCosts.npy", timeAverageOfCommunicationCostsNew)
    np.save("resultsP-ONRA/varOfQueueBacklogs.npy", varOfQueueBacklogs)

    np.save("resultsP-ONRA/runtime.npy", end_time - start_time)
    print("Simulation ends. Duration is %s sec." % (end_time - start_time,))
