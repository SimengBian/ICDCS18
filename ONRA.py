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

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCosts = nfInformation['processingCosts']  # processingCosts[f]

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']  # serviceChains[c, i]

# Substrate Network Information (mainly about the servers)
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']  # serverCapacities[s]
idleEnergies = snInformation['idleEnergies']  # idleEnergies[s]
maxEnergies = snInformation['maxEnergies']  # maxEnergies[s]

'''
the observed states at the beginning of time-slot t, we maintain states for each trade-off parameter V
'''
# queueBacklogs[V][c, f, s] saves the queue backlogs of server s, VM f, type c.
queueBacklogs = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# resourceAllocations[V][c, f, s] denotes how many resources is allocated.
resourceAllocations = {V: np.zeros((numOfSC, numOfNF, numOfServer)) for V in Vs}

# actualServices[V][c, f, s] denotes actually how many requests are finished.
actualServices = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# placements[V][(c, f)] = s means the NF f of service chain c is placed on server s.
placements = {V: {} for V in Vs}

# energyCosts[V][s] denotes the energy consumption on server s.
energyCosts = {V: np.zeros(numOfServer, dtype=int) for V in Vs}

# partitionCosts[V][c] denotes the partition cost of chain c.
partitionCosts = {V: np.zeros(numOfSC, dtype=int) for V in Vs}

'''
records of states over time
'''
varOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
cumulativeQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}

cumulativeEnergyCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfEnergyCosts = {V: np.zeros(maxTime) for V in Vs}

cumulativePartitionCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfPartitionCosts = {V: np.zeros(maxTime) for V in Vs}


def VNFPlacement(t, V, queues, resources, services):
    '''
    :param t: current time-slot t
    :param V: current trade-off parameter V
    :param queues: current queue backlogs
    :param resources: resource allocation of previous time-slot
    :param services: actual number of services of previous time-slot
    :returns placement: VNF placement decision
    :returns partitions: resulting partition costs
    '''
    #  placement[(c,f)] = s means VNF f of SC type c is placed on server s
    placement = {}

    #  mValue[c, f] denotes the arrivals of VNF f of SC type c
    mValue = np.zeros((numOfSC, numOfNF))
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            #  If f is the ingress of SC type c,
            if i == 0:
                mValue[c, f] += arrivals[c, t]
            #  Otherwise,
            else:
                fPre = serviceChains[c, i-1]
                for s in range(numOfServer):
                    mValue[c, f] += services[c, fPre, s]

    #  For each SC and VNF pair (c,f), decide on which server to place it
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            pair = (c, f)
            weights = np.zeros(numOfServer)
            for s in range(numOfServer):
                if i == 0:
                    weights[s] = queues[c, f, s] * mValue[c, f]
                else:
                    fPre = serviceChains[c, i - 1]
                    weights[s] = queues[c, f, s] * mValue[c, f] - gamma * V * services[c, fPre, s]

            chosenServer = weights.argmin()
            placement[pair] = chosenServer

    return placement, mValue


def ResourceAllocation(t, V, queues, placement, mValue):
    '''
    :param t: current time-slot t
    :param V: current trade-off parameter V
    :param queues: current queue backlogs
    :returns allocation: resource allocation decision
    :returns services: resulting actual number of services
    :returns energies: energy consumptions
    '''
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
                numService = np.floor(restCapacity/processingCosts[chosenVM])
                allocation[chosenType, chosenVM, s] = min(numService * processingCosts[chosenVM], neededResource)
                restCapacity -= allocation[chosenType, chosenVM, s]
            weights[chosenType, chosenVM] = float('inf')

            if restCapacity == 0:
                break

    return allocation


def QueueUpdate(t, V, queues, servicesPre, placement, resources):
    '''
    :param t: current time-slot t
    :param V: current trade-off parameter V
    :param queues: current queue backlogs
    :param servicesPre: actual number of services of previous time-slot
    :param servicesCur: actual number of services of current time-slot
    :param placement: actual VNF placement
    :param resources: resources allocated at current time slot
    :return queues: updated queues
    :return partitions:
    :return services:
    :return energies:
    '''
    #  updatedQueues[c, f, s] is queue length after updating
    updatedQueues = queues.copy()
    #  partitions[c] is the partition cost of SC c
    partitions = np.zeros(numOfSC)
    #  services[c, f, s] denotes the number of finished requests of queue (c,f,s) of current time-slot
    services = np.zeros((numOfSC, numOfNF, numOfServer))
    #  energies[s] denotes the energy consumption of server s
    energies = np.array(idleEnergies)

    # Appending arrivals
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            chosenServer = placement[tuple([c, f])]

            if i == 0:
                updatedQueues[c, f, chosenServer] += arrivals[c, t]
            else:
                fPre = serviceChains[c, i - 1]

                for s in range(numOfServer):
                    updatedQueues[c, f, chosenServer] += servicesPre[c, fPre, s]
                    if s != chosenServer:
                        partitions[c] += servicesPre[c, fPre, s] * unitCommCost

    # Service process
    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                services[c, f, s] = np.floor(resources[c, f, s] / float(processingCosts[f]))
                energies[s] += (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s]) * resources[c, f, s]
                updatedQueues[c, f, s] -= services[c, f, s]

    return updatedQueues, partitions, services, energies


def VNFGreedy(t, V):
    '''
    :param t: current time-slot t
    :param V: current trade-off parameter V
    '''
    global queueBacklogs, resourceAllocations, placements, energyCosts, partitionCosts, actualServices

    placements[V], mValue = VNFPlacement(t, V, queueBacklogs[V], resourceAllocations[V], actualServices[V])

    resourceAllocations[V] = ResourceAllocation(t, V, queueBacklogs[V], placements[V], mValue)

    queueBacklogs[V], partitionCosts[V], actualServices[V], energyCosts[V] = \
        QueueUpdate(t, V, queueBacklogs[V], actualServices[V], placements[V], resourceAllocations[V])


if __name__ == "__main__":

    start_time = time()

    for V in Vs:
        # print("Now V is %s" % (V, ))
        for t in range(maxTime):
            print("Now V is %s and time slot is %s" % (V, t))
            VNFGreedy(t, V)

            cumulativeQueueBacklogs[V][t] += cumulativeQueueBacklogs[V][t - 1] + np.sum(queueBacklogs[V])
            timeAverageOfQueueBacklogs[V][t] = cumulativeQueueBacklogs[V][t] / float(t + 1)

            cumulativeEnergyCosts[V][t] += cumulativeEnergyCosts[V][t - 1] + np.sum(energyCosts[V])
            timeAverageOfEnergyCosts[V][t] = cumulativeEnergyCosts[V][t] / float(t + 1)

            cumulativePartitionCosts[V][t] += cumulativePartitionCosts[V][t - 1] + np.sum(partitionCosts[V])
            timeAverageOfPartitionCosts[V][t] = cumulativePartitionCosts[V][t] / float(t + 1)

            varOfQueueBacklogs[V][t] = np.var(queueBacklogs[V])

    end_time = time()

    # Be careful of the Vs mapping to i. [1, 10, 20, 50, 100] --> [0, 1, 2, 3, 4]
    timeAverageOfQueueBacklogsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfEnergyCostsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfPartitionCostsNew = np.zeros((lenOfVs, maxTime))

    for i in range(lenOfVs):
        timeAverageOfQueueBacklogsNew[i, :] = np.array(timeAverageOfQueueBacklogs[Vs[i]])
        timeAverageOfEnergyCostsNew[i, :] = np.array(timeAverageOfEnergyCosts[Vs[i]])
        timeAverageOfPartitionCostsNew[i, :] = np.array(timeAverageOfPartitionCosts[Vs[i]])

    np.save("resultsONRA/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogsNew)
    np.save("resultsONRA/timeAverageOfEnergyCosts.npy", timeAverageOfEnergyCostsNew)
    np.save("resultsONRA/timeAverageOfCommunicationCosts.npy", timeAverageOfPartitionCostsNew)
    np.save("resultsONRA/varOfQueueBacklogs.npy", varOfQueueBacklogs)

    np.save("resultsONRA/runtime.npy", end_time - start_time)
    print("Simulation ends. Duration is %s sec." % (end_time - start_time,))
