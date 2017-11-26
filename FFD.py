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

'''
the observed states at the beginning of time-slot t
'''
# queueBacklogs[c, f, s] saves the queue backlogs of server s, VM f, type c.
queueBacklogs = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

# actualServices[c, f, s] denotes actually how many requests are finished.
actualServices = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

# energyCosts[s] denotes the energy consumption on server s.
energyCosts = np.zeros(numOfServer, dtype=int)

# partitionCosts[c] denotes the partition cost of chain c.
partitionCosts = np.zeros(numOfSC, dtype=int)

'''
records of states over time
'''
varOfQueueBacklogs = np.zeros(maxTime)
cumulativeQueueBacklogs = np.zeros(maxTime)
timeAverageOfQueueBacklogs = np.zeros(maxTime)

cumulativeEnergyCosts = np.zeros(maxTime)
timeAverageOfEnergyCosts = np.zeros(maxTime)

cumulativePartitionCosts = np.zeros(maxTime)
timeAverageOfPartitionCosts = np.zeros(maxTime)


def FFD():
    placement = {}
    resources = np.zeros((numOfSC, numOfNF, numOfServer))
    restCapacities = serverCapacities.copy()
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            serverWithLargestRestCapacity = np.argmax(restCapacities)
            if restCapacities[serverWithLargestRestCapacity] >= processingCost[f]:
                placement[(c, f)] = serverWithLargestRestCapacity
                restCapacities[serverWithLargestRestCapacity] -= processingCost[f]
            else:
                raise Exception("No legal placement decision in First Fit Decreasing Scheme")

    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                if placement[(c, f)] == s:
                    resources[c, f, s] = processingCost[f] * serverCapacities[s] / float(serverCapacities[s] - restCapacities[s])

    return placement, resources


def QueueUpdate(t, queues, servicesPre, placement, resources):
    '''
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
            chosenServer = placement[(c, f)]

            if i == 0:
                updatedQueues[c, f, chosenServer] += arrivals[c, t]
            else:
                fPre = serviceChains[c, i - 1]

                for s in range(numOfServer):
                    updatedQueues[c, f, chosenServer] += servicesPre[c, fPre, s]
                    if s != chosenServer:
                        partitions[c] += servicesPre[c, fPre, s] * pCost

    # Service process
    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                services[c, f, s] = min(int(resources[c, f, s] / float(processingCost[f])), updatedQueues[c, f, s])
                energies[s] += (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s]) * services[c, f, s] * processingCost[f]
                updatedQueues[c, f, s] -= services[c, f, s]

    return updatedQueues, partitions, services, energies


if __name__ == "__main__":

    start_time = time()

    placements, resourceAllocations = FFD()

    for t in range(maxTime):
        print("Now time slot is %s" % t)

        queueBacklogs, partitionCosts, actualServices, energyCosts = \
            QueueUpdate(t, queueBacklogs, actualServices, placements, resourceAllocations)

        cumulativeQueueBacklogs[t] += cumulativeQueueBacklogs[t - 1] + np.sum(queueBacklogs)
        timeAverageOfQueueBacklogs[t] = cumulativeQueueBacklogs[t] / float(t + 1)

        cumulativeEnergyCosts[t] += cumulativeEnergyCosts[t - 1] + np.sum(energyCosts)
        timeAverageOfEnergyCosts[t] = cumulativeEnergyCosts[t] / float(t + 1)

        cumulativePartitionCosts[t] += cumulativePartitionCosts[t - 1] + np.sum(partitionCosts)
        timeAverageOfPartitionCosts[t] = cumulativePartitionCosts[t] / float(t + 1)

        varOfQueueBacklogs[t] = np.var(queueBacklogs)

    end_time = time()

    np.save("resultsFFD/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogs)
    np.save("resultsFFD/timeAverageOfEnergyCosts", timeAverageOfEnergyCosts)
    np.save("resultsFFD/timeAverageOfPartitionCosts", timeAverageOfPartitionCosts)
    np.savez("resultsFFD/varOfQueueBacklogs", varOfQueueBacklogs)
    print("Simulation of First Fit Decreasing ends. Duration is %s sec." % (end_time - start_time,))
