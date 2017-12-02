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
the observed states at the beginning of time-slot t
'''
# queueBacklogs[c, f, s] saves the queue backlogs of server s, VM f, type c.
queueBacklogs = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

# actualServices[c, f, s] denotes actually how many requests are finished.
actualServices = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

# energyCosts[s] denotes the energy consumption on server s.
energyCosts = np.zeros(numOfServer, dtype=int)

# communicationCosts[c] denotes the communication cost of chain c.
communicationCosts = np.zeros(numOfSC, dtype=int)

'''
records of states over time
'''
varOfQueueBacklogs = np.zeros(maxTime)
cumulativeQueueBacklogs = np.zeros(maxTime)
timeAverageOfQueueBacklogs = np.zeros(maxTime)

cumulativeEnergyCosts = np.zeros(maxTime)
timeAverageOfEnergyCosts = np.zeros(maxTime)

cumulativeCommunicationCosts = np.zeros(maxTime)
timeAverageOfCommunicationCosts = np.zeros(maxTime)


def FirstFit():
    placement = {}
    resources = np.zeros((numOfSC, numOfNF, numOfServer))
    restCapacities = serverCapacities.copy()
    #  Decide placement
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            flag = False
            for s in range(numOfServer):
                if restCapacities[s] >= processingCosts[f]:
                    placement[(c, f)] = s
                    restCapacities[s] -= processingCosts[f]
                    flag = True
                    break

            if flag is False:
                raise Exception("No legal placement decision in First Fit Scheme")
    #  Decide resource allocation
    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                if placement[(c, f)] == s:
                    resources[c, f, s] = serverCapacities[s] * processingCosts[f] / \
                                         float(serverCapacities[s] - restCapacities[s])

    return placement, resources


def QueueUpdate(t, queues, servicesPre, placement, resources):
    '''
    :param queues: current queue backlogs
    :param servicesPre: actual number of services of previous time-slot
    :param servicesCur: actual number of services of current time-slot
    :param placement: actual VNF placement
    :param resources: resources allocated at current time slot
    :return queues: updated queues
    :return communications:
    :return services:
    :return energies:
    '''
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
                updatedQueues[c, f, chosenServer] += arrivals[c, t]
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
                services[c, f, s] = min(np.floor(resources[c, f, s] / float(processingCosts[f])), updatedQueues[c, f, s])
                energies[s] += (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s]) \
                                    * services[c, f, s] * processingCosts[f]
                updatedQueues[c, f, s] -= services[c, f, s]

    return updatedQueues, communications, services, energies


if __name__ == "__main__":

    start_time = time()

    placements, resourceAllocations = FirstFit()

    for t in range(maxTime):
        print("Now time slot is %s" % t)

        queueBacklogs, communicationCosts, actualServices, energyCosts = \
            QueueUpdate(t, queueBacklogs, actualServices, placements, resourceAllocations)

        cumulativeQueueBacklogs[t] += cumulativeQueueBacklogs[t - 1] + np.sum(queueBacklogs)
        timeAverageOfQueueBacklogs[t] = cumulativeQueueBacklogs[t] / float(t + 1)

        cumulativeEnergyCosts[t] += cumulativeEnergyCosts[t - 1] + np.sum(energyCosts)
        timeAverageOfEnergyCosts[t] = cumulativeEnergyCosts[t] / float(t + 1)

        cumulativeCommunicationCosts[t] += cumulativeCommunicationCosts[t - 1] + np.sum(communicationCosts)
        timeAverageOfCommunicationCosts[t] = cumulativeCommunicationCosts[t] / float(t + 1)

        varOfQueueBacklogs[t] = np.var(queueBacklogs)

    end_time = time()

    np.save("resultsFF/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogs)
    np.save("resultsFF/timeAverageOfEnergyCosts.npy", timeAverageOfEnergyCosts)
    np.save("resultsFF/timeAverageOfCommunicationCosts.npy", timeAverageOfCommunicationCosts)
    np.save("resultsFF/varOfQueueBacklogs.npy", varOfQueueBacklogs)
    print("Simulation of First Fit ends. Duration is %s sec." % (end_time - start_time,))
