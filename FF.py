import numpy as np
from time import time

filename = 'config3/'

'''
configurations
'''
scInformation = np.load(filename + "SC Information.npz")
nfInformation = np.load(filename + "NF Information.npz")
snInformation = np.load(filename + "SN Information.npz")
systemInformation = np.load(filename + "System Information.npz")

# System Information
maxTime = int(systemInformation['maxTime'])
# maxTime = int(1e2)
gamma = int(systemInformation['gamma'])
# gamma = 10
Vs = systemInformation['Vs']
# Vs = [1, 100]
lenOfVs = len(Vs)
arrivals = systemInformation['arrivals']
unitCommCost = systemInformation['unitCommCost']

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCosts = nfInformation['processingCosts']

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']

# Substrate Network Information (mainly about the servers)
numOfServer = int(snInformation['numOfServer'])
serverCapacities = snInformation['serverCapacities']
idleEnergies = snInformation['idleEnergies']
maxEnergies = snInformation['maxEnergies']

'''
the observed states at the beginning of time-slot t
'''
queueBacklogs = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

actualServices = np.zeros((numOfSC, numOfNF, numOfServer), dtype=int)

energyCosts = np.zeros(numOfServer, dtype=int)

communicationCosts = np.zeros(numOfSC, dtype=int)

'''
records of states over time
'''
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
    :param t: current time slot t
    :param queues: current queue backlogs
    :param servicesPre: services record of previous time slot
    :param placement: placement decision
    :param resources: resources allocation
    :return updatedQueues: queue backlogs after arriving and serving
    :return communications: communication cost of current time slot
    :return services: services record of current time slot
    :return energies: energy cost of current time slot
    '''
    updatedQueues = queues.copy()
    communications = np.zeros(numOfSC)
    services = np.zeros((numOfSC, numOfNF, numOfServer))
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
        if t % int(1e4) == 0:
            print("Now time slot is %s" % t)

        queueBacklogs, communicationCosts, actualServices, energyCosts = \
            QueueUpdate(t, queueBacklogs, actualServices, placements, resourceAllocations)

        cumulativeQueueBacklogs[t] += cumulativeQueueBacklogs[t - 1] + np.sum(queueBacklogs)
        timeAverageOfQueueBacklogs[t] = cumulativeQueueBacklogs[t] / float(t + 1)

        cumulativeEnergyCosts[t] += cumulativeEnergyCosts[t - 1] + np.sum(energyCosts)
        timeAverageOfEnergyCosts[t] = cumulativeEnergyCosts[t] / float(t + 1)

        cumulativeCommunicationCosts[t] += cumulativeCommunicationCosts[t - 1] + np.sum(communicationCosts)
        timeAverageOfCommunicationCosts[t] = cumulativeCommunicationCosts[t] / float(t + 1)

    end_time = time()

    np.save("resultsFF/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogs)
    np.save("resultsFF/timeAverageOfEnergyCosts.npy", timeAverageOfEnergyCosts)
    np.save("resultsFF/timeAverageOfCommunicationCosts.npy", timeAverageOfCommunicationCosts)
    np.save("resultsFF/varOfQueueBacklogs.npy", varOfQueueBacklogs)
    print("Simulation of First Fit ends. Duration is %s sec." % (end_time - start_time,))
