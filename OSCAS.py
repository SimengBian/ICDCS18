import numpy as np
from time import time
from request import *

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
the observed states at the beginning of time-slot t, we maintain states for each trade-off parameter V
'''
finished_requests = {V: [] for V in Vs}

queueBacklogs = {V: np.array(
                [[[RequestQueue()
                    for k in range(numOfServer)]
                    for j in range(numOfNF)]
                    for i in range(numOfSC)]
                ) for V in Vs}

resourceAllocations = {V: np.zeros((numOfSC, numOfNF, numOfServer)) for V in Vs}

actualServices = {V: [[[[]
                        for k in range(numOfServer)]
                       for j in range(numOfNF)]
                      for i in range(numOfSC)] for V in Vs}

# placements[V][(c, f)] = s means the NF f of service chain c is placed on server s.
placements = {V: {} for V in Vs}

energyCosts = {V: np.zeros(numOfServer, dtype=int) for V in Vs}

communicationCosts = {V: np.zeros(numOfSC, dtype=int) for V in Vs}

'''
records of states over time
'''
cumulativeQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}

cumulativeEnergyCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfEnergyCosts = {V: np.zeros(maxTime) for V in Vs}

cumulativeCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}


def VNFPlacement(t, V, queues, services):
    '''
    :param t: current time slot t.
    :param V: trade-off parameter V.
    :param queues: current queue backlogs of V.
    :param services: services record of previous time slot.
    :return placement: placement decision.
    :return mValue: arrivals record.
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
                    mValue[c, f] += len(services[c][fPre][s])

    #  For each SC and VNF pair (c,f), decide on which server to place it
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            pair = (c, f)
            weights = np.zeros(numOfServer)
            for s in range(numOfServer):
                if i == 0:
                    weights[s] = queues[c, f, s].length() * mValue[c, f]
                else:
                    fPre = serviceChains[c, i - 1]
                    weights[s] = queues[c, f, s].length() * mValue[c, f] - gamma * V * len(services[c][fPre][s])

            chosenServer = weights.argmin()
            placement[pair] = chosenServer

    return placement, mValue

def ResourceAllocation(t, V, queues, placement, mValue):
    '''
    :param t: current time slot t
    :param V: trade-off parameter V
    :param queues: current queue backlogs of V
    :returns allocation: resource allocation decision
    '''
    #  allocation[c,f,s] denotes the number of resources allocated to queue (c,f,s)
    allocation = np.zeros((numOfSC, numOfNF, numOfServer))

    for s in range(numOfServer):
        term1 = V * (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s])
        weights = term1 * np.ones((numOfSC, numOfNF))
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                weights[c, f] -= queues[c, f, s].length() / float(processingCosts[f])

        restCapacity = serverCapacities[s]
        while True:
            (chosenType, chosenVM) = np.unravel_index(weights.argmin(), weights.shape)
            if weights[chosenType, chosenVM] >= 0:
                break

            neededResource = queues[chosenType, chosenVM, s].length() * processingCosts[chosenVM]
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
    :param t: current time slot t
    :param V: trade-off parameter V
    :param queues: current queue backlogs of V
    :param servicesPre: services record of previous time slot.
    :param placement: current placement decision
    :param resources: current resource allocation
    :return updatedQueues: queue backlogs after arriving and serving
    :return communications: communication cost of current time slot
    :return services: services record of current time slot
    :return energies: energy cost of current time slot
    '''

    global finished_requests

    updatedQueues = queues.copy()
    communications = np.zeros(numOfSC)
    services = [[[[]
                   for k in range(numOfServer)]
                  for j in range(numOfNF)]
                 for i in range(numOfSC)]
    energies = np.array(idleEnergies)

    # Appending arrivals
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            chosenServer = placement[(c, f)]

            if i == 0:
                new_reqs = [Request(t) for r in range(arrivals[c, t])]
                updatedQueues[c, f, chosenServer].append_reqs(new_reqs)
            else:
                fPre = serviceChains[c, i - 1]

                for s in range(numOfServer):
                    new_reqs = servicesPre[c][fPre][s]
                    updatedQueues[c, f, chosenServer].append_reqs(new_reqs)
                    if s != chosenServer:
                        communications[c] += len(new_reqs) * unitCommCost

    # Service process
    for s in range(numOfServer):
        for c in range(numOfSC):
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                srv_num = int(np.floor(resources[c, f, s] / float(processingCosts[f])))
                energies[s] += (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s]) * resources[c, f, s]
                services[c][f][s] = updatedQueues[c, f, s].serve(srv_num)

                # judge whether f is the last VNF in SC c
                if i == lengthOfSC - 1:
                    for req in services[c][f][s]:
                        req.set_finished_time(t)
                        finished_requests[V].append(req)

    return updatedQueues, communications, services, energies


def VNFGreedy(t, V):
    '''
    :param t: current time slot t
    :param V: trade-off parameter V
    '''
    global queueBacklogs, resourceAllocations, placements, energyCosts, communicationCosts, actualServices

    placements[V], mValue = VNFPlacement(t, V, queueBacklogs[V], actualServices[V])

    resourceAllocations[V] = ResourceAllocation(t, V, queueBacklogs[V], placements[V], mValue)

    # Note that queueBacklogs[V] is now a tensor of RequestQueue objects
    queueBacklogs[V], communicationCosts[V], actualServices[V], energyCosts[V] = \
        QueueUpdate(t, V, queueBacklogs[V], actualServices[V], placements[V], resourceAllocations[V])


if __name__ == "__main__":

    start_time = time()

    for V in Vs:
        # print("Now V is %s" % (V, ))
        for t in range(maxTime):
            if t % int(1e4) == 0:
                print("Now V is %s and time slot is %s" % (V, t))

            VNFGreedy(t, V)

            queueSizes = np.vectorize(lambda e: e.length())(queueBacklogs[V])

            cumulativeQueueBacklogs[V][t] += cumulativeQueueBacklogs[V][t - 1] + np.sum(queueSizes)
            timeAverageOfQueueBacklogs[V][t] = cumulativeQueueBacklogs[V][t] / float(t + 1)

            cumulativeEnergyCosts[V][t] += cumulativeEnergyCosts[V][t - 1] + np.sum(energyCosts[V])
            timeAverageOfEnergyCosts[V][t] = cumulativeEnergyCosts[V][t] / float(t + 1)

            cumulativeCommunicationCosts[V][t] += cumulativeCommunicationCosts[V][t - 1] + np.sum(communicationCosts[V])
            timeAverageOfCommunicationCosts[V][t] = cumulativeCommunicationCosts[V][t] / float(t + 1)

    end_time = time()

    # Be careful of the Vs mapping to i. [1, 10, 20, 50, 100] --> [0, 1, 2, 3, 4]
    timeAverageOfQueueBacklogsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfEnergyCostsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfCommunicationCostsNew = np.zeros((lenOfVs, maxTime))

    for i in range(lenOfVs):
        timeAverageOfQueueBacklogsNew[i, :] = np.array(timeAverageOfQueueBacklogs[Vs[i]])
        timeAverageOfEnergyCostsNew[i, :] = np.array(timeAverageOfEnergyCosts[Vs[i]])
        timeAverageOfCommunicationCostsNew[i, :] = np.array(timeAverageOfCommunicationCosts[Vs[i]])

    np.save("resultsOSCAS/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogsNew)
    np.save("resultsOSCAS/timeAverageOfEnergyCosts.npy", timeAverageOfEnergyCostsNew)
    np.save("resultsOSCAS/timeAverageOfCommunicationCosts.npy", timeAverageOfCommunicationCostsNew)
    np.save("resultsOSCAS/runtime.npy", end_time - start_time)
    print("Simulation ends. Duration is %s sec." % (end_time - start_time,))
