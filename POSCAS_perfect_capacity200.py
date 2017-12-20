import numpy as np
from time import time
from request import *

filename = 'config3/'
"""
configurations
"""
scInformation = np.load(filename + "SC Information.npz")
nfInformation = np.load(filename + "NF Information.npz")
snInformation = np.load(filename + "SN Information.npz")
systemInformation = np.load(filename + "System Information.npz")

# System Information
maxTime = systemInformation['maxTime']
# maxTime = int(1e2)
gamma = systemInformation['gamma']
print('gamma: ', gamma)
# gamma = 1
# Vs = systemInformation['Vs']
Vs = [40]
lenOfVs = len(Vs)
arrivals = systemInformation['arrivals']  # arrivals[c, t]
unitCommCost = systemInformation['unitCommCost']
alpha = systemInformation['alpha']

windowSizes = [i*2 for i in range(0, 6)] + [j * 5 for j in range(3, 21)] + [1000, 5000]
print(windowSizes)

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCosts = nfInformation['processingCosts']  # processingCosts[f]

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']  # serviceChains[c, i]
print(serviceChains)
policy = Policy.FIFO

# Substrate Network Information (mainly about the servers)
# numOfServer = int(snInformation['numOfServer'])
numOfServer = 5
# serverCapacities = snInformation['serverCapacities']  # serverCapacities[s]
serverCapacities = np.array([200 for s in range(numOfServer)])

idleEnergies = snInformation['idleEnergies']  # idleEnergies[s]
maxEnergies = snInformation['maxEnergies']  # maxEnergies[s]


# Different from ONRA, in this stage,
# P-ONRA also needs to decide the placement
# as well as the number of admitted requests.
#
# Note that in this stage, prediction queues
# are also updated.
def VNFPlacement(t, V, predictionQueues, queues, services):
    # Placement[(c,f)] = s indicates VNF f
    # of SC type c is placed on server s
    placement = {}
    predictionServices = [[] for c in range(numOfSC)]

    restResourceCapacities = []

    for s in range(numOfServer):
        usedResources = sum([
            sum([queues[sc, f, s].length()
                 for sc in range(numOfSC)]) * processingCosts[f]
            for f in range(numOfNF)
        ])
        restResourceCapacities.append(serverCapacities[s] - usedResources)

    #  For the ingress VNF (f = serviceChains[c, 0])
    for c in range(numOfSC):
        ingressF = serviceChains[c, 0]
        chosenServer = int(np.argmin(queues[c, ingressF, :]))
        placement[(c, ingressF)] = chosenServer

        # Pre-admission is enabled only when the server has spare resources
        if restResourceCapacities[chosenServer] > predictionQueues[c][0].length() * processingCosts[ingressF]:
            # Calculate the queue size of each prediction queue of a SC type
            predictionQueueSizes = list(map(lambda e: e.length(), predictionQueues[c]))
            num_req_to_srv = int(min(np.floor(restResourceCapacities[chosenServer] / float(processingCosts[ingressF])),
                                     np.sum(predictionQueueSizes)))

            # All requests arriving in current time-slot must be forwarded
            rest_num_req_to_srv = num_req_to_srv
            rest_num_req_to_srv -= predictionQueues[c][0].length()
            # Append all requests  arriving in current time slot
            # to the admission buffer.
            predictionServices[c] += predictionQueues[c][0].withdraw_all_reqs()

            # Pre-admit requests in future windows according to
            # some scheduling principle, e.g., FIFO, LIFO
            for w in range(1, windowSizesOfSC[c]+1):
                idx = w
                if policy is Policy.FIFO:
                    pass
                elif policy is Policy.LIFO:
                    idx = windowSizesOfSC[c] - idx + 1
                else:
                    raise Exception("Unknown serving principle %s" % (policy,))

                if rest_num_req_to_srv > 0:
                    admitted_num = min(predictionQueues[c][idx].length(), rest_num_req_to_srv)
                    predictionServices[c] += predictionQueues[c][idx].serve(admitted_num)
                    rest_num_req_to_srv -= admitted_num
                else:
                    break

            restResourceCapacities[chosenServer] -= num_req_to_srv * processingCosts[ingressF]

        else:
            new_reqs = predictionQueues[c][0].withdraw_all_reqs()
            predictionServices[c] += new_reqs
            restResourceCapacities[chosenServer] -= len(new_reqs) * processingCosts[ingressF]

    #  mValue[c, f] denotes the arrivals of VNF f of SC type c
    mValue = np.zeros((numOfSC, numOfNF))
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            #  If f is the ingress of SC type c,
            if i == 0:
                mValue[c, f] += len(predictionServices[c])
            #  Otherwise,
            else:
                fPre = serviceChains[c, i-1]
                for s in range(numOfServer):
                    mValue[c, f] += len(services[c][fPre][s])

    #  For the non-ingress VNF (f != serviceChains[c, 0]):
    for c in range(numOfSC):
        for i in range(1, lengthOfSC):
            f = serviceChains[c, i]
            weights = np.zeros(numOfServer)

            for s in range(numOfServer):
                fPre = serviceChains[c, i - 1]
                weights[s] = queues[c, f, s].length() * mValue[c, f] - gamma * V * len(services[c][fPre][s])

            chosenServer = weights.argmin()
            placement[(c, f)] = chosenServer

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


def QueueUpdate(t, V, queues, servicesPre, predictionServices, placement, resources):

    global finished_requests

    #  updatedQueues[c, f, s] is queue length after updating
    updatedQueues = queues
    #  communications[c] is the communication cost of SC c
    communications = np.zeros(numOfSC)
    #  services[c, f, s] denotes the number of finished requests of queue (c,f,s) of current time-slot
    services = [[[[]
                  for k in range(numOfServer)]
                 for j in range(numOfNF)]
                for i in range(numOfSC)]
    #  energies[s] denotes the energy consumption of server s
    energies = np.array(idleEnergies)

    # Appending arrivals
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            chosenServer = placement[(c, f)]

            if i == 0:
                updatedQueues[c, f, chosenServer].append_reqs(predictionServices[c])
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

                if i == lengthOfSC - 1:
                    for req in services[c][f][s]:
                        req.set_finished_time(t)
                        finished_requests[V].append(req)

    return updatedQueues, communications, services, energies


def PredictionQueueUpdate(t, V, predictionQueues):
    for c in range(numOfSC):
        predictionQueues[c] = predictionQueues[c][1:]

        if t + windowSizesOfSC[c] + 1 < maxTime:
            new_pred_queue = RequestQueue()
            # actual arrival time is t + D + 1
            new_reqs = [Request(t + windowSizesOfSC[c] + 1)
                        for i in range(arrivals[c, t + windowSizesOfSC[c] + 1])]
            new_pred_queue.append_reqs(new_reqs)
            predictionQueues[c].append(new_pred_queue)
        else:
            predictionQueues[c].append(RequestQueue())


def VNFGreedy(t, V):
    """
    :param t: current time-slot t
    :param V: current trade-off parameter V
    """
    global predictionQueueBacklogs, predictionServiceCapacities, queueBacklogs, placements, resourceAllocations, \
        actualServices, energyCosts, communicationCosts

    placements[V], predictionServiceCapacities[V], mValue = \
        VNFPlacement(t, V, predictionQueueBacklogs[V], queueBacklogs[V], actualServices[V])

    resourceAllocations[V] = ResourceAllocation(t, V, queueBacklogs[V], placements[V], mValue)

    queueBacklogs[V], communicationCosts[V], actualServices[V], energyCosts[V] = \
        QueueUpdate(t, V, queueBacklogs[V], actualServices[V], predictionServiceCapacities[V], placements[V],
                    resourceAllocations[V])

    PredictionQueueUpdate(t, V, predictionQueueBacklogs[V])


if __name__ == "__main__":

    start_time = time()

    mtime = maxTime
    avgDelays = np.array([0.0 for wsize in windowSizes])
    # avgDelays = {wsize: [] for for wsize in windowSizes}
    # for wsize in windowSizes:
    for wsize_index in range(len(windowSizes)):
        wsize = windowSizes[wsize_index]
        windowSizesOfSC = wsize * np.ones(numOfSC, dtype=int)
        """
        the observed states at the beginning of time-slot t, we maintain states for each trade-off parameter V
        """
        finished_requests = {V: [] for V in Vs}

        # A[V][c, 0], A[V][c, 1], ...
        predictionQueueBacklogs = {V: [[RequestQueue()
                                        for j in range(wsize + 1)]
                                       for i in range(numOfSC)]
                                   for V in Vs}

        # mu[V][c] is the number of requests of SC c that been sent into the system.
        predictionServiceCapacities = {V: [[] for k in range(numOfSC)]
                                       for V in Vs}

        # U. queueBacklogs[V][c, f, s] saves the queue backlogs of server s, VM f, type c.
        queueBacklogs = {V: np.array(
            [[[RequestQueue()
               for k in range(numOfServer)]
              for j in range(numOfNF)]
             for i in range(numOfSC)]
        ) for V in Vs}

        # X. placements[V][(c, f)] = s means the NF f of service chain c is placed on server s.
        placements = {V: {} for V in Vs}

        # R. resourceAllocations[V][c, f, s] denotes how many resources are allocated.
        resourceAllocations = {V: np.zeros((numOfSC, numOfNF, numOfServer)) for V in Vs}

        # H. actualServices[V][c, f, s] denotes actually how many requests are finished.
        actualServices = {V: [[[[]
                                for k in range(numOfServer)]
                               for j in range(numOfNF)]
                              for i in range(numOfSC)] for V in Vs}

        # E. energyCosts[V][s] denotes the energy consumption on server s.
        energyCosts = {V: np.zeros(numOfServer, dtype=int) for V in Vs}

        # P. communicationCosts[V][c] denotes the communication cost of chain c.
        communicationCosts = {V: np.zeros(numOfSC, dtype=int) for V in Vs}

        """
        records of states (e.g., cumulative metrics) over time
        """

        varOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
        cumulativeQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
        timeAverageOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}

        cumulativeEnergyCosts = {V: np.zeros(maxTime) for V in Vs}
        timeAverageOfEnergyCosts = {V: np.zeros(maxTime) for V in Vs}

        cumulativeCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}
        timeAverageOfCommunicationCosts = {V: np.zeros(maxTime) for V in Vs}

        for V in Vs:
            for c in range(numOfSC):
                for d in range(windowSizesOfSC[c] + 1):
                    if d >= mtime:
                        break
                    new_reqs = [Request(d) for k in range(arrivals[c, d])]
                    predictionQueueBacklogs[V][c][d].append_reqs(new_reqs)

        for V in Vs:
            for t in range(mtime):
                if t % int(1e4) == 0:
                    print("Now V is %s and time slot is %s" % (V, t))

                VNFGreedy(t, V)

        delays = {V: list(map(lambda e: (e.id, e.get_latency()), finished_requests[V])) for V in Vs}
        delaysNew = np.array([delays[V] for V in Vs])
        avgDelay = sum(delaysNew[0][:, 1]) / float(len(delaysNew[0][:, 1]))
        avgDelays[wsize_index] = avgDelay
        print("Average delay (wsize = %s) is: %s" % (wsize, avgDelay))

    np.save("resultsPOSCAS_perfect_Cap/capacity200/avgDelays.npy", avgDelays)
    end_time = time()
    np.save("resultsPOSCAS_perfect_Cap/capacity200/runtime.npy", end_time - start_time)
    print("Simulation ends. Duration is %s sec." % (end_time - start_time,))
