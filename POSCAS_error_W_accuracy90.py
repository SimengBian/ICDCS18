import numpy as np
from time import time
from request import *

filename = 'configAccuracy90/'
"""
configurations
"""
scInformation = np.load(filename + "SC Information.npz")
nfInformation = np.load(filename + "NF Information.npz")
snInformation = np.load(filename + "SN Information.npz")
systemInformation = np.load(filename + "System Information.npz")

observed_arrivals = np.load(filename + "Arrival_error.npz")["arrivals_error"]
arrivals = systemInformation['arrivals']

# System Information
# maxTime = int(systemInformation['maxTime'])
maxTime = int(1e2)
gamma = int(systemInformation['gamma'])
print(gamma)
# Vs = systemInformation['Vs']
Vs = [40]
unitCommCost = int(systemInformation['unitCommCost'])
alpha = int(systemInformation['alpha'])

# Network Function Information
numOfNF = int(nfInformation['numOfNF'])
processingCosts = nfInformation['processingCosts']

# Service Chain Information
numOfSC = int(scInformation['numOfSC'])
lengthOfSC = int(scInformation['lengthOfSC'])
serviceChains = scInformation['serviceChains']  # serviceChains[c, i]
print(serviceChains)

windowSizes = [i*2 for i in range(0, 6)] + [j*5 for j in range(4, 21)]

policy = Policy.FIFO

# Substrate Network Information
# numOfServer = int(snInformation['numOfServer'])
numOfServer = 5
# serverCapacities = snInformation['serverCapacities']  # serverCapacities[s]
serverCapacities = np.array([300 for s in range(numOfServer)])
print(serverCapacities)
idleEnergies = snInformation['idleEnergies']  # idleEnergies[s]
maxEnergies = snInformation['maxEnergies']  # maxEnergies[s]


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

    # For the ingress VNF (f = serviceChains[c, 0])
    for c in range(numOfSC):
        ingressF = serviceChains[c, 0]
        chosenServer = int(np.argmin(queues[c, ingressF, :]))
        placement[(c, ingressF)] = chosenServer

        # Pre-admission is enabled only when the server has spare resources
        if restResourceCapacities[chosenServer] > predictionQueues[c][0].length() * processingCosts[ingressF]:
            # Calculate the queue size of each prediction queue of a SC type
            predictionQueueSizes = list(map(lambda e: e.length(), predictionQueues[c]))
            num_req_to_srv = int(min(np.floor(restResourceCapacities[chosenServer] / processingCosts[ingressF]),
                                     np.sum(predictionQueueSizes)))

            # All requests arriving in current time-slot must be forwarded
            rest_num_req_to_srv = num_req_to_srv
            rest_num_req_to_srv -= predictionQueues[c][0].length()
            # Append all requests  arriving in current time slot
            # to the admission buffer.
            predictionServices[c] += predictionQueues[c][0].withdraw_all_reqs()

            # Pre-admit requests in future windows according to
            # some scheduling principle, e.g., FIFO, LIFO
            for w in range(1, wsize+1):
                idx = w
                if policy is Policy.FIFO:
                    pass
                elif policy is Policy.LIFO:
                    idx = wsize - idx + 1
                else:
                    raise Exception("Unknown serving principle %s" % (policy,))

                if rest_num_req_to_srv > 0:
                    admitted_num = min(predictionQueues[c][idx].length(), rest_num_req_to_srv)
                    admitted_reqs = predictionQueues[c][idx].serve(admitted_num)

                    # change all admitted requests' state into ADMITTED
                    for req in admitted_reqs:
                        req.admit()

                    predictionServices[c] += admitted_reqs
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

                for req in predictionServices[c]:
                    req.set_loc(c, f, chosenServer)

            else:
                fPre = serviceChains[c, i - 1]
                for s in range(numOfServer):
                    new_reqs = servicesPre[c][fPre][s]

                    for req in new_reqs:
                        req.transfer()
                        req.set_loc(c, f, s)

                    updatedQueues[c, f, chosenServer].append_reqs(new_reqs)

                    for req in new_reqs:
                        req.set_loc(c, f, chosenServer)

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

                for req in services[c][f][s]:
                    req.transfer()
                    req.set_loc(c, f, s)

                if i == lengthOfSC - 1:
                    for req in services[c][f][s]:
                        req.set_finished_time(t)
                        finished_requests[V].append(req)

    return updatedQueues, communications, services, energies


def PredictionQueueUpdate(t, V, predictionQueues):
    for c in range(numOfSC):
        predictionQueues[c] = predictionQueues[c][1:]

        if t + wsize + 1 < maxTime:
            new_pred_queue = RequestQueue()
            # actual arrival time is t + D + 1
            new_reqs = [Request(t + wsize + 1)
                        for i in range(observed_arrivals[c, t + wsize + 1])]
            new_pred_queue.append_reqs(new_reqs)
            predictionQueues[c].append(new_pred_queue)
        else:
            predictionQueues[c].append(RequestQueue())


def VNFGreedy(t, V, request_record):
    global predictionQueueBacklogs, predictionServiceCapacities, queueBacklogs, placements, resourceAllocations, \
        actualServices, finished_requests, energyCosts, communicationCosts

    errors = []
    for c in range(numOfSC):
        error = observed_arrivals[c, t] - arrivals[c, t]
        errors.append(error)

        # If error is positive, then the actual arrivals are over-estimated
        # As a result, some pre-served requests should be dropped
        if error > 0:
            rest_req_num = predictionQueueBacklogs[V][c][0].length()
            dropped_reqs = predictionQueueBacklogs[V][c][0].serve(error)

            for req in dropped_reqs:
                req.drop()

            if rest_req_num >= error:
                pass
            else:
                num_to_drop = error - rest_req_num

                # drop some requests either in the system
                # or in finished requests list..
                observed_reqs = request_record[c][t] + []
                sorted(observed_reqs, key=lambda _req: _req.state)

                for idx in range(num_to_drop):
                    req = observed_reqs[idx]

                    if req.state is Request.ADMITTED_IN:
                        # make sure the request is truly admitted
                        assert Request.INVALID_LOCATION not in req.loc

                        sc, f, server = req.loc
                        queueBacklogs[V][sc, f, server].remove(req)
                        req.drop()

                    elif req.state is Request.ADMITTED_OUT:
                        assert Request.INVALID_LOCATION not in req.loc

                        sc, f, server = req.loc
                        actualServices[V][sc][f][server].remove(req)
                        req.drop()

                    elif req.state is Request.FINISHED:
                        finished_requests[V].remove(req)
                        req.drop()

                    else:
                        pass

        # else, the actual arrivals are under-estimated.
        # In this case, missing requests should be appended to current queue backlogs.
        elif error < 0:
            new_reqs = [Request(t) for k in range(abs(error))]
            # missing_req_counts[V] += len(new_reqs)
            predictionQueueBacklogs[V][c][0].append_reqs(new_reqs)
        else:
            pass

    placements[V], predictionServiceCapacities[V], mValue = \
        VNFPlacement(t, V, predictionQueueBacklogs[V], queueBacklogs[V], actualServices[V])

    resourceAllocations[V] = ResourceAllocation(t, V, queueBacklogs[V], placements[V], mValue)

    queueBacklogs[V], communicationCosts[V], actualServices[V], energyCosts[V] = \
        QueueUpdate(t, V, queueBacklogs[V], actualServices[V], predictionServiceCapacities[V], placements[V],
                    resourceAllocations[V])

    PredictionQueueUpdate(t, V, predictionQueueBacklogs[V])

    # Update track of requests in the range of lookahead window
    for c in range(numOfSC):
        del request_record[c][t]
        if t + wsize + 1 < maxTime:
            request_record[c][t + wsize + 1] = \
                predictionQueueBacklogs[V][c][-1].record_reqs()
    # End of track update


if __name__ == "__main__":

    mTime = maxTime

    start_time = time()

    avgDelays = {wsize: 0.0 for wsize in range(len(windowSizes))}
    for wsize in windowSizes:
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

        for V in Vs:
            for c in range(numOfSC):
                for d in range(wsize + 1):
                    if d >= mTime:
                        break
                    new_reqs = [Request(d) for k in range(observed_arrivals[c, d])]
                    predictionQueueBacklogs[V][c][d].append_reqs(new_reqs)

        for V in Vs:
            request_record = []

            for c in range(numOfSC):
                request_record.append({})
                for d in range(wsize + 1):
                    if d >= mTime:
                        break
                    else:
                        request_record[c][d] = predictionQueueBacklogs[V][c][d].record_reqs()
            # end of initial tracking

            for t in range(mTime):
                if t % int(1e4) == 0:
                    print("Now V is %s and time slot is %s" % (V, t))

                VNFGreedy(t, V, request_record)

        delays = {V: list(map(lambda e: (e.id, e.get_latency()), finished_requests[V])) for V in Vs}
        delaysNew = np.array([delays[V] for V in Vs])
        avgDelay = sum(delaysNew[0][:, 1]) / float(len(delaysNew[0][:, 1]))
        avgDelays[wsize] = avgDelay
        print("Average delay (wsize = %s) is: %s" % (wsize, avgDelay))

    avgDelaysNew = np.array([avgDelays[wsize] for wsize in windowSizes])
    np.save("resultsPOSCAS_error_W/accuracy90/avgDelays.npy", avgDelaysNew)
    end_time = time()
    np.save("resultsPOSCAS_error_W/accuracy90/runtime.npy", end_time - start_time)
    print("Simulation ends. Duration is %s sec." % (end_time - start_time,))
