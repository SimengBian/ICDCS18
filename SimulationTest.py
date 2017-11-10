import numpy as np

# Load configurations
scInformation = np.load("config/SC Information.npz")
nfInformation = np.load("config/NF Information.npz")
snInformation = np.load("config/SN Information.npz")
systemInformation = np.load("config/System Information.npz")

# System Information
maxTime = systemInformation['maxTime']
gamma = systemInformation['gamma']
Vs = systemInformation['Vs']
Vs = [Vs[-1]]
lenOfVs = len(Vs)
arrivals = systemInformation['arrivals']  # arrivals[c, t]

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

# the observed states, we maintain states for each parameter V

# queueBacklogs[V][c, f, s] saves the queue backlogs of server s, VM f, type c. **current time-slot t**.
# (Notice that f here means both Network Functions and VMs on the server)
queueBacklogs = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# resourceAllocations[V][c, f, s] denotes how many resources is allocated to type c on VM f, on server s. **previous time-slot t-1**
resourceAllocations = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# actualServices[V][c, f, s] denotes how many requests are finished. **previous time-slot t-1**
actualServices = {V: np.zeros((numOfSC, numOfNF, numOfServer), dtype=int) for V in Vs}

# placements[V][(c, f)] = s means the NF f of service chain c is placed on server s. **previous time-slot t-1**
placements = {V: {} for V in Vs}

# energyCosts[V][s] denotes the energy consumption on server s. **previous time-slot t-1**
energyCosts = {V: np.zeros(numOfServer, dtype=int) for V in Vs}
partitionCosts = {V: np.zeros(numOfSC, dtype=int) for V in Vs}

cumulativeQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfQueueBacklogs = {V: np.zeros(maxTime) for V in Vs}

cumulativeEnergyCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfEnergyCosts = {V: np.zeros(maxTime) for V in Vs}

cumulativePartitionCosts = {V: np.zeros(maxTime) for V in Vs}
timeAverageOfPartitionCosts = {V: np.zeros(maxTime) for V in Vs}

def generateSpace(numOfNF, numOfServer):
    space = []
    for space_index in range(numOfServer**numOfNF):
        space_temp = {}
        k = numOfNF
        for i in range(numOfNF):
            tem_index = space_index%numOfServer
            space_temp[k-1] = tem_index
            k -= 1
            space_index = int(space_index/numOfServer)
        space.append(space_temp)
    return space

possiblePlacements = generateSpace(numOfNF, numOfServer)
# print(possiblePlacements)


def VNFPlacement(V, queues, resources):
    # print("VNFPlacement")
    placement = {}
    partitions = np.zeros(numOfSC)

    mValue = np.zeros((numOfSC, numOfNF))
    for c in range(numOfSC):
        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            if i == 0:
                mValue[c, f] += arrivals[c, f]
            else:
                fPre = serviceChains[c, i-1]
                for s in range(numOfServer):
                    mValue[c, f] += resources[c, fPre, s] / processingCost[fPre] # maximum # of arrivals but not actual arrival!

    #  For each service chain type c
    for c in range(numOfSC):
        optVPOValue = float('inf')
        optJ = 0
        optPartition = 0
        for j in range(len(possiblePlacements)):
            curPlacement = possiblePlacements[j]
            VPOValue = 0
            curPartition = lengthOfSC - 1
            for i in range(lengthOfSC):
                f = serviceChains[c, i]
                # pair = tuple([c, f])
                server = curPlacement[f]
                VPOValue += queues[c, f, server] * mValue[c, f]

            for i in range(1, lengthOfSC):
                f = serviceChains[c, i]
                fpre = serviceChains[c, i-1]
                if curPlacement[f] == curPlacement[fpre]:
                    VPOValue -= gamma * V
                    curPartition -= 1

            if VPOValue < optVPOValue:
                optVPOValue = VPOValue
                optJ = j
                optPartition = curPartition

        for i in range(lengthOfSC):
            f = serviceChains[c, i]
            pair = tuple([c, f])
            chosenServer = possiblePlacements[optJ][f]
            placement[pair] = chosenServer
        partitions[c] = optPartition

    return placement, partitions


def ResourceAllocation(V, queues):
    # print("ResourceAllocation")
    allocation = np.zeros((numOfSC, numOfNF, numOfServer))
    services = np.zeros((numOfSC, numOfNF, numOfServer))
    energies = np.zeros(numOfServer)
    for s in range(numOfServer):
        term1 = V * (maxEnergies[s] - idleEnergies[s]) / float(serverCapacities[s])
        weights = term1 * np.ones((numOfSC, numOfNF))
        for c in range(numOfSC):
            for f in range(numOfNF):
                weights[c, f] -= queues[c, f, s] / float(processingCost[f])
        (chosenType, chosenVM) = np.unravel_index(weights.argmin(), weights.shape)
        if weights[chosenType, chosenVM] < 0:
            allocation[chosenType, chosenVM, s] = serverCapacities[s]
            services[chosenType, chosenVM, s] = min(serverCapacities[s] / processingCost[chosenVM], queues[chosenType, chosenVM, s])
            energies[s] += services[chosenType, chosenVM, s] * processingCost[chosenVM]

    return allocation, services, energies


def QueueUpdate(V, queues, services, placement):
    # print("QueueUpdate")
    for c in range(numOfSC):
        for f in range(numOfNF):
            for s in range(numOfServer):
                queues[c, f, s] -= services[c, f, s]
                if tuple([c, f]) in placement.keys() and placement[tuple([c, f])] == s:
                    if f == serviceChains[c, 0]:
                        queues[c, f, s] += arrivals[c, t]
                    else:
                        chain = list(serviceChains[c, :])
                        fPre = serviceChains[c][chain.index(f) - 1]
                        for ss in range(numOfServer):
                            queues[c, f, s] += services[c, fPre, ss]

    return queues

def VNFGreedy(t, V):
    '''
    :param t: current time-slot.
    :param V: the trade-off parameter of queue backlog and cost
    :return: the total queue backlogs and total energy cost incurred in this time-slot
    '''
    global queueBacklogs, VMStates, resourceAllocations, placements

    placements[V], partitionCosts[V] = VNFPlacement(V, queueBacklogs[V], resourceAllocations[V])

    resourceAllocations[V], actualServices[V], energyCosts[V] = ResourceAllocation(V, queueBacklogs[V])

    queueBacklogs[V] = QueueUpdate(V, queueBacklogs[V], actualServices[V], placements[V])


if __name__ == "__main__":
    for V in Vs:
        for t in range(maxTime):
            print("Now V is", V, " and time slot is", t)
            VNFGreedy(t, V)

            cumulativeQueueBacklogs[V][t] += cumulativeQueueBacklogs[V][t - 1] + np.sum(queueBacklogs[V])
            timeAverageOfQueueBacklogs[V][t] = cumulativeQueueBacklogs[V][t] / float(t + 1)
            cumulativeEnergyCosts[V][t] += cumulativeEnergyCosts[V][t - 1] + np.sum(energyCosts[V])
            timeAverageOfEnergyCosts[V][t] = cumulativeEnergyCosts[V][t] / float(t + 1)
            cumulativePartitionCosts[V][t] += cumulativePartitionCosts[V][t-1] + np.sum(partitionCosts[V])
            timeAverageOfPartitionCosts[V][t] = cumulativePartitionCosts[V][t] / float(t + 1)

    # Be careful of the Vs mapping to i. [1, 10, 20, 50, 100] --> [0, 1, 2, 3, 4]
    timeAverageOfQueueBacklogsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfEnergyCostsNew = np.zeros((lenOfVs, maxTime))
    timeAverageOfPartitionCostsNew = np.zeros((lenOfVs, maxTime))
    for i in range(lenOfVs):
        timeAverageOfQueueBacklogsNew[i, :] = np.array(timeAverageOfQueueBacklogs[Vs[i]])
        timeAverageOfEnergyCostsNew[i, :] = np.array(timeAverageOfEnergyCosts[Vs[i]])
        timeAverageOfPartitionCostsNew[i, :] = np.array(timeAverageOfPartitionCosts[Vs[i]])

    np.save("resultsTest/timeAverageOfQueueBacklogs.npy", timeAverageOfQueueBacklogsNew)
    np.save("resultsTest/timeAverageOfEnergyCosts", timeAverageOfEnergyCostsNew)
    np.save("resultsTest/timeAverageOfPartitionCosts", timeAverageOfPartitionCostsNew)
    print("end")