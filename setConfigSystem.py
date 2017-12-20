import random
import numpy as np

filename = 'config1/'
td = [0, 0.08514851485148515, 0.15247524752475247, 0.19603960396039605, 0.2297029702970297, 0.2514851485148515, 0.26732673267326734, 0.28316831683168314, 0.2891089108910891, 0.297029702970297, 0.299009900990099, 0.30297029702970296, 0.3069306930693069, 0.41386138613861384, 0.49504950495049505, 0.592079207920792, 0.7346534653465346, 0.8138613861386138, 0.9603960396039604, 1.0]
xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 250, 400, 450, 650, 1000, 1600, 8000, 25000]  # us

scInformation = np.load(filename + "SC Information.npz")
numOfSC = int(scInformation['numOfSC'])
pOfSC = scInformation['pOfSC']

'''
System Information
'''
arrivalRate = 5.88
maxTime = int(1e6)
# Vs = [1]
Vs = [i*5 for i in range(1, 21)]
alpha = 1
gamma = 100
unitCommCost = 1
# maxWindowSize = 20


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
    totalArrivals = np.zeros((numOfSC, maxTime), dtype=int)
    currentTime = 0
    arrivalTimePoints = []
    while currentTime < maxTime:

        if len(arrivalTimePoints) % 1200000 == 0:
            print(currentTime, ":", len(arrivalTimePoints))

        interval = arrProc(arrRate)
        currentTime += interval
        # SCtype = random.choice(range(numOfSC))
        SCtype = np.random.choice(range(numOfSC), p=pOfSC)
        arrivalTimePoints.append((SCtype, currentTime))

    print("Length:", len(arrivalTimePoints))

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


from time import time

#  arrivals[c, t] is the number of arrival requests of SC type c at time-slot t
begin_time = time()
arrivals = generate(maxTime, arrivalRate, procs['trace'])
end_time = time()

print("Duration:", end_time - begin_time)

np.savez(filename + "System Information.npz", unitCommCost=unitCommCost, maxTime=maxTime, arrivals=arrivals, Vs=Vs, gamma=gamma, alpha=alpha, maxWindowSize=maxWindowSize)