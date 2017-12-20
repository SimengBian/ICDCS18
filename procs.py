from random import *
from numpy import var

td = [0, 0.08514851485148515, 0.15247524752475247, 0.19603960396039605, 0.2297029702970297, 0.2514851485148515, 0.26732673267326734, 0.28316831683168314, 0.2891089108910891, 0.297029702970297, 0.299009900990099, 0.30297029702970296, 0.3069306930693069, 0.41386138613861384, 0.49504950495049505, 0.592079207920792, 0.7346534653465346, 0.8138613861386138, 0.9603960396039604, 1.0]
xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 250, 400, 450, 650, 1000, 1600, 8000, 25000]
lenOfTS = 10000


def interval_generator(r):
    rand = random()
    for i in range(1, len(td)):
        if rand <= td[i]:
            return 0.7*(xs[i] + xs[i-1])/1e4
        else:
            continue
    raise Exception("It shouldn't reach here..")


procs = {
    'exp': (lambda lam: expovariate(lam)),
    'pareto': (lambda alpha: paretovariate(0.966*alpha+1)-1),
    # 'uni': (lambda r: uniform(0, 2/r)),
    'constant': (lambda r: 1.0/r),
    'normal': (lambda r: max(0, normalvariate(0.935/r, 0.075))),
    'trace': interval_generator
}


def generate(ns, s, arrProc, arrRate):
    total = 0
    count = 0
    
    while True:
        total += arrProc(arrRate)
        if total < 1: count += 1
        else: break
            
    return count


for p in procs.keys():
    As = [generate(10, k, procs[p], 5.88) for k in range(100000)]
    print(p, sum(As)/float(len(As)))

# As = [generate(10, k, procs['trace'], 5.88) for k in range(500000)]
# print('var', var(As), 5.88)