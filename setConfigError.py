import numpy as np
import random
import util

filename = 'configAccuracy10/'

scInformation = np.load(filename + "SC Information.npz")
# nfInformation = np.load(filename + "NF Information.npz")
# snInformation = np.load(filename + "SN Information.npz")
systemInformation = np.load(filename + "System Information.npz")

# numOfNF = int(nfInformation['numOfNF'])
numOfSC = int(scInformation['numOfSC'])

# gamma = systemInformation['gamma']
# alpha = systemInformation['alpha']
# Vs = systemInformation['Vs']
arrivals = systemInformation['arrivals']
maxTime = systemInformation['maxTime']
# maxTime = 100
# unitCommCost = systemInformation['unitCommCost']
# maxWindowSize = systemInformation['maxWindowSize']

"""
delta(4.0) leads to succ. pr. 10.0%
delta(2.0) leads to succ. pr. 20.0%
delta(1.3) leads to succ. pr. 30.0%
delta(0.95) leads to succ. pr. 40.0%
delta(0.74) leads to succ. pr. 50.0%
delta(0.59) leads to succ. pr. 60.0%
delta(0.48) leads to succ. pr. 70.0%
delta(0.39) leads to succ. pr. 80.0%
delta(0.3) leads to succ. pr. 90.0%
delta(0.0) leads to succ. pr. 100.0%
"""
pr_delta_dict = {
    0.1: 4.0,
    0.2: 2.0,
    0.3: 1.3,
    0.4: 0.95,
    0.5: 0.74,
    0.6: 0.59,
    0.7: 0.48,
    0.8: 0.39,
    0.9: 0.3,
    1.0: 0.0,
}

# To adjust the error probability of prediction
succ_pr = 0.1

mu = 0
sigma = pr_delta_dict[succ_pr]

arrivals_error = arrivals

for c in range(numOfSC):
    for t in range(maxTime):
        error = int(round(random.normalvariate(mu, sigma)))
        arrivals_error[c, t] = max(arrivals_error[c, t] + error, 0)

np.savez(filename + "Arrival_error.npz", arrivals_error=arrivals_error)