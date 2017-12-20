import numpy as np

filename = 'config1/'
'''
Network Function (NF)
'''
numOfNF = 30  # number of NF types

# processingCosts = np.array([5, 2, 1])
processingCosts = np.array(15 * [1] + 10 * [2] + 5 * [5])

np.savez(filename + "NF Information.npz", numOfNF=numOfNF, processingCosts=processingCosts)
