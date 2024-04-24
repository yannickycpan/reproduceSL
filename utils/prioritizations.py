import numpy as np
import math

def true_1storder_priority(dataset):
    priorities = np.zeros(dataset.shape[0])
    priorities[dataset[:,0] < 0] = np.abs(4*math.pi*np.cos(4*math.pi*dataset[dataset[:,0] < 0, 0]))
    priorities[dataset[:,0] >= 0] = np.abs(0.2*math.pi*np.cos(0.2*math.pi*dataset[dataset[:,0] >= 0, 0]))
    priorities = priorities / np.sum(priorities)
    #exppriority = np.exp(priorities)
    #return exppriority/np.sum(exppriority)
    ratio = 1./(priorities + 1e-6) * 1./priorities.shape[0]
    return priorities, ratio

def true_2ndorder_priority(dataset):
    priorities = np.zeros(dataset.shape[0])
    priorities[dataset[:,0] < 0] = np.abs(16*math.pi**2*np.sin(4*math.pi*dataset[dataset[:,0] < 0, 0]))
    priorities[dataset[:,0] >= 0] = np.abs(0.04*math.pi**2*np.sin(0.2*math.pi*dataset[dataset[:,0] >= 0, 0]))
    priorities = priorities / np.sum(priorities)
    #exppriority = np.exp(priorities)
    #return exppriority/np.sum(exppriority)
    ratio = 1./(priorities + 1e-6) * 1./priorities.shape[0]
    return priorities, ratio

def prioritized_sampling(priorities, mini_batch_size, ratio = None, niter = None):
    inds = np.random.randint(0, priorities.shape[0], mini_batch_size)
    weights = np.ones(mini_batch_size)
    #proportion = int(mini_batch_size*(1. - niter/20000.))
    proportion = 16
    if proportion > 0:
        pinds = np.random.choice(priorities.shape[0], proportion, replace=False,
                                       p=priorities)
        inds[-proportion:] = np.array(pinds)
        #weights[-proportion:] = ratio[pinds]
    return inds, weights

def prioritized_left_part(X, mini_batch_size, ratio = None, niter = None):
    inds = np.random.randint(0, X.shape[0], mini_batch_size)
    weights = np.ones(mini_batch_size)
    #proportion = int(mini_batch_size*(1. - niter/20000.))
    proportion = 16
    if proportion > 0:
        validinds = np.arange(X.shape[0])
        validinds = validinds[X[:,0] < 0]
        pinds = np.random.choice(validinds, proportion, replace=False)
        inds[-proportion:] = np.array(pinds)
    return inds, weights