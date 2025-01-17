import sys

import numpy as np

import time

import json

from numba import njit, jit, prange, int_, float_


# useful functions

@jit(int_(int_, int_, float_, int_, int_))
def simulate_trajectory(N, M, s, tmax, initial_state=1):
    #states = np.zeros(tmax)
    #states[0] = initial_state
    current_state = initial_state
    t=0
    b = True
    while t<tmax-1 and b:
        #perform hypergeometrical sampling
        nb_mutants_before_update = np.random.hypergeometric(current_state, N - current_state, M)

        # perform binomial sampling 
        x = current_state / N
        prob = x*(1+s) / (1+x*s)
        n_trials = M
        nb_mutants_after_update = np.random.binomial(n_trials, prob)

        # update nb of mutants in the node
        
        if M==1: #Moran case
            rand = np.random.rand()
            if rand <= x*(1-prob):
                current_state -= 1
            elif rand <= x*(1-prob) + prob*(1-x):
                current_state += 1
        
        if M > 1: 
            current_state = current_state - nb_mutants_before_update + nb_mutants_after_update


        b = 0 < current_state and current_state < N
        t+=1
        #states[t] = current_state
    
    #if t<tmax:
        #states[t+1:] = current_state

    if current_state == N:
        fixation = 1
    else:
        fixation = 0
    return fixation

@njit(parallel=True)
def simulate_multiple_trajectories(N, M, s, tmax, nb_trajectories=100, initial_state=1):
    #all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))
    #all_trajectories[:,0] = initial_state

    #fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    count_fixation = 0


    #for trajectory_index in tqdm(range(nb_trajectories)):
    for trajectory_index in prange(nb_trajectories):
        #if trajectory_index%(10**3) == 0:
            #print('trajectory:', trajectory_index)
        fixation = simulate_trajectory(N,M,s,tmax,initial_state)

        count_fixation += fixation

        #fixation_seq[trajectory_index] = fixation

        #all_trajectories[trajectory_index,:] = states[:]
        
        

    return count_fixation





