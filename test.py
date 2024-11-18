## Varying s

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

import json

from numba import njit, prange, jit, int_, float_

import time


# useful functions

@jit
def simulate_clique(N, M, nb_colonies, migration_rate, s, tmax):
    assert 1 - (nb_colonies - 1) * migration_rate >= 0

    b = True
    t = 1

    #i_nodes = np.zeros(nb_colonies, dtype=int_)  # list of the number of mutants in each node
    #i_nodes[0] = 1
    i_nodes = (N-1) * np.ones(nb_colonies, dtype = int_)
    #N_nodes = N * np.ones(nb_colonies, dtype=int_)  # population size in each node
    #M_nodes = M * np.ones(nb_colonies, dtype=int_)  # update size in each node

    # Creating a directed graph (migration rates)
    DG = np.zeros((nb_colonies, nb_colonies), dtype=float_)
    for node1 in range(nb_colonies):
        for node2 in range(nb_colonies):
            if node1 == node2:
                weight = 1 - (nb_colonies - 1) * migration_rate
            else:
                weight = migration_rate
            DG[node1, node2] = weight

    trajectories = np.zeros((tmax, nb_colonies))
    trajectories[0, :] = i_nodes

    while t < tmax and b:
        # Choose a random node
        selected_node = np.random.randint(0, nb_colonies) # not (nb_colonies - 1) when using np.random !!!!!

        # Hypergeometrical sampling
        ngood = i_nodes[selected_node]
        nbad = N - ngood
        nb_mutants_before_update = np.random.hypergeometric(ngood, nbad, M)

        # Binomial sampling
        x_tilde = sum([i_nodes[k] * DG[k, selected_node] for k in range(nb_colonies)])/N
        print('x_tilde:', x_tilde)
        prob = x_tilde * (1 + s) / (1 + x_tilde * s)
        print('prob:', prob)
        n_trials = M
        nb_mutants_after_update = np.random.binomial(n_trials, prob)

        # Update mutants in the node
        i_nodes[selected_node] = ngood - nb_mutants_before_update + nb_mutants_after_update

        for node in range(nb_colonies):
            trajectories[t, node] = i_nodes[node]
        t += 1
        b = sum(i_nodes) < nb_colonies*N and max(i_nodes) > 0

    if sum(i_nodes) == nb_colonies*N:
        fixation = 1
    else:
        fixation = 0

    if t < tmax:
        for tt in range(t, tmax):
            trajectories[tt, :] = trajectories[t - 1, :]

    return fixation, trajectories



@njit(parallel=True)
def simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories=100):
    all_trajectories = np.zeros((int(nb_trajectories), int(tmax)))
    #fixation_seq = np.zeros(nb_trajectories)
    count_fixation = 0

    for trajectory_index in prange(nb_trajectories):  #parallelized
        #print('trajectory:', trajectory_index)
        fixation, trajectories = simulate_clique(N, M, nb_colonies, migration_rate, s, tmax)

        count_fixation += fixation
        #fixation_seq[trajectory_index] = fixation
        all_trajectories[trajectory_index, :] = np.sum(trajectories, axis=1)

    return count_fixation, all_trajectories


def plot_traj(N, M, nb_colonies, migration_rate, s, tmax):


    fixation, trajectories = simulate_clique(N, M, nb_colonies, migration_rate, s, tmax)

    xx = list(range(tmax))
    for k in range(nb_colonies):
        yy = trajectories[:,k]
        plt.plot(xx, yy, label = f"dème {k}")
    tot_mutants = np.sum(trajectories, axis=1)
    plt.plot(xx, tot_mutants, 'k--', alpha=0.5, label = "total")
    plt.xlim([0, 100])
    plt.legend()
    plt.show()
    print("fixation: ",fixation)

def plot_multiple_trajectories(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories=100):
    count_fixation, all_trajectories = simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories)
    xx = list(range(tmax))
    for k in range(nb_trajectories):
        yy = all_trajectories[k,:]
        plt.plot(xx, yy, 'b-', alpha = 0.2)
    plt.xlim([0, 100])
    plt.show()
    print("count fixation: ",count_fixation)


if __name__ == "__main__":
    N = 10
    M = 5
    nb_colonies = 3
    migration_rate = 0.1
    s = 0.1
    tmax = 50000

    plot_multiple_trajectories(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories=50)


