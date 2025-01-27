## Varying s

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


import json

from numba import njit, prange, jit, int_, float_

import time


# useful functions

@jit(int_(int_, int_, int_, float_, float_, int_ ))
def simulate_clique(N, M, nb_colonies, migration_rate, s, tmax):
    assert 1 - (nb_colonies - 1) * migration_rate >= 0

    b = True
    t = 1

    i_nodes = np.zeros(nb_colonies, dtype=int_)  # list of the number of mutants in each node
    i_nodes[0] = 1
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

    #trajectories = np.zeros((tmax, nb_colonies))
    #trajectories[0, :] = i_nodes

    while t < tmax and b:
        # Choose a random node
        selected_node = np.random.randint(0, nb_colonies) #!!!!

        # Hypergeometrical sampling
        ngood = i_nodes[selected_node]
        nbad = N - ngood
        nb_mutants_before_update = np.random.hypergeometric(ngood, nbad, M)

        # Binomial sampling
        x_tilde = sum([i_nodes[k] * DG[k, selected_node]/N for k in range(nb_colonies)])
        #print('x_tilde:', x_tilde)
        prob = x_tilde * (1 + s) / (1 + x_tilde * s)
        n_trials = M
        nb_mutants_after_update = np.random.binomial(n_trials, prob)

        # Update mutants in the node
        i_nodes[selected_node] = ngood - nb_mutants_before_update + nb_mutants_after_update

        #trajectories[t, :] = i_nodes
        t += 1
        b = sum(i_nodes) < nb_colonies*N and (i_nodes > 0).any()

    if sum(i_nodes) == nb_colonies*N:
        fixation = 1
    else:
        fixation = 0

    #if t < tmax:
        #for tt in range(t, tmax):
            #trajectories[tt, :] = trajectories[t - 1, :]

    return fixation



@njit(parallel=True)
def simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories=100):
    #all_trajectories = np.zeros((int(nb_trajectories), int(tmax)))
    #fixation_seq = np.zeros(nb_trajectories)
    count_fixation = 0

    for trajectory_index in prange(nb_trajectories):  #parallelized
        #print('trajectory:', trajectory_index)
        fixation = simulate_clique(N, M, nb_colonies, migration_rate, s, tmax)

        count_fixation += fixation
        #fixation_seq[trajectory_index] = fixation
        #all_trajectories[trajectory_index, :] = np.sum(trajectories, axis=1)

    return count_fixation

def phi(N,s,rho,x):
    num = 1 - np.exp(-2*N*s*x / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom


# generating the graph

def run(nb_trajectories, N, nb_colonies, plot=True):
    s_range = np.logspace(-4, -1, num=10)
    tmax = 50000
    
    migration_rate = 0.1


    Ms = np.array([1, N//4, N//2, 3*N//4, N])
    rhos = Ms* 1. /N

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(Ms)))

    fig, ax = plt.subplots()

    fig_data = np.zeros((5, len(Ms)*len(s_range)))

    N_tot = N*nb_colonies



    for i,M in enumerate(Ms):
        print('M:',M)
        fig_data[0, i*len(s_range):(i+1)*len(s_range)] = M*np.ones(len(s_range))

        color = colors[i]

        for j,s in enumerate(s_range):
            print('s:',s)

            count_fixation = simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories)
            fixation_freq = count_fixation / nb_trajectories
            std = np.sqrt(fixation_freq * (1-fixation_freq) / nb_trajectories)
            
            fig_data[1, i*len(s_range) + j] = s
            fig_data[2, i*len(s_range) + j] = fixation_freq
            fig_data[3, i*len(s_range) + j] = 2*std
            fig_data[4, i*len(s_range) + j] = count_fixation
        if plot:
            ax.errorbar(s_range, fig_data[2,i*len(s_range):(i+1)*len(s_range)], yerr= fig_data[3,i*len(s_range):(i+1)*len(s_range)], label = f"M={M} (update fraction: {round(M/N,2)} )", fmt = 'o', alpha=0.5, color=color)
            #ax.plot(s_range, [phi(N_tot,s,M/N,1/N_tot) for s in s_range], label = f"M={M} (update fraction: {round(M/N,2)} )", color= color)


    if plot:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel('Relative fitness')
        ax.set_ylabel('Fixation probability')
        ax.legend()
        plt.savefig(f'clique_results/clique-graphe_phi-vs-s_n-traj={nb_trajectories}_N={N}_D={nb_colonies}.png')



    simulation_parameters = {
        'N_tot': N_tot,
        'N':N,
        'number of colonies':nb_colonies,
        'migration rate': migration_rate,
        'tmax':tmax,
        'nb_trajectories':nb_trajectories,
        's_range':(min(s_range), max(s_range))
    }
    return simulation_parameters, fig_data


if __name__ == "__main__":
    #nb_trajectories=10**7
    #run(10, plot=False) #compiling the function

    N = 10
    nb_colonies = 10


    nb_trajectories = 5*10000
    
    start_time = time.time()

    simulation_parameters, fig_data = run(nb_trajectories, N, nb_colonies)

    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time)

    df = pd.DataFrame({
        'M': fig_data[0,:],
        's': fig_data[1,:],
        'fixation_freq': fig_data[2,:],
        'fixation_err': fig_data[3,:],
        'count_fixation': fig_data[4,:]
    })

    df.to_csv(f'clique_results/clique-graphe_phi-vs-s_n-traj={nb_trajectories}_N={N}_D={nb_colonies}_figdata.csv')



    with open(f'clique_results/clique-graphe_phi-vs-s_n-traj={nb_trajectories}_N={N}_D={nb_colonies}_parameters.json', "w") as outfile:
        json.dump(simulation_parameters, outfile, indent=4)