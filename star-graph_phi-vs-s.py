## Varying s

import matplotlib as mpl
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tqdm import tqdm

import pandas as pd

import json

from numba import njit, prange, jit, int_, float_

import time

# useful functions
@jit(int_(int_, int_, int_, float_, float_,int_, float_, int_ ))
def simulate_star(N, M, nb_colonies, migration_rate, alpha, initial_node, s, tmax):
    assert 1 - (nb_colonies-1)*alpha*migration_rate >= 0
    assert 1 - migration_rate >= 0


    b = True
    t = 1


    i_nodes = np.zeros(nb_colonies, dtype=int_) # list of the number of mutants in each node
    i_nodes[initial_node] = 1
    # i_nodes[np.random.choice(nb_colonies)] = 1 # for a random starting mutant
    #N_nodes = N * np.ones(nb_colonies, dtype=int) # list of the population size in each node
    #M_nodes = M * np.ones(nb_colonies, dtype=int) # list of the update size in each node



    # creating a directed graph
    DG = np.zeros((nb_colonies, nb_colonies), dtype=float)
    #DG_nodes = np.arange(nb_colonies)


    #adding weighted edges for the center of the star


    for node in range(1,nb_colonies):
        # outward edges
        DG[0, node]= migration_rate

        # inward edges
        DG[node, 0] = alpha*migration_rate

        # loops
        DG[node, node] = 1 - migration_rate
    
    DG[0, 0] = 1 -(nb_colonies - 1)*alpha*migration_rate



    #trajectories = np.zeros((tmax,nb_colonies))
    #trajectories[0,:] = i_nodes

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

    #if t<tmax:
            
        #for tt in range(t,tmax):
            #trajectories[tt,:] = trajectories[t-1,:]

    return fixation



@njit(parallel=True)
def simulate_multiple_trajectories_star(N, M, nb_colonies, migration_rate, alpha, s, tmax, nb_trajectories=100):
    #all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))

    #fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    count_fixation = 0


    for trajectory_index in prange(nb_trajectories):
        #print('trajectory:', trajectory_index)
        initial_node = np.random.randint(nb_colonies)
        fixation = simulate_star(N, M, nb_colonies, migration_rate, alpha, initial_node, s, tmax)

        count_fixation += fixation

        #fixation_seq[trajectory_index] = fixation

        #all_trajectories[trajectory_index,:] = np.sum(trajectories, axis = 1)
        
        

    return count_fixation


def phi(N,s,rho,x):
    num = 1 - np.exp(-2*N*s*x / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom


# generating the graph

def run(nb_trajectories, N, M, nb_colonies):
    

    s_range = np.logspace(-4, -1, num=10)
    tmax = 10000
    
    migration_rate = 0.01
    alphas = np.logspace(-1, 1, num=5)


    #Ms = np.array([1, N//4, N//2, 3*N//4, N])
    #rhos = Ms* 1. /N

    M = N # WF case

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(alphas)))

    fig, ax = plt.subplots()

    N_tot = N*nb_colonies

    fig_data = np.zeros((5, len(alphas)*len(s_range)))


    for i,alpha in enumerate(alphas):
        print('alpha:',alpha)

        color = colors[i]

        fig_data[0, i*len(s_range):(i+1)*len(s_range)] = alpha*np.ones(len(s_range))

        for j,s in enumerate(s_range):
            print('s:',s)
            
            count_fixation = simulate_multiple_trajectories_star(N, M, nb_colonies, migration_rate, alpha, s, tmax, nb_trajectories)
            fixation_freq = count_fixation / nb_trajectories
            std = np.sqrt(fixation_freq * (1-fixation_freq) / nb_trajectories)

            fig_data[1, i*len(s_range) + j] = s
            fig_data[2, i*len(s_range) + j] = fixation_freq
            fig_data[3, i*len(s_range) + j] = 2*std
            fig_data[4, i*len(s_range) + j] = count_fixation
        ax.errorbar(s_range, fig_data[2,i*len(s_range):(i+1)*len(s_range)], yerr= fig_data[3,i*len(s_range):(i+1)*len(s_range)], fmt = 'o', label = f"migration assymmetry={alpha}", alpha=0.5, color=color)
        #ax.plot(s_range, [phi(N_tot,s,M/N,1/N_tot) for s in s_range], label = f"M={M} (update fraction: {round(M/N,2)} )", color= color)



    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    plt.savefig(f'star_results/star-graph_phi-vs-s_n-traj={nb_trajectories}_N={N}_M={M}_D={nb_colonies}.png')



    simulation_parameters = {
        'N_tot': N_tot,
        'N':N,
        'M':M,
        'number of demes':nb_colonies,
        'type_graph': 'star',
        'migration rate': migration_rate,
        'alpha_range': (min(alphas), max(alphas)),
        'initial_node': 'uniformly random',
        'tmax':tmax,
        'nb_trajectories':nb_trajectories,
        's_range':(min(s_range), max(s_range))
    }
    return simulation_parameters, fig_data


if __name__ == "__main__":
    #nb_trajectories=10**7
    nb_trajectories = 10000

    N = 10
    M = N #WF Case
    nb_colonies = 5

    start_time = time.time()
    simulation_parameters, fig_data = run(nb_trajectories,N,M,nb_colonies)
    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time)

    df = pd.DataFrame({
        'alpha': fig_data[0,:],
        's': fig_data[1,:],
        'fixation_freq': fig_data[2,:],
        'fixation_err': fig_data[3,:],
        'count_fixation': fig_data[4,:]
    })

    df.to_csv(f'star_results/star-graph_phi-vs-s_n-traj={nb_trajectories}_figdata.csv')


    with open(f'star_results/star-graph_phi-vs-s_n-traj={nb_trajectories}_parameters.json', "w") as outfile:
        json.dump(simulation_parameters, outfile, indent=4)