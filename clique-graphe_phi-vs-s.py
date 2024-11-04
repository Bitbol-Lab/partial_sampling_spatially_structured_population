## Varying s

import matplotlib as mpl
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import networkx as nx

from time import sleep
from tqdm import tqdm

import json

from numba import njit, jit


# useful functions

@jit
def simulate_clique(N, M, nb_colonies, migration_rate, s, tmax):
    assert 1 - (nb_colonies-1)*migration_rate >= 0


    b = True
    t = 1


    i_nodes = np.zeros(nb_colonies, dtype=int) # list of the number of mutants in each node
    i_nodes[0] = 1
    # i_nodes[np.random.choice(nb_colonies)] = 1 # for a random starting mutant
    N_nodes = N * np.ones(nb_colonies, dtype=int) # list of the population size in each node
    M_nodes = M * np.ones(nb_colonies, dtype=int) # list of the update size in each node


    # creating a directed graph
    DG = nx.DiGraph()
    DG.add_nodes_from(list(range(nb_colonies)))


    #adding weighted edges
    for node1 in DG.nodes:
        for node2 in DG.nodes:
            if node1==node2:
                
                weight = 1 - (nb_colonies-1)*migration_rate
            else:
                weight = migration_rate
            DG.add_weighted_edges_from([(node1, node2, weight)])


    trajectories = np.zeros((tmax,nb_colonies))
    trajectories[0,:] = i_nodes

    while t<tmax and b :
        #choose randomly one node
        selected_node = rd.randint(0, nb_colonies-1)

        #perform hypergeometrical sampling
        ngood = i_nodes[selected_node]
        nbad = N_nodes[selected_node] - ngood
        nb_mutants_before_update = np.random.hypergeometric(ngood, nbad, M_nodes[selected_node])

        # perform binomial sampling 
        x_tilde = sum([DG.edges[node, selected_node]['weight']*i_nodes[node]/N_nodes[node]  for node in list(DG.successors(selected_node))])
        prob = x_tilde * (1+s) / (1 + x_tilde*s)
        n_trials = M_nodes[selected_node]
        nb_mutants_after_update = np.random.binomial(n_trials, prob)

        # update nb of mutants in the node
        i_nodes[selected_node] = ngood - nb_mutants_before_update + nb_mutants_after_update

        trajectories[t,:] = i_nodes

        t += 1
        b = sum(i_nodes) < sum(N_nodes) and (i_nodes > 0).any()

    fixation = sum(i_nodes) == sum(N_nodes)

    if t<tmax:
            
        for tt in range(t,tmax):
            trajectories[tt,:] = trajectories[t-1,:]

    return trajectories, fixation




@jit
def simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories=100):
    all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))

    fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    count_fixation = 0


    for trajectory_index in tqdm(range(nb_trajectories)):
        #print('trajectory:', trajectory_index)
        trajectories, fixation = simulate_clique(N, M, nb_colonies, migration_rate, s, tmax)

        count_fixation += fixation

        fixation_seq[trajectory_index] = fixation

        all_trajectories[trajectory_index,:] = np.sum(trajectories, axis = 1)
        
        

    return all_trajectories, count_fixation, fixation_seq


@jit
def phi(N,s,rho,x):
    num = 1 - np.exp(-2*N*s*x / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom


# generating the graph

@jit
def run(nb_trajectories):
    N = 10
    s_range = np.logspace(-4, -1, num=10)
    tmax = 50000
    
    nb_colonies = 3
    migration_rate = 0.1


    Ms = np.array([1, N//4, N//2, 3*N//4, N])
    rhos = Ms* 1. /N

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(Ms)))

    fig, ax = plt.subplots()

    N_tot = N*nb_colonies



    for i,M in enumerate(Ms):
        print('M:',M)
        fixation_freqs = []
        fixation_err = []
        color = colors[i]
        for s in s_range:
            print('s:',s)
            _, count_fixation, fixation_seq = simulate_multiple_trajectories_clique(N, M, nb_colonies, migration_rate, s, tmax, nb_trajectories)
            fixation_freq = count_fixation / nb_trajectories
            fixation_freqs.append(fixation_freq)
            std = np.sqrt(fixation_freq * (1-fixation_freq) / nb_trajectories)
            fixation_err.append(2* std)
        ax.errorbar(s_range, fixation_freqs, yerr= fixation_err, fmt = 'o', label = f"M={M} (update fraction: {round(M/N,2)} )", alpha=0.5, color=color)
        #ax.plot(s_range, [phi(N_tot,s,M/N,1/N_tot) for s in s_range], label = f"M={M} (update fraction: {round(M/N,2)} )", color= color)



    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    plt.savefig(f'clique_results/clique-graphe_phi-vs-s_n-traj={nb_trajectories}.png')



    simulation_parameters = {
        'N_tot': N_tot,
        'N':N,
        'number of colonies':nb_colonies,
        'migration rate': migration_rate,
        'tmax':tmax,
        'nb_trajectories':nb_trajectories,
        's_range':(min(s_range), max(s_range))
    }
    return simulation_parameters


if __name__ == "__main__":
    #nb_trajectories=10**7
    nb_trajectories = 100

    simulation_parameters = run(nb_trajectories)


    with open(f'clique_results/clique-graphe_phi-vs-s_n-traj={nb_trajectories}_parameters.json', "w") as outfile:
        json.dump(simulation_parameters, outfile, indent=4)