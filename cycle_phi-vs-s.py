## Varying s

import matplotlib as mpl
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tqdm import tqdm

import json
import pandas as pd

from numba import njit, jit


# useful functions

def simulate_cycle(N, M, nb_colonies, migration_rate, alpha, s, tmax):
    assert 1 - (1 + alpha)*migration_rate >= 0


    b = True
    t = 1


    i_nodes = np.zeros(nb_colonies, dtype=int) # list of the number of mutants in each node
    i_nodes[0] = 1
    # i_nodes[np.random.choice(nb_colonies)] = 1 # for a random starting mutant
    N_nodes = N * np.ones(nb_colonies, dtype=int) # list of the population size in each node
    M_nodes = M * np.ones(nb_colonies, dtype=int) # list of the update size in each node


    # creating a directed graph
    DG = np.zeros((nb_colonies, nb_colonies), dtype=float)
    DG_nodes = np.arange(nb_colonies)


    #adding weighted edges for the center of the star


    for node in DG_nodes:
        next_node = (node + 1) % nb_colonies
        prev_node = (node - 1) % nb_colonies

        # next edge
        DG[node, next_node] = migration_rate

        # prev edge
        DG[node, prev_node] = alpha*migration_rate

        # loop
        DG[node, node] = 1 - (1+alpha)*migration_rate
    

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
        x_vector = np.divide(i_nodes, N_nodes, dtype=float)
        x_tilde = np.inner(x_vector, DG[:,selected_node])
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




def simulate_multiple_trajectories_cycle(N, M, nb_colonies, migration_rate, alpha, s, tmax, nb_trajectories=100):
    all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))

    fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    count_fixation = 0


    for trajectory_index in tqdm(range(nb_trajectories)):
        #print('trajectory:', trajectory_index)
        trajectories, fixation = simulate_cycle(N, M, nb_colonies, migration_rate, alpha, s, tmax)

        count_fixation += fixation

        fixation_seq[trajectory_index] = fixation

        all_trajectories[trajectory_index,:] = np.sum(trajectories, axis = 1)
        
    return all_trajectories, count_fixation, fixation_seq


def phi(N,s,rho,x):
    num = 1 - np.exp(-2*N*s*x / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom


# generating the graph

def run(nb_trajectories):
    

    N = 20
    s_range = np.logspace(-4, -1, num=10)
    tmax = 10000
    
    nb_colonies = 5
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
            
            _, count_fixation, _ = simulate_multiple_trajectories_cycle(N, M, nb_colonies, migration_rate, alpha, s, tmax, nb_trajectories)
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
    plt.savefig(f'cycle_results/cycle_phi-vs-s_n-traj={nb_trajectories}.png')



    simulation_parameters = {
        'N_tot': N_tot,
        'N':N,
        'M':M,
        'number of demes':nb_colonies,
        'type_graph': 'cycle',
        'migration rate': migration_rate,
        'alpha_range': (min(alphas), max(alphas)),
        'tmax':tmax,
        'nb_trajectories':nb_trajectories,
        's_range':(min(s_range), max(s_range))
    }
    return simulation_parameters, fig_data


if __name__ == "__main__":
    #nb_trajectories=10**7
    nb_trajectories = 300

    simulation_parameters, fig_data = run(nb_trajectories)

    df = pd.DataFrame({
        'alpha': fig_data[0,:],
        's': fig_data[1,:],
        'fixation_freq': fig_data[2,:],
        'fixation_err': fig_data[3,:],
        'count_fixation': fig_data[4,:]
    })

    df.to_csv(f'cycle_results/cycle_phi-vs-s_n-traj={nb_trajectories}_figdata.csv')


    with open(f'cycle_results/cycle_phi-vs-s_n-traj={nb_trajectories}_parameters.json', "w") as outfile:
        json.dump(simulation_parameters, outfile, indent=4)