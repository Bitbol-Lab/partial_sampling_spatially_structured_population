## Varying s

import matplotlib as mpl
import numpy as np
import random as rd
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

import json

from numba import njit, jit


# useful functions

@jit(nopython=True)
def simulate_trajectory(N, M, s, tmax, initial_state=1):
    states = np.zeros(tmax)
    states[0] = initial_state
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
        current_state = current_state - nb_mutants_before_update + nb_mutants_after_update
        b = 0 < current_state and current_state < N
        t+=1
        states[t] = current_state
    
    if t<tmax:
        states[t+1:] = current_state

    fixation = current_state == N
    return states, fixation

@jit(parallel=True)
def simulate_multiple_trajectories(N, M, s, tmax, nb_trajectories=100, initial_state=1):
    all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))
    all_trajectories[:,0] = initial_state

    fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    count_fixation = 0


    #for trajectory_index in tqdm(range(nb_trajectories)):
    for trajectory_index in tqdm(range(nb_trajectories)):
        if trajectory_index%(10**3) == 0:
            #print('trajectory:', trajectory_index)
            a= 1
        states, fixation = simulate_trajectory(N,M,s,tmax,initial_state)

        count_fixation += fixation

        fixation_seq[trajectory_index] = fixation

        all_trajectories[trajectory_index,:] = states[:]
        
        

    return all_trajectories, count_fixation, fixation_seq


@njit
def phi(N,s,rho,x):
    num = 1 - np.exp(-2*N*s*x / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom




# generating the graph
@jit(parallel=True)
def run(nb_trajectories):

    N = 1000
    s_range = np.logspace(-5, -1, num=10)
    tmax = 10000
    


    Ms = np.array([1, N//4, N//2, 3*N//4, N])
    rhos = Ms* 1. /N

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(Ms)))

    fig, ax = plt.subplots()

    fig_data = np.zeros((5, len(Ms)*len(s_range)))



    for i,M in enumerate(Ms):
        print('M:',M)
        fig_data[0, i*len(s_range):(i+1)*len(s_range)] = M*np.ones(len(s_range))

        color = colors[i]
        for j,s in enumerate(s_range):
            print('s:',s)
            
            _, count_fixation, _ = simulate_multiple_trajectories(N,M,s,tmax, nb_trajectories)
            
            fixation_freq = count_fixation / nb_trajectories
            std = np.sqrt(fixation_freq * (1-fixation_freq) / nb_trajectories)
            
            fig_data[1, i*len(s_range) + j] = s
            fig_data[2, i*len(s_range) + j] = fixation_freq
            fig_data[3, i*len(s_range) + j] = 2*std
            fig_data[4, i*len(s_range) + j] = count_fixation
        ax.errorbar(s_range, fig_data[2,i*len(s_range):(i+1)*len(s_range)], yerr= fig_data[3,i*len(s_range):(i+1)*len(s_range)], fmt = 'o', alpha=0.5, color=color)
        ax.plot(s_range, [phi(N,s,M/N,1/N) for s in s_range], label = f"M={M} (update fraction: {round(M/N,2)} )", color= color)


    


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    plt.savefig(f'well-mixed_results/well-mixed_phi-vs-s_n-traj={nb_trajectories}.png')

    simulation_parameters = {
        'N':N,
        'tmax':tmax,
        'nb_trajectories':nb_trajectories,
        's_range': (min(s_range),max(s_range))
    }
    return simulation_parameters, fig_data

    

if __name__ == "__main__":

    #nb_trajectories=10**7
    nb_trajectories = 400

    simulation_parameters, fig_data = run(nb_trajectories)

    df = pd.DataFrame({
        'M': fig_data[0,:],
        's': fig_data[1,:],
        'fixation_freq': fig_data[2,:],
        'fixation_err': fig_data[3,:],
        'count_fixation': fig_data[5,:]
    })

    df.to_csv(f'well-mixed_results/well-mixed_phi-vs-s_n-traj={nb_trajectories}_figdata.csv')

    with open(f'well-mixed_results/well-mixed_phi-vs-s_n-traj={nb_trajectories}_parameters.json', "w") as outfile:
        json.dump(simulation_parameters, outfile, indent=4)
