## Varying s
import sys
import numpy as np

import time

import json

from numba import njit, jit, prange, int_, float_


# useful functions

@jit(int_(int_, int_, int_, float_, float_,int_, float_, int_ ))
def simulate_star(N, M, nb_colonies, migration_rate, alpha, initial_node, s, tmax):
    


    b = True
    t = 1


    i_nodes = np.zeros(nb_colonies, dtype=int_) # list of the number of mutants in each node
    i_nodes[initial_node] = 1
    # i_nodes[np.random.choice(nb_colonies)] = 1 # for a random starting mutant
    #N_nodes = N * np.ones(nb_colonies, dtype=int) # list of the population size in each node
    #M_nodes = M * np.ones(nb_colonies, dtype=int) # list of the update size in each node



    # creating a directed graph
    assert 1 - (nb_colonies-1)*alpha*migration_rate >= 0
    assert 1 - migration_rate >= 0
    DG = np.zeros((nb_colonies, nb_colonies), dtype=float)


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


def run(N, M, log_s_min, log_s_max, nb_trajectories, migration_rate, nb_colonies, alpha):
    s_range = np.logspace(log_s_min, log_s_max, num=10)
    tmax = 100000

    data = np.zeros_like(s_range)
    

    for i,s in enumerate(s_range):
        s = float_(s)
        count_fixation = simulate_multiple_trajectories_star(N, M, nb_colonies, migration_rate, alpha, s, tmax, nb_trajectories)
        data[i] = count_fixation

    return s_range, data


if __name__ == "__main__":
    job_array_nb = int(sys.argv[1])
    N = int(sys.argv[2])
    M = int(sys.argv[3])
    log_s_min = int(sys.argv[4])
    log_s_max = int(sys.argv[5])
    nb_trajectories = int(sys.argv[6])
    migration_rate = float(sys.argv[7])
    nb_colonies = int(sys.argv[8])
    alpha = float(sys.argv[9])


    start_time = time.time()

    s_range, data = run(N, M, log_s_min, log_s_max, nb_trajectories, migration_rate, nb_colonies, alpha)

    end_time = time.time()
    execution_time = end_time - start_time

    parameters = {
        'job_array_nb':job_array_nb,
        'type': 'star graph',
        'N':N,
        'M':M,
        'log_s_min':log_s_min,
        'log_s_max':log_s_max,
        'nb_trajectories':nb_trajectories,
        'migration_rate':migration_rate,
        'nb_colonies':nb_colonies,
        'alpha': alpha
    }

    output = {
        'parameters': parameters,
        's_range': list(s_range),
        'nb_fixations': list(data)
    }

    print('Execution time:', execution_time)

    filename = f'results/Star_{job_array_nb}_{N}_{M}_{log_s_min}_{log_s_max}_{nb_trajectories}_{migration_rate}_{nb_colonies}_{alpha}.json'
    with open(filename, "w") as outfile:
        json.dump(output, outfile, indent=4)
