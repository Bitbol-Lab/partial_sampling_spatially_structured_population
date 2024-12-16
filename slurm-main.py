import json

import sys

import time

from utils.graph_simulation import sweep_s_graph
from utils.graph_generation import generate_clique_graph, generate_cycle_graph, generate_star_graph



if __name__ == "__main__":
    prefix = 'test'
    results_dir = 'results/'
    tmax = 100000
    num = 10
    initial_node = 0 # changed if type == 'star'

    type = sys.argv[1]

    job_array_nb = int(sys.argv[2])
    N = int(sys.argv[3])
    M = int(sys.argv[4])
    log_s_min = int(sys.argv[5])
    log_s_max = int(sys.argv[6])
    nb_trajectories = int(sys.argv[7])
    parameters = {
        'type': type,
        'job_array_nb':job_array_nb,
        'N':N,
        'M':M,
        'log_s_min':log_s_min,
        'log_s_max':log_s_max,
        'nb_trajectories':nb_trajectories
    }

    if type == 'clique':
        migration_rate = float(sys.argv[8])
        nb_demes = int(sys.argv[9])

        DG = generate_clique_graph(nb_demes, migration_rate)

        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes

    elif type == 'cycle':
        migration_rate = float(sys.argv[8])
        nb_demes = int(sys.argv[9])
        alpha = float(sys.argv[10])

        DG = generate_cycle_graph(nb_demes, migration_rate, alpha)

        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes
        parameters['alpha'] = alpha

    elif type == 'star':
        migration_rate = float(sys.argv[8])
        nb_demes = int(sys.argv[9])
        alpha = float(sys.argv[10])
        initial_node = int(sys.argv[11])

        DG = generate_star_graph(nb_demes, migration_rate, alpha)

        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes
        parameters['alpha'] = alpha
        parameters['initial_node'] = initial_node


    
            


    

    start_time = time.time()

    

    s_range, fixation_counts, all_extinction_times, all_fixation_times = sweep_s_graph(
        DG, nb_demes, N, M, log_s_min, log_s_max, initial_node, nb_trajectories, tmax, num)

    end_time = time.time()
    execution_time = end_time - start_time


    output = {
        'parameters': parameters,
        's_range': list(s_range),
        'nb_fixations': list(fixation_counts),
        'all_extinction_times': list([[all_extinction_times[i,j] for i in range(num)] for j in range(nb_trajectories)]) ,
        'all_fixation_times': list([[all_fixation_times[i,j] for i in range(num)] for j in range(nb_trajectories)])
    }

    print('Execution time:', execution_time)

    

    filename = results_dir + f'{prefix}_{type}_{job_array_nb}_{N}_{M}_{log_s_min}_{log_s_max}_{nb_trajectories}_{migration_rate}_{nb_demes}'
    if type == 'cycle' or type == 'star':
        filename += f'_{alpha}'

    filename += '.json'

    with open(filename, "w") as outfile:
        json.dump(output, outfile, indent=4)
