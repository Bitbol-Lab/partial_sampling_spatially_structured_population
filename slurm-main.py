import json

import sys

import time

from utils.graph_simulation import sweep_s_graph
from utils.graph_generation import generate_clique_graph, generate_cycle_graph, generate_star_graph
from utils.wm_sim import sweep_s_wm_sim
from utils.wm_mat import sweep_s_wm_mat

graph_types = ['star', 'cycle', 'clique', 'line']

sim_types = ['star', 'cycle', 'clique', 'line', 'wm_sim']

all_types = ['star', 'cycle', 'clique', 'line', 'wm_sim', 'wm_mat']

if __name__ == "__main__":
    prefix = 'expA'
    results_dir = 'results/expA/'
    tmax = 1000000
    num = 10
    initial_node = 0    # will be changed if type == 'star'

    type = sys.argv[1]

    job_array_nb = int(sys.argv[2])
    N = int(sys.argv[3])
    M = int(sys.argv[4])
    log_s_min = int(sys.argv[5])
    log_s_max = int(sys.argv[6])

    parameters = {
            'type': type,
            'job_array_nb':job_array_nb,
            'N':N,
            'M':M,
            'log_s_min':log_s_min,
            'log_s_max':log_s_max,
    }

    if type in sim_types:
        nb_trajectories = int(sys.argv[7])
        parameters['nb_trajectories'] = nb_trajectories

    ##### Parameters for each type of simulation

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
    

    elif type == 'line':
        # TODO
        x = 0


    
            


    

    start_time = time.time()


    ### Simulations or computations for each type

    if type in graph_types:
        s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_graph(
            DG, nb_demes, N, M, log_s_min, log_s_max, initial_node, nb_trajectories, tmax, num)



        output = {
            'parameters': parameters,
            's_range': list(s_range),
            'nb_fixations': list(fixation_counts),
            'all_extinction_times': list([[all_extinction_times[i,j] for i in range(num)] for j in range(nb_trajectories)]) ,
            'all_fixation_times': list([[all_fixation_times[i,j] for i in range(num)] for j in range(nb_trajectories)]),
            'all_fixation_bools': list([[all_fixation_bools[i,j] for i in range(num)] for j in range(nb_trajectories)])
        }

    elif type == 'wm_sim':
        initial_state = 1
        s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_wm_sim(
            N, M, log_s_min, log_s_max, initial_state, nb_trajectories, tmax, num)

        output = {
            'parameters': parameters,
            's_range': list(s_range),
            'nb_fixations': list(fixation_counts),
            'all_extinction_times': list([[all_extinction_times[i,j] for i in range(num)] for j in range(nb_trajectories)]) ,
            'all_fixation_times': list([[all_fixation_times[i,j] for i in range(num)] for j in range(nb_trajectories)]),
            'all_fixation_bools': list([[all_fixation_bools[i,j] for i in range(num)] for j in range(nb_trajectories)])
        }

    elif type == 'wm_mat':
        s_range, fixation_probabilities = sweep_s_wm_mat(N, M, log_s_min, log_s_max, num)

        output = {
            'parameters': parameters,
            's_range': list(s_range),
            'fixation_probabilities': list(fixation_probabilities)
        }


    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time)

    

    filename = results_dir + f'{prefix}_{type}_{job_array_nb}_{N}_{M}_{log_s_min}_{log_s_max}'

    if type in sim_types:
        filename += f'_{nb_trajectories}'
    
    if type in graph_types:
        filename += f'_{migration_rate}_{nb_demes}'
        if type == 'cycle' or type == 'star':
            filename += f'_{alpha}'

    filename += '.json'

    with open(filename, "w") as outfile:
        json.dump(output, outfile, indent=4)
