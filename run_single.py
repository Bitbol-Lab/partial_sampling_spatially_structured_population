import json

import sys

import time

from utils.graph_simulation import simulate_multiple_trajectories_graph
from utils.graph_generation import generate_clique_graph, generate_cycle_graph, generate_star_graph, generate_line_graph
from utils.wm_sim import simulate_multiple_trajectories
from utils.wm_mat import compute_fixation_probability

graph_types = ['star', 'cycle', 'clique', 'line']

sim_types = ['star', 'cycle', 'clique', 'line', 'wm_sim']

all_types = ['star', 'cycle', 'clique', 'line', 'wm_sim', 'wm_mat']

STORE_FIXATION_TIMES = False

if __name__ == "__main__":
    prefix = 'A'
    results_dir = 'results/singletest/'
    tmax = 1000000
    initial_node = 0    # will be changed if type == 'star'

    type = sys.argv[1]

    job_array_nb = int(sys.argv[2])
    N = int(sys.argv[3])
    M = int(sys.argv[4])
    s = float(sys.argv[5])

    parameters = {
            'type': type,
            'job_array_nb':job_array_nb,
            'N':N,
            'M':M,
            's':s
    }

    if type in sim_types:
        nb_trajectories = int(sys.argv[6])
        parameters['nb_trajectories'] = nb_trajectories

    ##### Parameters for each type of simulation

    if type == 'clique':
        migration_rate = float(sys.argv[7])
        nb_demes = int(sys.argv[8])

        DG = generate_clique_graph(nb_demes, migration_rate)

        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes

    elif type == 'cycle':
        migration_rate = float(sys.argv[7])
        nb_demes = int(sys.argv[8])
        alpha = float(sys.argv[9])

        DG = generate_cycle_graph(nb_demes, migration_rate, alpha)

        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes
        parameters['alpha'] = alpha

    elif type == 'star' or type == 'line':
        migration_rate = float(sys.argv[7])
        nb_demes = int(sys.argv[8])
        alpha = float(sys.argv[9])
        initial_node = int(sys.argv[10])

        if type == 'star':
            DG = generate_star_graph(nb_demes, migration_rate, alpha)
        else: ## for lines
            DG = generate_line_graph(nb_demes, migration_rate, alpha)
        
        parameters['migration_rate'] = migration_rate
        parameters['nb_demes'] = nb_demes
        parameters['alpha'] = alpha
        parameters['initial_node'] = initial_node
    




    start_time = time.time()


    ### Simulations or computations for each type

    if type in graph_types:
        count_fixation, fixation_seq, fixation_times, extinction_times = simulate_multiple_trajectories_graph(
            DG, nb_demes, N, M, s, tmax, initial_node, nb_trajectories)


        if STORE_FIXATION_TIMES:
            output = {
                'parameters': parameters,
                'nb_fixations': count_fixation,
                'extinction_times': list(extinction_times) ,
                'fixation_times': list(fixation_times),
                'fixation_seq': list(fixation_seq)
            }
        else:
            output = {
                'parameters': parameters,
                'nb_fixations': count_fixation
            }

    elif type == 'wm_sim':
        initial_state = 1
        count_fixation, fixation_seq, fixation_times, extinction_times = simulate_multiple_trajectories(
            N, M, s, tmax, nb_trajectories, initial_state=1)
        
        if STORE_FIXATION_TIMES:
            output = {
                'parameters': parameters,
                'nb_fixations': count_fixation,
                'extinction_times': list(extinction_times) ,
                'fixation_times': list(fixation_times),
                'fixation_seq': list(fixation_seq)
            }
        else:
            output = {
                'parameters': parameters,
                'nb_fixations': count_fixation
            }


    elif type == 'wm_mat':
        fixation_probability = compute_fixation_probability(N,M,s)
        output = {
            'parameters': parameters,
            'fixation_probability': fixation_probability
        }


    end_time = time.time()
    execution_time = end_time - start_time

    print('Execution time:', execution_time)

    

    filename = results_dir + f'{prefix}_single_{type}_{job_array_nb}_{N}_{M}_{s}'

    if type in sim_types:
        filename += f'_{nb_trajectories}'
    
    if type in graph_types:
        filename += f'_{migration_rate}_{nb_demes}'
        if type == 'cycle' or type == 'star':
            filename += f'_{alpha}'

    filename += '.json'

    with open(filename, "w") as outfile:
        json.dump(output, outfile, indent=4)
