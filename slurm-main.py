import json

import numpy as np

import sys

import time

from utils.graph_simulation import sweep_s_graph
from utils.graph_generation import generate_clique_graph, generate_cycle_graph, generate_star_graph, generate_line_graph
from utils.wm_sim import sweep_s_wm_sim
from utils.wm_mat import sweep_s_wm_mat
from utils.misc import store_output

prefix = 'simC'
results_dir = 'results/simC/'
tmax = 10000000
num = 50

graph_types = ['star', 'cycle', 'clique', 'line']

symmetric_graph_types = ['cycle', 'clique']

sim_types = ['star', 'cycle', 'clique', 'line', 'wm_sim']

all_types = ['star', 'cycle', 'clique', 'line', 'wm_sim', 'wm_mat']

STORE_FIXATION_TIMES = False



if __name__ == "__main__":
    
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

    elif type == 'star' or type == 'line':
        migration_rate = float(sys.argv[8])
        nb_demes = int(sys.argv[9])
        alpha = float(sys.argv[10])
        initial_node = sys.argv[11]
        if initial_node != 'avg': #simulation will be run with a specified initial node
            initial_node = int(initial_node)

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

    if type=='line' and initial_node=='avg':
        assert not STORE_FIXATION_TIMES     #storing fixations times is not supported in this case (would not work with the script generating histograms)
        nb_fixations_nodes = np.zeros((nb_demes, num))
        #all_extinction_times_nodes = np.zeros((nb_demes, num, nb_trajectories))
        #all_fixation_times_nodes = np.zeros((nb_demes, num, nb_trajectories))
        #all_fixation_bools_nodes = np.zeros((nb_demes, num, nb_trajectories))

        for node in range(nb_demes):
            s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_graph(
            DG, nb_demes, N, M, log_s_min, log_s_max, node, nb_trajectories, tmax, num)
            nb_fixations_nodes[node,:] = fixation_counts[:]
            #all_extinction_times_nodes[node,:,:] = all_extinction_times[:,:]
            #all_fixation_times_nodes[node,:,:] = all_fixation_times[:,:]
            #all_fixation_bools_nodes[node,:,:] = all_fixation_bools[:,:]
        averaged_nb_fixations = np.mean(nb_fixations_nodes, axis=0)
        

        output = store_output(
            STORE_FIXATION_TIMES, parameters, s_range, averaged_nb_fixations, all_extinction_times, all_fixation_times, all_fixation_bools)
    
    elif type=='star'and initial_node=='avg':
        assert not STORE_FIXATION_TIMES     #storing fixations times is not supported in this case (would not work with the script generating histograms)
        
        
        #center node
        s_range, fixation_counts_center, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_graph(
            DG, nb_demes, N, M, log_s_min, log_s_max, 0, nb_trajectories, tmax, num)
        
        #leaf node
        s_range, fixation_counts_leaf, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_graph(
            DG, nb_demes, N, M, log_s_min, log_s_max, 1, nb_trajectories, tmax, num)

        averaged_nb_fixations = (fixation_counts_center + (nb_demes-1)*fixation_counts_leaf)/nb_demes
        
        output = store_output(
            STORE_FIXATION_TIMES, parameters, s_range, averaged_nb_fixations, all_extinction_times, all_fixation_times, all_fixation_bools)
    


    
    elif type == 'wm_sim':
        initial_state = 1
        s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_wm_sim(
            N, M, log_s_min, log_s_max, initial_state, nb_trajectories, tmax, num)

        output = store_output(STORE_FIXATION_TIMES, parameters, s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools)


    elif type == 'wm_mat':
        s_range, fixation_probabilities = sweep_s_wm_mat(N, M, log_s_min, log_s_max, num)

        output = {
            'parameters': parameters,
            's_range': list(s_range),
            'fixation_probabilities': list(fixation_probabilities)
        }

    else:
        s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools = sweep_s_graph(
            DG, nb_demes, N, M, log_s_min, log_s_max, initial_node, nb_trajectories, tmax, num)

        output = store_output(STORE_FIXATION_TIMES, parameters, s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools)




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
