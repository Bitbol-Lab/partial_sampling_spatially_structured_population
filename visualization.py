# imports

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import json

import os

from utils import phi


# variables

results_dir = 'results/'

# functions

def recover_data(type: str, job_number: int):
    """
    inputs: 
     -> type: str ('WM', 'WM_mat', 'Star', 'Cycle', 'Clique')
     -> job_number: int
    output:
     -> data: dict
    recovers the data file "type_[job_number]_*.json and returns it as a dictionary
    """
    datafile = None
    beginning = type + '_' + str(job_number) + '_'
    for filename in os.listdir('results'):
        root, ext = os.path.splitext(filename)
        if root.startswith(beginning) and ext == '.json':
            datafile = filename
    with open(results_dir + datafile, 'r') as file:
        data = json.load(file)
    return data

def compare_WM_mat(job_number: int):
    mat_data = recover_data('WM_mat',1)
    sim_data = recover_data('WM',1)

    assert mat_data['s_range'] == sim_data['s_range']
    s = np.array(mat_data['s_range'])
    y1 = np.array(mat_data['fixation_probability'])

    nb_trajectories = sim_data['parameters']['nb_trajectories']
    y2 = np.array(sim_data['nb_fixations'], dtype=float)/nb_trajectories
    y2_err = 2*np.sqrt(y2 * (1. - y2) / nb_trajectories)          # 2*standard deviation

    N, M = sim_data['parameters']['N'], sim_data['parameters']['M']
    assert N == mat_data['parameters']['N'] and M == mat_data['parameters']['M']
    y3 = np.array([phi(N,s_value, M/N, 1/N) for s_value in s])
    
    fig, ax = plt.subplots()

    ax.errorbar(s, y2, yerr= y2_err, fmt = 'o', alpha=0.5, label = 'Simulations')
    ax.scatter(s, y1, marker='x', alpha=1, label = 'Matrix inversions')
    ax.plot(s, y3, alpha=0.8, label = 'Diffusion approximation')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    plt.show()



if __name__ == '__main__':
    compare_WM_mat(1)