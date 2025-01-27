# imports

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import json

import os

from utils.misc import phi


# variables


# functions

def recover_data(prefix: str, type: str, results_dir: str, job_number: int):
    """
    inputs: 
     -> prefix: str ('expA', 'test', ...)
     -> type: str ('wm', 'wm_mat', 'star', 'cycle', 'clique')
     -> results_dir: str (results directory)
     -> job_number: int
    output:
     -> data: dict
    recovers the data file "prefix_type_[job_number]_*.json and returns it as a dictionary
    """
    datafile = None
    beginning = prefix + '_' + type + '_' + str(job_number) + '_'
    for filename in os.listdir(results_dir):
        root, ext = os.path.splitext(filename)
        if root.startswith(beginning) and ext == '.json':
            datafile = filename
    with open(results_dir + datafile, 'r') as file:
        data = json.load(file)
    return data

def compare_WM_mat(job_number: int):
    results_dir = 'results/compare_sim-mat/'
    mat_data = recover_data('','WM_mat',results_dir, job_number)
    sim_data = recover_data('','WM',results_dir, job_number)

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


def WM_paper(min_job_number: int, max_job_number: int, results_dir : str= 'results/'):
    n_jobs = 1 + max_job_number - min_job_number
    first_data = recover_data('WM', results_dir, min_job_number)
    s_range = np.array(first_data['s_range'])
    nb_fixations = np.zeros((n_jobs, len(s_range)))
    Ms = np.zeros(n_jobs)
    Ms[0] = first_data['parameters']['M']
    N= first_data['parameters']['N']
    nb_trajectories = first_data['parameters']['nb_trajectories']

    for i in range(min_job_number+1, max_job_number + 1):
        data_i = recover_data('WM', results_dir, i)
        Ms[i - min_job_number] = data_i['parameters']['M']
        nb_fixations[i-min_job_number,:] = np.array(data_i['nb_fixations'])



    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(Ms)))

    fig, ax = plt.subplots()


    for i,M in enumerate(Ms):
        color = colors[i]
        y = nb_fixations[i,:] / nb_trajectories
        y_err = np.sqrt(y * (1-y)/nb_trajectories)
        y_th = np.array([phi(N,s,M/N, 1/N) for s in s_range])
        ax.errorbar(s_range, y, yerr= y_err, fmt = 'o', alpha=0.5, color=color)
        ax.plot(s_range, y_th, label = f"M={round(M)} (update fraction: {round(M/N,2)} )", color= color)


    


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    fig.savefig(results_dir + 'WM_paper_1')


def expA(prefix: str, min_job_number: int, max_job_number: int,type: str,  results_dir : str= 'results/expA/'):
    """
    Plots with y=fixation probability, x=s, different lines for different sampling fractions
    """
    n_jobs = 1 + max_job_number - min_job_number
    first_data = recover_data(prefix,type, results_dir, min_job_number)
    s_range = np.array(first_data['s_range'])

    metadata = first_data['parameters'].copy()


    if type == 'wm_mat':
        fixation_probabilities = np.zeros((n_jobs, len(s_range)))
        Ms = np.zeros(n_jobs)
        Ms[0] = first_data['parameters']['M']
        fixation_probabilities[0,:] = np.array(first_data['fixation_probabilities'])
        N= first_data['parameters']['N']
        
        for i in range(min_job_number+1, max_job_number + 1):
            data_i = recover_data(prefix,type, results_dir, i)
            Ms[i - min_job_number] = data_i['parameters']['M']
            fixation_probabilities[i-min_job_number,:] = np.array(data_i['fixation_probabilities'])

    else:
        fixation_probabilities = np.zeros((n_jobs, len(s_range)))
        Ms = np.zeros(n_jobs)
        Ms[0] = first_data['parameters']['M']
        nb_trajectories = first_data['parameters']['nb_trajectories']

        fixation_probabilities[0,:] = np.array(first_data['nb_fixations'])/nb_trajectories
        N= first_data['parameters']['N']
        
        for i in range(min_job_number+1, max_job_number + 1):
            data_i = recover_data(prefix,type, results_dir, i)
            Ms[i - min_job_number] = data_i['parameters']['M']
            fixation_probabilities[i-min_job_number,:] = np.array(data_i['nb_fixations'])/nb_trajectories

    metadata['M'] = list(Ms)

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(Ms)))

    fig, ax = plt.subplots()


    for i,M in enumerate(Ms):
        color = colors[i]
        y = fixation_probabilities[i,:]
        y_err = np.sqrt(y * (1-y)/nb_trajectories)
        #y_th = np.array([phi(N,s,M/N, 1/N) for s in s_range])
        ax.errorbar(s_range, y, yerr=y_err, fmt = 'o', alpha=0.5,label = f"M={round(M)} (update fraction: {round(M/N,2)})", color=color)
        #ax.plot(s_range, y_th, label = f"M={round(M)} (update fraction: {round(M/N,2)} )", color= color)


    


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability (' + type +')')
    ax.legend()

    fig_name = results_dir + prefix + '_' + type
    fig.savefig(fig_name)

    filename = fig_name + '_metadata.json'

    with open(filename, "w") as outfile:
        json.dump(metadata, outfile, indent=4)




def time_histograms(results_dir, prefix, type, job_number):
    n_bins = 20
    
    data = recover_data(prefix, type, results_dir, job_number)
    metadata = data['parameters'].copy()
    s_range = np.array(data['s_range'])
    num = len(s_range)
    nb_trajectories = data['parameters']['nb_trajectories']
    s_index = num-1
    metadata['s'] = s_range[s_index]
    
    all_fixation_bools = np.array(data['all_fixation_bools'])[:, s_index]
    all_extinction_times = np.array(data['all_extinction_times'])[:, s_index]
    all_fixation_times = np.array(data['all_fixation_times'])[:, s_index]

    extinction_times = all_extinction_times[all_fixation_bools == 0.]

    fixation_times = all_fixation_times[all_fixation_bools == 1.]

    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    dist1 = extinction_times
    dist2 = fixation_times
    
    axs[0].hist(dist1, bins=n_bins)
    axs[0].set_xlabel('Extinction time')

    axs[1].hist(dist2, bins=n_bins)
    axs[1].set_xlabel('Fixation time')

    fig_name = results_dir + 'histogram_' + prefix + '_' + type

    plt.savefig(fig_name)
    
    
    filename = fig_name + '_metadata.json'

    with open(filename, "w") as outfile:
        json.dump(metadata, outfile, indent=4)

    


    
        




if __name__ == '__main__':
    #compare_WM_mat(1)
    #WM_paper(1, 5, 'results/WM_paper/')
    for type in ['clique','cycle','line','star']:
        expA('expA',1,5,type, 'results/expA_graph_e7_runs/')
    time_histograms('results/expA_graph_e7_runs/', 'expA', 'clique',5)