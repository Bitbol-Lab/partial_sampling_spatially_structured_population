# imports

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import json

import os

from utils.misc import phi



# variables

clean_type = {
        'wm_sim': 'Well-mixed',
        'wm_mat': 'Well-mixed (matrix computations)',
        'cycle': 'Cycle',
        'clique': 'Clique',
        'star': 'Star',
        'line': 'Line'
    }

graph_types = ['star', 'cycle', 'clique', 'line']

symmetric_graph_types = ['cycle', 'clique']

sim_types = ['star', 'cycle', 'clique', 'line', 'wm_sim']

all_types = ['star', 'cycle', 'clique', 'line', 'wm_sim', 'wm_mat']



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
    """
    must take data generated by run_single.py
    """
    n_bins = 20
    
    data = recover_data(prefix + '_single', type, results_dir, job_number)
    metadata = data['parameters'].copy()
    nb_trajectories = data['parameters']['nb_trajectories']
    
    fixation_seq = np.array(data['fixation_seq'])
    extinction_times = np.array(data['extinction_times'])
    fixation_times = np.array(data['fixation_times'])

    extinction_times = extinction_times[fixation_seq == 0.]

    fixation_times = fixation_times[fixation_seq == 1.]

    avg_fixation_time = np.mean(fixation_times)
    avg_extinction_time = np.mean(extinction_times)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    dist1 = extinction_times
    dist2 = fixation_times

    title = clean_type[type] + f' simulations ({metadata['nb_trajectories']} runs)'

    fig.suptitle(title)
    
    textstr = '\n'.join((
    f'N = {metadata['N']}',
    f'M = {metadata['M']}',
    f's = {metadata['s']}'))

    if type in graph_types:
        textstr += '\n' + f'D = {metadata['nb_demes']}'
        textstr += '\n' + f'm = {metadata['migration_rate']}'
        if type != 'clique':
            textstr += '\n' + r'$\alpha$ = ' + f'{metadata['alpha']}'
            if type != 'cycle':
                textstr += '\n' + r'Initial node: ' + f'{metadata['initial_node']}'

    axs[0].hist(dist1, bins=n_bins, density=True)
    axs[0].axvline(avg_extinction_time, color='k', linestyle='dashed', linewidth=1)
    axs[0].set_xlabel('Extinction time')
    axs[0].set_ylabel('Probability density')


    axs[1].hist(dist2, bins=n_bins, density=True)
    axs[1].set_xlabel('Fixation time')
    axs[1].axvline(avg_fixation_time, color='k', linestyle='dashed', linewidth=1)

    # Display the mean
    axs[0].text(0.25, 0.5, f'Mean: {round(avg_extinction_time,1)}', transform=axs[0].transAxes, fontsize=10,
        verticalalignment='top')
    axs[1].text(0.25, 0.5, f'Mean: {round(avg_fixation_time,1)}', transform=axs[1].transAxes, fontsize=10,
        verticalalignment='top')

    # Display the parameters
    axs[1].text(0.5, 0.95, textstr, transform=axs[1].transAxes, fontsize=12,
        verticalalignment='top')

    fig_name = results_dir + 'histogram_' + prefix + '_' + type

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")


    plt.savefig(fig_name)
    
    
    filename = fig_name + '_metadata.json'

    with open(filename, "w") as outfile:
        json.dump(metadata, outfile, indent=4)




def simB(prefix: str, job_number: int, results_dir : str= 'results/simB/'):
    """
    Plots with y=fixation probability, x=s, different lines for different structures
    """
   

    metadata = {}

    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, 6))

    fig, ax = plt.subplots()

    ## wm_mat
    wm_mat_data = recover_data(prefix,'wm_mat', results_dir, job_number)
    s_range = np.array(wm_mat_data['s_range'])

    metadata['s_range'] = list(s_range)
    metadata['wm_mat_parameters'] = wm_mat_data['parameters'].copy()

    y_wm_mat = wm_mat_data['fixation_probabilities']
    
    ax.scatter(s_range, y_wm_mat , marker='x', alpha=1, label = 'wm_mat', color=colors[0])

    for i,type in enumerate(['wm_sim', 'clique', 'cycle', 'star', 'line']):
        type_data = recover_data(prefix, type, results_dir, job_number)
        nb_trajectories = type_data['parameters']['nb_trajectories']

        metadata[type + '_parameters'] = type_data['parameters'].copy()
        y_type = np.array(type_data['nb_fixations'])/nb_trajectories
        y_err = np.sqrt(y_type * (1-y_type)/nb_trajectories)
        ax.errorbar(s_range, y_type, yerr=y_err, fmt = 'o', alpha=0.5, label = type, color=colors[i+1])
        #y_th = np.array([phi(N,s,M/N, 1/N) for s in s_range])
        #ax.plot(s_range, y_th, label = f"M={round(M)} (update fraction: {round(M/N,2)} )", color= color)


    


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()

    fig_name = results_dir + prefix + '_' + type
    fig.savefig(fig_name)

    filename = fig_name + '_metadata.json'

    with open(filename, "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    


    
        




if __name__ == '__main__':
    #compare_WM_mat(1)
    #WM_paper(1, 5, 'results/WM_paper/')
    for type in ['clique','cycle','line','star']:
        expA('expA',1,5,type, 'results/expA_graph_e7_runs/')
    time_histograms('results/expA_graph_e7_runs/', 'expA', 'clique',5)