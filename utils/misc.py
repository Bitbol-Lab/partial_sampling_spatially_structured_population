import numpy as np


def phi(N,s,rho,x_initial):
    num = 1 - np.exp(-2*N*s*x_initial / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom

def store_output(STORE_FIXATION_TIMES, parameters, s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools):
    num = len(s_range)
    nb_trajectories = parameters['nb_trajectories']
    if STORE_FIXATION_TIMES:
        output = {
                'parameters': parameters,
                's_range': list(s_range),
                'nb_fixations': list(fixation_counts),
                'all_extinction_times': list([[all_extinction_times[i,j] for i in range(num)] for j in range(nb_trajectories)]) ,
                'all_fixation_times': list([[all_fixation_times[i,j] for i in range(num)] for j in range(nb_trajectories)]),
                'all_fixation_bools': list([[all_fixation_bools[i,j] for i in range(num)] for j in range(nb_trajectories)])
            }
    else:
        output = {
                'parameters': parameters,
                's_range': list(s_range),
                'nb_fixations': list(fixation_counts)
            }

    return output
