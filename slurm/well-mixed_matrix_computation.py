import sys

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import time

import json

from numba import njit, jit, prange, int_, float_

from scipy.special import comb


### Computing the standard transition matrix

def compute_hypergeometric_prob(N, M, i, k):
    return comb(i,k)*comb(N-i, M-k) / comb(N,M)

def compute_binomial_prob(N,j,p):
    if p==0:
        if j==0:
            res = 1
        else:
            res = 0
    elif p==1:
        if j==N:
            res = 1
        else:
            res=0
    elif j<0 or j>N:
        res = 0
    else:
        res = comb(N,j) * (p**j) * ((1-p)**(N-j))
    return res

def compute_transition_matrix(N,M,s):
    P = np.zeros((N+1,N+1))
    for i in range(N+1):
        x = i/N
        p = x*(1+s)/(1+x*s)
        for j in range(N+1):
            
            coeff = sum([compute_hypergeometric_prob(N,M,i,k) * compute_binomial_prob(M,k+j-i,p)
                         for k in range(i-j,i+1)])
            P[i,j] = coeff
    return P


def truncated_sum_method(A, power=10):
    dim = A.shape[0]
    res = np.eye(dim)
    A_power = np.eye(dim)
    for _ in range(power):
        A_power = A_power@A
        res += A_power
    return res

def inverse_method(A):
    n = A.shape[0]
    I = np.eye(n)
    if np.linalg.det(I-A) != 0:
        return np.linalg.inv(I - A)
    else:
        print('I-A not invertible')
        return truncated_sum_method(A)
    
### computing fixation probability



def compute_fixation_probability(N,M,s):
    P = compute_transition_matrix(N,M,s)
    

    A = P[1:N, 1:N]
    B = P[1:N, [0,N]]

    inv = inverse_method(A)  # (I-A)^-1
    inv_vect = inv[0,:]
    B_vect = B[:,1]

    return np.inner(inv_vect, B_vect)

def run(N, M, log_s_min, log_s_max):
    s_range = np.logspace(log_s_min, log_s_max, num=10)

    data = np.zeros_like(s_range)
    

    for i,s in enumerate(s_range):
        fixation_probability = compute_fixation_probability(N,M,s)
        data[i] = fixation_probability

    return s_range, data


    

if __name__ == "__main__":
    job_array_nb = int(sys.argv[1])
    N = int(sys.argv[2])
    M = int(sys.argv[3])
    log_s_min = int(sys.argv[4])
    
    log_s_max = int(sys.argv[5])


    start_time = time.time()

    s_range, data = run(N, M, log_s_min, log_s_max)

    end_time = time.time()
    execution_time = end_time - start_time

    parameters = {
        'job_array_nb':job_array_nb,
        'type': 'well-mixed - matrix computations',
        'N':N,
        'M':M,
        'log_s_min':log_s_min,
        'log_s_max':log_s_max,
    }

    output = {
        'parameters': parameters,
        's_range': list(s_range),
        'fixation_probability': list(data)
    }

    print('Execution time:', execution_time)

    filename = f'slurm/results/WM_mat_{job_array_nb}_{N}_{M}_{log_s_min}_{log_s_max}.json'
    with open(filename, "w") as outfile:
        json.dump(output, outfile, indent=4)
