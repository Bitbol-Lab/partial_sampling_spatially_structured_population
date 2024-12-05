import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt

import time

from utils import phi

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


### Estimating (I-A)^(-1)

def infty_operator_norm(A):
    u = np.sum(np.abs(A), axis=1)
    return max(u)

#A = np.array([[0,1],
#              [1,1]])

#print(infty_operator_norm(A))

def estimate_power(epsilon, A): #should this work ????
    A_norm = infty_operator_norm(A)
    assert A_norm != 1
    res = np.log(epsilon * (1-A_norm)) / np.log(A_norm)
    return res

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


def plot(N, M, log_s_min=-4, log_s_max=-1, num=10):
    fig, ax = plt.subplots()

    x = np.logspace(log_s_min, log_s_max, num)
    y = np.zeros_like(x)
    y_th = np.zeros_like(x)
    for i,s in enumerate(x):
        print('s:',s)
        y[i] = compute_fixation_probability(N,M,s)
        y_th[i] = phi(N,s,M/N,1/N)
    ax.scatter(x, y, label='Matrix computation')
    ax.plot(x,y_th,'k--', label='theory in diffusion approximation')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Relative fitness')
    ax.set_ylabel('Fixation probability')
    ax.legend()
    plt.show()



if __name__ == '__main__':
    N,M,s = 50,3,0.3
    P = compute_transition_matrix(N,M,s)
    I = np.eye(N-1)
    
    #print("stochasticity: ",(P>=0).all(),np.sum(P, axis=1))   # check stochasticity

    start_time = time.time()
    fixation_probabilty = compute_fixation_probability(N,M,s)
    end_time = time.time()
    execution_time = end_time - start_time

    print('fixation probability: ', fixation_probabilty)
    print('execution_time: ', execution_time)   #around 5s for N=100

    plot(N,M, -6, -1, 20)


    
    
