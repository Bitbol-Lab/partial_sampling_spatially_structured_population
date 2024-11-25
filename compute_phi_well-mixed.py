import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt

### Computing the standard transition matrix

### ERRORS !!!

def compute_hypergeometric_prob(N, M, i, k):
    return comb(i,k)*comb(N-i, M-k) / comb(N,M)

def compute_WF_prob(N,i,j,s):
    x = i/N
    p = x*(1+s) / (1+x*s)
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
    elif j<0:
        res = 0
    else:
        res = comb(N,j) * (p**j) * ((1-p)**(N-j))
    return res

def compute_transition_matrix(N,M,s):
    P = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            coeff = sum([compute_hypergeometric_prob(N,M,i,k) * compute_WF_prob(M,i,k+j-i,s)
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
    for i in range(power):
        res += A@A_power
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

def permutation(i:int, N:int):  # the permutation s.t. P_new[i,j] = P[s(i),s(j)]
    assert i>=0 and i<=N
    if i == 0:
        i_new = N-1
    elif i == N:
        i_new = N
    else:
        i_new = i-1
    return i_new  


def compute_fixation_probability(N,M,s):
    P = compute_transition_matrix(N,M,s)
    

    A = P[1:N, 1:N]
    B = P[1:N, [0,N]]

    inv = inverse_method(A)  # (I-A)^-1
    inv_vect = inv[0,:]
    B_vect = B[:,1]

    return np.inner(inv_vect, B_vect)


if __name__ == '__main__':

    print(compute_transition_matrix(10,3,0)<=1)


    
    
