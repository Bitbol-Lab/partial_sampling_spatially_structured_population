import numpy as np

from scipy.special import comb



# useful functions



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

def sweep_s_wm_mat(N, M, log_s_min, log_s_max, num = 10):
    """
    Sweeps a logspace interval of relative fitness values [10**log_s_min, 10**log_s_max]
    Arguments:
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> log_s_min: int,
     -> log_s_max: int,
     -> (optional) num: int = 10, number of points in the interval of relative fitness values
    Returns:
     -> s_range: np.logspace(log_s_min, log_s_max, num=num), interval of s values
     -> fixation_probabilities: ndArray(num), fixation_probablilties[i] is the fixation probability for N, M, and s = s_range[i]
     """
    s_range = np.logspace(log_s_min, log_s_max, num=10)

    fixation_probabilities = np.zeros_like(s_range)
    

    for i,s in enumerate(s_range):
        fixation_probability = compute_fixation_probability(N,M,s)
        fixation_probabilities[i] = fixation_probability

    return s_range, fixation_probabilities











