import numpy as np


def phi(N,s,rho,x_initial):
    num = 1 - np.exp(-2*N*s*x_initial / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom
