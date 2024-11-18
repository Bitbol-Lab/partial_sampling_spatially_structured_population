import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt


@njit
def hypergeo_draw():
    res = np.random.hypergeometric(10, 90, 10)
    return res

@njit(parallel = True)
def draw_multiple(N):
    X = []
    for i in prange(N):
        x = hypergeo_draw()
        X.append(x)
    return X

X = draw_multiple(1000)
plt.hist(X)
plt.show()