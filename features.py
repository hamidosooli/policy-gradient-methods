import numpy as np
from itertools import product


# Polynomial Basis
def polynomials(states, approximation_order):
    num_states = len(states)
    c = np.asarray(list(product(range(approximation_order + 1), repeat=num_states)))
    return np.prod(np.tile(states, (len(c), 1)) ** c, axis=1)


# Fourier Basis
def state_range(states, UB, LB):
    return np.divide(np.subtract(states, LB), np.subtract(UB, LB))


def fourier(states_in_range, approximation_order):
    num_states = len(states_in_range)
    # global c
    c = np.asarray(list(product(range(approximation_order + 1), repeat=num_states)))
    return np.cos(np.pi * (c @ states_in_range))


# def alpha_fourier(alpha=2e-12):
#     global c
#     denominator = np.sqrt(np.sum(c ** 2, axis=1))
#     Alpha = np.zeros_like(denominator)
#     for i, d in enumerate(denominator):
#         if d == 0.:
#             Alpha[i] = alpha
#         else:
#             Alpha[i] = alpha / d
#     return Alpha


def feature_st_act(acts, approximation_order, num_of_states, action, x_s):
    x_s_a = np.zeros((len(acts), (approximation_order + 1) ** num_of_states))
    x_s_a[acts.index(action), :] = x_s
    return x_s_a.flatten()