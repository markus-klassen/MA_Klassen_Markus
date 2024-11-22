"""
Contains some functions that are used in several scripts.
"""

import numpy as np
import torch
import pickle

from torch.distributions import Geometric
from torch.distributions import Binomial
from torch.distributions import Normal

import pypesto
import pypesto.sample as sample
import pypesto.optimize as optimize


def get_base():

    """
    Initialise basic set up, i.e. load data and number of samples.
    """

    # load data
    with open(rf'data.pkl', 'rb') as f:
        data = pickle.load(f)

    L = len(data)

    return data, L

def TruncGeo(p, c_crit):

    """
    Creates one sample of a truncated geometric distribution.
    """
    s = 1 + Geometric(probs= p).sample()
    if s > c_crit:
        s = c_crit
    
    return s

def MixedNorm(mu_1, mu_2, sigma_1, sigma_2):

    """
    Creates one sample of a mixed Gaussian distribution.
    """
    z = Binomial(total_count=1, probs = 0.5).sample()
    if (z == 1):
        h = Normal(loc = mu_1, scale = sigma_1).sample()
    else:
        h = Normal(loc = mu_2, scale = sigma_2).sample()
    return h

def prepare_pypesto(inferer, D, x_obs, n_chains):

    """
    Defines custom_problem, sampler, result_custom_problem in preparation to sample with pyPESTO.
    """

    # define objective function
    def f(log_params):

        # resize one prior sample
        thetas = torch.tensor(log_params, dtype = torch.float).repeat(1146,1)

        # get likelihoods
        likelihoods = inferer._loss(theta = thetas, x = x_obs[0])

        # sum likelihoods to include all observations
        return  torch.sum(likelihoods).detach().numpy()
    objective = pypesto.Objective(fun = f)

    
    
    
    # define optimisation bounds
    if D ==  8:
        lb = np.repeat(-7,D)
        ub = np.repeat(3,D)
    else:
        d = D-8
        lb = np.repeat(-7,D)
        lb[-d:] = np.repeat(-2,d)
        ub = np.repeat(3,D)
        ub[-d:] = np.repeat(2,d)

    # create problem
    custom_problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    # choose optimiser
    optimizer = optimize.ScipyOptimizer()

    # find startting conditions
    result_custom_problem = optimize.minimize(
        problem=custom_problem, optimizer=optimizer, n_starts=10
    )

    # choose sampling method
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(), n_chains = n_chains
    )
    
    return custom_problem, sampler, result_custom_problem