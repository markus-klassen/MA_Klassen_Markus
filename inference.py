"""
Sets up normalising flow and trains the networks.
"""

# install packages and dependencies

import pickle
import torch

from sbi.inference import SNLE, simulate_for_sbi
from sbi import utils 

from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

# custom packages
import models
import helper_functions as hp


# load true data
data, L = hp.get_base()

# hyper-parameters
p = 0.6
mu_f = 11.5
mu_m = 13.5
sigma = 2
c_obs = 26
nu = 1/33


# wrapper function for all models that sets up the network and trains it
def perform_inference(model):
    
    # initialise
    D, x_dim, simulate = models.get_model(model)

    # set seed
    torch.manual_seed(2024)

    # define prior
    if D == 8:
        prior = utils.BoxUniform(low = -7 * torch.ones(D), high = 3 * torch.ones(D))
    else:
        prior = utils.MultipleIndependent(
            [utils.BoxUniform(low = -7 * torch.ones(8), high = 3 * torch.ones(8)),
            utils.BoxUniform(low = -2 * torch.ones(D-8), high = 2 * torch.ones(D-8))]
        )

    # adapt prior and simulator to sbi-friendly format
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(
        simulate,
        prior,
        prior_returns_numpy,
    )

    # set up inference instance
    inferer = SNLE(prior, show_progress_bars=True, density_estimator="nsf")

    # create synthetic data
    theta, x = simulate_for_sbi(simulator, prior, 100000, simulation_batch_size=1000)

    # train normalising flow
    inferer.append_simulations(theta, x).train(training_batch_size=1000);


    # save trained flow
    with open(rf"flow_{model}.pkl", "wb") as handle:
        pickle.dump(inferer, handle)

"Train the models"

perform_inference("model_a")
perform_inference("model_b")
perform_inference("model_c")
perform_inference("model_d")