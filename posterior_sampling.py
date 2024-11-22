""""
computes the posterior plots and data for posterior analysis.
"""


# install packages
import pickle
import numpy as np
import torch

from sbi import utils as utils
from sbi import analysis
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)
import pypesto.sample as sample

# custom packages
import models
import helper_functions as hp


# initialise data
data, L = hp.get_base()

# wrapper for posterior sampling and quantities for posterior analysis
def perform_posteriors(model):

    # get model-specific hyper-parameters
    D, x_dim, simulate = models.get_model(model)

    # bring true data in correct shape
    if x_dim == 3:
        x_obs = [torch.zeros(L,x_dim)]
        for i in range(L):
            x_obs[0][i] = torch.tensor([data["initial_state"].iloc[i],
                                    data["final_state"].iloc[i],
                                    data["observation_time"].iloc[i]])
    elif x_dim == 4:
        x_obs = [torch.zeros(L,x_dim)]
        for i in range(L):
            x_obs[0][i] = torch.tensor([data["initial_state"].iloc[i],
                                    data["final_state"].iloc[i],
                                    data["observation_time"].iloc[i],
                                    data["hb"].iloc[i]])            
    
    # define prior
    if D == 8:
        prior = utils.BoxUniform(low = -7 * torch.ones(D), high = 3 * torch.ones(D))
    else:
        prior = utils.MultipleIndependent(
            [utils.BoxUniform(low = -7 * torch.ones(8), high = 3 * torch.ones(8)),
            utils.BoxUniform(low = -2 * torch.ones(D-8), high = 2 * torch.ones(D-8))]
        )

    # load flow
    with open(rf'flow_{model}.pkl', 'rb') as f:
        inferer = pickle.load(f)

    # number of chains for sampling
    n_chains = 4

    # number of chains used in the tempered MCMC
    n_temp_chains = 4

    # set up for pyPESTO
    custom_problem, sampler, result_custom_problem = hp.prepare_pypesto(inferer, D, x_obs, n_temp_chains)

    """posterior samples"""

    # number of chains for sampling
    n_chains = 4

    # number of posterior samples
    n_post_samples = 100000 

    # save burn in, effective sample size and the posterior samples
    gw_post_samples = np.zeros(n_chains)
    ess_post_samples = np.zeros(n_chains)
    post_samples = np.zeros((n_chains, n_post_samples + 1, D))

    for i in range(n_chains):

        # set up for pyPESTO
        custom_problem, sampler, result_custom_problem = hp.prepare_pypesto(inferer, D, x_obs, n_chains)

        # sample
        result = sample.sample(
        custom_problem, n_samples = n_post_samples, sampler = sampler, result = result_custom_problem, filename=None
        )

        # extract values from the chain with real temperature
        post_samples[i] = result.sample_result["trace_x"][0]
        gw_post_samples[i] = sample.geweke_test(result)
        ess_post_samples[i] = sample.effective_sample_size(result)
        

    with open(rf"post_samples_{model}.pkl", "wb") as handle:
        pickle.dump(post_samples, handle)

    with open(rf"gw_post_samples_{model}.pkl", "wb") as handle:
        pickle.dump(gw_post_samples, handle)

    with open(rf"ess_post_samples_{model}.pkl", "wb") as handle:
        pickle.dump(ess_post_samples, handle)


# define function for simlation-based calibration
def perform_sbc(model):
    """
    Simulation based calibration from sbi.
    """

    # initialise
    D, x_dim, simulate = models.get_model(model)

    # this needs a full sbi set-up, so we again load prior and simulator
    
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

    # load flow
    with open(rf'flow_{model}.pkl', 'rb') as f:
        inferer = pickle.load(f)

    # define posterior and sampling method
    mcmc_parameters = dict(
        num_chains=50,
        thin=5,
        warmup_steps=30,
        init_strategy="proposal",
    )
    mcmc_method = "slice_np_vectorized"

    posterior = inferer.build_posterior(
        mcmc_method=mcmc_method,
        mcmc_parameters=mcmc_parameters,
    )
    
    # number of sbc runs
    n_sbc_runs = 300  

    # number of posterior samples per run
    n_sbc_samples = 1000

    # generate ground truth parameters and corresponding simulated observations for SBC.
    thetas = prior.sample((n_sbc_runs,))
    xs = simulator(thetas)

    # run sbc
    ranks, dap_samples = analysis.run_sbc(
        thetas, xs, posterior, num_posterior_samples=n_sbc_samples
    )


    # save results
    with open(rf"thetas_{model}.pkl", "wb") as handle:
        pickle.dump(thetas, handle)

    with open(rf"ranks_{model}.pkl", "wb") as handle:
        pickle.dump(ranks, handle)

    with open(rf"dap_samples_{model}.pkl", "wb") as handle:
        pickle.dump(dap_samples, handle)





perform_posteriors("model_a")
perform_posteriors("model_b")
perform_posteriors("model_c")
perform_posteriors("model_d")

perform_sbc("model_a")
perform_sbc("model_b")
perform_sbc("model_c")
perform_sbc("model_d")