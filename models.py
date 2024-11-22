"""
This code is containing all models used in the thesis.
"""


# install packages
import torch
from torch.distributions import Binomial
from torch.distributions import Exponential

# custom package
import helper_functions as hp

# general setting
K = 5 # number of possible states


# set seed
torch.manual_seed(2024)


# hyper-parameters
p = 0.6
mu_f = 11.5
mu_m = 13.5
sigma = 2
c_obs = 26
nu = 1/33
c_trans = 15

"Model A"

def simulate_model_a(log_params):

    # transform from log-domain
    params = torch.exp(log_params) 

    # sample initial state
    s_0 = hp.TruncGeo(p = p, c_crit = K)
    

    # sample observation time and convert to integer
    T = c_obs + torch.floor(Exponential(nu).sample())

    # obtain transition parameters
    Lambda = params
    


    # simulate proposal time(s)
    if (int(s_0) == 1):
        tau_prop = Exponential(rate = Lambda[0]).sample()
        s_1 = torch.tensor([2])
    elif (int(s_0) == 5):
        tau_prop = Exponential(rate = Lambda[7]).sample()
        s_1 = torch.tensor([4])

    else:
        tau_props = torch.tensor([Exponential(rate = Lambda[2*int(s_0)-3]).sample(),Exponential(rate = Lambda[2*int(s_0)-2]).sample()])
        tau_prop = torch.min(tau_props)
        index = torch.argmin(tau_props)
        s_1 = s_0 + 2*index - 1

    # compute transition time
    tau = c_trans + tau_prop

    # check if transition happens in the observation frame
    if tau > T:
         return torch.tensor([s_0,s_0,T])
    else:
         return torch.tensor([s_0,s_1,T])


"Model B"

def simulate_model_b(log_params):
    
    # transform from log-domain
    params = torch.exp(log_params) 

    # sample initial state
    s_0 = hp.TruncGeo(p = p, c_crit = K)

    # sample observation time and convert to integer
    T = c_obs + torch.floor(Exponential(nu).sample())

    # obtain transition parameters
    Lambda = params
    
    # set time counter and current state for repeated transitions
    t = 0
    s = s_0
    while(t <= T):
        # simulate proposal time(s)
        if (int(s) == 1):
            tau_prop = Exponential(rate = Lambda[0]).sample()
            s_tilde = torch.tensor([2])
        elif (int(s) == 5):
            tau_prop = Exponential(rate = Lambda[7]).sample()
            s_tilde = torch.tensor([4])

        else:
            tau_props = torch.tensor([Exponential(rate = Lambda[2*int(s)-3]).sample(),Exponential(rate = Lambda[2*int(s)-2]).sample()])
            tau_prop = torch.min(tau_props)
            index = torch.argmin(tau_props)
            s_tilde = s + 2*index - 1

        # compute transition time
        tau = c_trans + tau_prop
        t = t + tau

        # check if transition happens in the observation frame
        if(t <= T):
            s = s_tilde

    
    return torch.tensor([s_0,s,T])

"Model C"

def simulate_model_c(log_params):
   
    # transform rate parameters from from log-domain
    params = torch.exp(log_params)
    params[-1] = log_params[-1] 

    # sample initial state and haemoglobin
    s_0 = hp.TruncGeo(p = p, c_crit = K)
    h = hp.MixedNorm(mu_1 = mu_f,mu_2 = mu_m, sigma_1 = sigma, sigma_2 = sigma)

    # sample observation time and convert to integer
    T = c_obs + torch.floor(Exponential(nu).sample())

    # obtain transition parameters and add linear dependency of h
    Lambda = params[:8] + params[8]*h
    # we have to ensure the parameters remain positive
    for i in range(8):
        Lambda[i] = torch.max(Lambda[i], torch.exp(torch.Tensor([-7])))
    

    # set time counter and current state for repeated transitions
    t = 0
    s = s_0
    while(t <= T):
        # simulate proposal time(s)
        if (int(s) == 1):
            tau_prop = Exponential(rate = Lambda[0]).sample()
            s_tilde = torch.tensor([2])
        elif (int(s) == 5):
            tau_prop = Exponential(rate = Lambda[7]).sample()
            s_tilde = torch.tensor([4])

        else:
            tau_props = torch.tensor([Exponential(rate = Lambda[2*int(s)-3]).sample(),Exponential(rate = Lambda[2*int(s)-2]).sample()])
            tau_prop = torch.min(tau_props)
            index = torch.argmin(tau_props)
            s_tilde = s + 2*index - 1

        # compute transition time
        tau = c_trans + tau_prop
        t = t + tau

        # check if transition happens in the observation frame
        if(t <= T):
            s = s_tilde

    return torch.tensor([s_0,s,T,h])

"Model D"

def simulate_model_d(log_params):
   
    # transform rate parameters from from log-domain
    params = torch.exp(log_params)
    params[-5:] = log_params[-5:]

    # sample initial state and haemoglobin
    s_0 = hp.TruncGeo(p = p, c_crit = K)
    h = hp.MixedNorm(mu_1 = mu_f,mu_2 = mu_m, sigma_1 = sigma, sigma_2 = sigma)

    # sample observation time and convert to integer
    T = c_obs + torch.floor(Exponential(nu).sample())

    # obtain transition parameters and add linear dependency of h depending on the state
    Lambda = params[:8]
    for i in range((8)):
        if i == 0:
            Lambda[i] = Lambda[i] + params[8]*h
        elif i == 1 or i == 2:
            Lambda[i] = Lambda[i] + params[9]*h
        elif i == 3 or i == 4:
            Lambda[i] = Lambda[i] + params[10]*h
        elif i == 5 or i == 6:
            Lambda[i] = Lambda[i] + params[11]*h
        else:
            Lambda[i] = Lambda[i] + params[12]*h

        # we have to ensure the parameters remain positive
        Lambda[i] = torch.max(Lambda[i], torch.exp(torch.Tensor([-7])))
    
    # set time counter and current state
    t = 0
    s = s_0
    while(t <= T):
        # simulate proposal time(s)
        if (int(s) == 1):
            tau_prop = Exponential(rate = Lambda[0]).sample()
            s_tilde = torch.tensor([2])
        elif (int(s) == 5):
            tau_prop = Exponential(rate = Lambda[7]).sample()
            s_tilde = torch.tensor([4])

        else:
            tau_props = torch.tensor([Exponential(rate = Lambda[2*int(s)-3]).sample(),Exponential(rate = Lambda[2*int(s)-2]).sample()])
            tau_prop = torch.min(tau_props)
            index = torch.argmin(tau_props)
            s_tilde = s + 2*index - 1

        # compute transition time
        tau = c_trans + tau_prop
        t = t + tau

        # check if transition happens in the observation frame
        if(t <= T):
            s = s_tilde

    
    return torch.tensor([s_0,s,T,h])

# define function for loading the models
def get_model(model):

    """
    Returns number of paramters, dimension of simulated observations and simulator function
    """

    # D ist the number of parameters
    # x_dim is the dimension of simulated observations

    if model == "model_a":
        D = 8
        x_dim = 3

        simulate = simulate_model_a
        
    elif model == "model_b":
        D = 8
        x_dim = 3

        simulate = simulate_model_b

    elif model == "model_c":
        D = 9
        x_dim = 4

        simulate = simulate_model_c
            
    elif model == "model_d":
        D = 13
        x_dim = 4

        simulate = simulate_model_d
            
    return D, x_dim, simulate
