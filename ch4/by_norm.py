# Bayes estimation 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def map_norm(x, alpha, beta, gamma, delta):
    N = len(x)
    mu = (np.sum(x) + gamma*delta) / (N + gamma)
    var_up = np.sum((x - mu)**2) + 2*beta + gamma*(delta-mu)**2
    var_down = N + 3 + 2*alpha
    var = var_up / var_down
    return mu, var

def mle_norm(x):
    N = len(x)
    mu = np.sum(x)/N
    var = np.sum((x - mu)**2)/N
    return mu, var

def by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    N = len(x)
    alpha_post = alpha_prior + N/2
    beta_post = sum(x.^2)/2 + beta_prior + (gamma_prior*delta_prior^2)/2 \
         - (gamma_prior*delta_prior + sum(x))^2 / (2*(gamma_prior + N))
    gamma_post = gamma_prior + N
    delta_post = (gamma_prior*delta_prior + sum(x)) / (gamma_prior + N)

    # Compute intermediate parameters.
    alpha_int = alpha_post + 0.5
    beta_int = (x_test**2)/2 + beta_post + (gamma_post*delta_post**2)/2 - \
        (gamma_post*delta_post + x_test)**2 / (2*gamma_post + 2)
    gamma_int = gamma_post + 1

    # Predict values for x_test.
    temp1 = np.sqrt(gamma_post) * (beta_post^alpha_post) * gamma(alpha_int)
    x_prediction_up = np.matlib.repmat(temp1, 1, len(x_test))
    x_prediction_down = np.sqrt(2*pi) * np.sqrt(gamma_int) * gamma(alpha_post) \
        * beta_int**alpha_int
    x_prediction =  x_prediction_up / x_prediction_down

    return alpha_post, beta_post, gamma_post, delta_post, x_prediction