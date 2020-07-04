# Bayes estimation 

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import gamma

def by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    N = len(x)
    alpha_post = alpha_prior + N/2
    beta_post = np.sum(x**2)/2 + beta_prior + (gamma_prior*delta_prior**2)/2 - (gamma_prior*delta_prior + np.sum(x))**2 / (2*(gamma_prior + N))
    gamma_post = gamma_prior + N
    delta_post = (gamma_prior*delta_prior + np.sum(x)) / (gamma_prior + N)

    # Compute intermediate parameters.
    alpha_int = alpha_post + 0.5
    beta_int = (x_test**2)/2 + beta_post + (gamma_post*delta_post**2)/2 - (gamma_post*delta_post + x_test)**2 / (2*gamma_post + 2)
    gamma_int = gamma_post + 1

    # Predict values for x_test.
    temp1 = np.sqrt(gamma_post) * (beta_post**alpha_post) * gamma(alpha_int)
    x_prediction_up = repmat(temp1, 1, len(x_test))
    x_prediction_down = np.sqrt(2*np.pi) * np.sqrt(gamma_int) * gamma(alpha_post) * beta_int**alpha_int
    x_prediction =  x_prediction_up / x_prediction_down

    return alpha_post, beta_post, gamma_post, delta_post, x_prediction

original_mu = 5
original_sig = 8
N = 5
r = original_mu + original_sig * np.random.randn(N,1)

# Estimate the mean and the variance for the data in r.
# Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
# sake of the example. Other values can be tried too.
x_test = np.arange(-20, 30, 0.01)
alpha_post, beta_post, gamma_post, delta_post, x_prediction = \
    by_norm(r, 1, 1, 1, 0, x_test)

# Plot the original and the estimated models for comparison.
original = norm.pdf(x_test, original_mu, original_sig)
# estimated = norm.pdf(x, estimated_mu, estimated_sig)
plt.plot(x_test, original, color='green', label="original")
plt.plot(x_test, x_prediction[0], color='blue', label="estimated")
plt.legend()
plt.show()