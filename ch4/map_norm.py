# Computer vision: models, learning and inference
# Chapter 4.2
# Maximum a posteriori (MAP) estimation 

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

original_mu = 5
original_sig = 8
# I can be modified in order to see how MAP behaves for small vs big amounts of data.
# N = 1000000
N = 100
r = original_mu + original_sig * np.random.randn(N,1)

# Estimate the mean and the variance for the data in r.
# Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
# sake of the example. Other values can be tried too.
estimated_mu, estimated_var = map_norm(r, 1, 1, 1, 0)
estimated_sig = np.sqrt(estimated_var)

# Estimate and print the error for the mean and the standard deviation.

# Plot the original and the estimated models for comparison.
x = np.arange(-20, 30, 0.01)
original = norm.pdf(x, original_mu, original_sig)
estimated = norm.pdf(x, estimated_mu, estimated_sig)
plt.plot(x, original, color='green', label="original")
plt.plot(x, estimated, color='blue', label="estimated")
plt.legend()
plt.show()