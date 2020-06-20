# Computer vision: models, learning and inference
# Chapter 4.1
# Maximum likelyhood estimation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def mle_norm(x):
    N = len(x)
    mu = np.sum(x)/N                # (4.12)
    var = np.sum((x - mu)**2)/N     # (4.13)
    return mu, var

original_mu = 5
original_sig = 8
r = original_mu + original_sig * np.random.randn(100,1)

# Estimate the mean and the variance for the data in r.
[estimated_mu, estimated_var] = mle_norm(r)
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