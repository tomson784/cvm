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

def mle_norm(x):
    N = len(x)
    mu = np.sum(x)/N
    var = np.sum((x - mu)**2)/N
    return mu, var

original_mu = 5
original_sig = 8
# I can be modified in order to see how MAP behaves for small vs big
# amounts of data.
# N = 1000000
N = 100
r = original_mu + original_sig * np.random.randn(N,1)

# Estimate the mean and the variance for the data in r.
# Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
# sake of the example. Other values can be tried too.
[estimated_mu, estimated_var] = map_norm(r, 1, 1, 1, 0)
# disp(['Estimated mean: ', num2str(estimated_mu)])
estimated_sig = np.sqrt(estimated_var)

[mle_mu, mle_var] = mle_norm(r)
mle_sig = np.sqrt(mle_var)

# Estimate and print the error for the mean and the standard deviation.
# muError = abs(original_mu - estimated_mu)
# sigError = abs(original_sig - estimated_sig)
# disp(['Errors: ', num2str(muError), ', ', num2str(sigError)])

# Plot the original and the estimated models for comparison.
x = np.arange(-20, 30, 0.01)
original = norm.pdf(x, original_mu, original_sig)
estimated = norm.pdf(x, estimated_mu, estimated_sig)
mle = norm.pdf(x, mle_mu, mle_sig)
plt.plot(x, original, color='green', label="original")
plt.plot(x, estimated, color='blue', label="MAP")
plt.plot(x, mle, color='red', label="MLE")
plt.xlim(-20,30)
plt.ylim(0,0.1)
plt.legend()
plt.show()