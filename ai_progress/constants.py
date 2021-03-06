# weird CDF bounds that are not documented anywhere
min_open_lo = .01
max_open_lo = .98
min_open_hi = .01
max_open_hi = .99

# How many components to use in the submitted logistic mixture? (max = 5)
n_lm_comp = 3

# (mu, sigma, w) bounds
mu_bounds = 0, 1
sigma_bounds = 1e-2, 1
w_bounds = 1e-3, 1

# Point in (mu, sigma, w) space where scipy.optimize.curve_fit should start
mu0, sigma0, w0 = .5, .5, 1/3