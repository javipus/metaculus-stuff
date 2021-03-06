import numpy as np

from scipy.optimize import curve_fit

def f(x, *coeffs):
  return sum(np.dot(x**i, c) for i, c in enumerate(coeffs))

def f_wrapper(x, n, *args):
  nargs = map(lambda i: np.array(args[i::n]), range(n))
  return f(x, *nargs)

deg = 2
n_dim = 2
fit_func = lambda x, *args: f_wrapper(x, n_dim, *args)
eps = 1e-3
a, b, c = np.random.randn(deg+1, n_dim)

x_ = np.linspace(-5, 5, 101)
xy = [np.array((x, y)) for x in x_ for y in x_]
z = [a*xy_**2 + b*xy_ + c + eps*np.random.randn(*xy_.shape) for xy_ in xy]

opt, cov = curve_fit(
  f=fit_func,
  xdata=xy,
  ydata=z,
  p0=None,
)