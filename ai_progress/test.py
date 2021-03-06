import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recalibrated_distributions import sample_cdf

def test_sample_cdf(x0, x1, cdf, n_samples=int(1e4), n_points=101):
  x = np.linspace(x0, x1, n_points)
  y = np.linspace(0, 1, n_points)
  real_cdf = pd.DataFrame([x, cdf(x)], index=['x', 'y']).T
  samples = sample_cdf(x0, x1, cdf, n_samples)
  sampled_cdf = pd.DataFrame([
    y,
    pd.Series(samples).quantile(y).values
    ], index=['y', 'x']).T

  fig, ax = plt.subplots()
  ax.plot(real_cdf.x, real_cdf.y, label='real')
  ax.plot(sampled_cdf.x, sampled_cdf.y, label='sampled')
  ax.legend()
  return ax, samples

def test_sample_pdf(x0, x1, pdf, n_samples=int(1e4), n_points=101):
  x = np.linspace(x0, x1, n_points)
  y = np.linspace(0, 1, n_points)
  real_pdf = pd.DataFrame([x, pdf(x)], index=['x', 'y']).T
  samples = sample_cdf(x0, x1, pdf, n_samples)
  sampled_pdf = pd.DataFrame([
    y,
    pd.Series(samples).quantile(y).values
    ], index=['y', 'x']).T

  fig, ax = plt.subplots()
  ax.plot(real_pdf.x, real_pdf.y, label='real')
  ax.plot(sampled_pdf.x, sampled_pdf.y, label='sampled')
  ax.legend()
  return ax, samples