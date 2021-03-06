import numpy as np

from scipy.optimize import curve_fit

from model import logit, logit_pdf

from utils import (
  get_histogram,
  get_community_cdf,
  get_norm_resolution,
  interpolate
)

def scoring_(p_star, pc_star, pu, N):
  B = 20 * np.log2(1 + N / 30)
  A = 40 + B
  abs_sc = A * np.log(max(p_star / pu, .02))
  rel_sc = B * np.log(max(p_star / pc_star, .02))
  return abs_sc + rel_sc

def score_histogram(question, p, tol=1e-3):
  # TODO pc is added to pu then renormalized, see https://discord.com/channels/694850840200216657/694850840707596320/798616689918738453
  # TODO pc is actually calculated by logistic best fit
  # Normalized resolution
  x_norm = get_norm_resolution(question)

  # Is question range open or closed?
  pc = get_histogram(question)['c'].values.tolist()
  assert len(pc) == len(p)
  n_bins = len(p) - 1
  dx = 1 / n_bins
  closed = abs(sum(pc)*dx - 1) < tol # for lack of anything better

  # Community distribution
  pc_star = interpolate(pc, x_norm, closed)

  # Uniform distribution
  pu = 1 / n_bins if closed else (1-.15) / n_bins
  
  # Own prediction
  p_star = interpolate(p, x_norm, closed)

  # # predictions
  N = question['number_of_predictions']

  my_sc = scoring_(p_star, pc_star, pu, N)
  comm_sc = scoring_(pc_star, pc_star, pu, N)
  
  return my_sc - comm_sc

def score_logistic_mixture(question, fitted_lm):
  x_norm = get_norm_resolution(question)
  ccdf = get_community_cdf(question)
  comm_opt, _ = curve_fit(f=logit, xdata=np.linspace(0,1,len(ccdf)), ydata=ccdf)
  
  pc_star = logit_pdf(x_norm, *comm_opt)
  p_star = fitted_lm.pdf(x_norm)

  # TODO question range open or closed
  # TODO add + pu and renormalize

  return scoring_(p_star, pc_star, pu, N)