from utils import get_histogram, pickle_save
from functools import reduce
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit

from utils import (
  get_community_cdf,
  is_low_closed,
  is_high_closed,
)

from constants import (
  min_open_lo, max_open_lo,
  min_open_hi, max_open_hi,
  n_lm_comp,
  mu0, sigma0, w0,
  mu_bounds, sigma_bounds, w_bounds,
)

class Pipeline:

  def __init__(self, ir_model):
    # TODO add previous steps:
    # - fetching and filtering training questions
    # - build up calibration dataset from said questions
    # - fitting ir model
    self.ir_model = ir_model

  def predict(self, question: dict):
    ccdf = get_community_cdf(question)
    rcdf = self.ir_model.predict(ccdf)
    lm = LogisticMixture()
    lm.fit(rcdf)
    return lm.get_prediction_data(
      low_closed=is_low_closed(question),
      high_closed=is_high_closed(question),
    )

# TODO this could be split into 2 files: recalibrate.py and lm_fit.py

# Recalibration

def F_star(question):
  """CDF evaluated at resolution"""
  h = get_histogram(question)
  return h[h['x'] <= question['resolution']]['c'].sum() / h.shape[0]

def P_hat(questions):
  """Empirical probability that resolution is indeed below CDF_community(resolution)"""
  return lambda p: len([q for q in (questions.values() if isinstance(questions, dict) else questions) if F_star(q) <= p]) / len(questions)

def calibration_dataset(questions):
  """Generate pairs (CDF_community(resolution), empirical_prob(resolution <= CDF_community(resolution)) to run regression on"""
  d = []
  p_hat = P_hat(questions)
  for q in (questions.values() if isinstance(questions, dict) else questions):
    try:
      fstar = F_star(q)
      d.append((fstar, p_hat(fstar)))
    except Exception as e:
      print(f"Could not add question {q['id']} to calibration dataset")
      print(e)
  return d

def fit_ir(qs, save_paths={}):
  cd = calibration_dataset(qs)
  df = pd.DataFrame(cd, columns=['x', 'y'])
  
  if save_paths:
    np.save(save_paths['calibration'], cd)
  
  model = IsotonicRegression(
    y_min=0,
    y_max=1,
    increasing=True,
    out_of_bounds='clip'
    ).fit(df['x'], df['y'])
  
  if save_paths:
    pickle_save(model, save_paths['model'])

  return model

def predict_ir(ir_model, q):
  return ir_model.predict(get_histogram(q)['c'].values.tolist())

# Logistic mixture fitting

def logit(x, x0, dx):
  _x = (x-x0)/dx
  return 1 / (1 + np.exp(-_x))

def logit_pdf(x, x0, dx):
  _x = (x-x0)/dx
  return np.exp(-_x) / (dx * (1 + np.exp(-_x))**2)

def logistic_mixture(
  x,
  x01, x02, x03,# x04, x05,
  dx1, dx2, dx3,# dx4, dx5,
  w1, w2, w3,# w4,
):
  x0s = x01, x02, x03#, x04, x05
  dxs = dx1, dx2, dx3#, dx4, dx5
  ws = w1, w2, w3#, w3, w4, 1-w1-w2-w3-w4
  if sum(ws) != 1: ws = [w / sum(ws) for w in ws]
  return sum([w * logit(x, x0, dx) for (w, x0, dx) in zip(ws, x0s, dxs)])

# TODO preparing submission for API shouldn't be in this class maybe?
class LogisticMixture:

  def __init__(self, xdata=None, n_comp=n_lm_comp, func=logistic_mixture): #, pdf=lm_pdf):
    self.func = func
    self.xdata = xdata or np.linspace(0, 1, 201)
    self.n_comp = n_comp

    concatMap = lambda f, xs: reduce(lambda x, y: x+y, map(f, xs))
    buildPoint = lambda xs: concatMap(lambda x: [x]*self.n_comp, xs)
    self._p0 = buildPoint((mu0, sigma0, w0))
    self._bounds = (
      buildPoint((mu_bounds[0], sigma_bounds[0], w_bounds[0])),
      buildPoint((mu_bounds[1], sigma_bounds[1], w_bounds[1])),
    )

  def fit(self, y, x=None, **kwds):
    self.opt, self.cov = curve_fit(
      f=self.func,
      xdata=x or self.xdata,
      ydata=y,
      p0=self._p0,
      bounds=self._bounds,
      **kwds
    )

  def predict(self, x):
    return self.func(x, *self.opt)

  # def pdf(self, x):


  @property
  def x0s(self):
    return self.opt[:3]

  @property
  def dxs(self):
    return self.opt[3:-3]

  @property
  def ws(self):
    return self.opt[-3:]

  @classmethod
  def format_logistic_for_api(cls, x0, dx, w, low_closed, high_closed):
    
    lo = 0 if low_closed else np.clip(logit(0, x0, dx), min_open_lo, max_open_lo)
    hi = 1 if high_closed else np.clip(logit(1, x0, dx), min_open_hi + lo, max_open_hi)
    
    return {
      "kind": "logistic",
      "x0": float(x0),
      "s": float(dx),
      "w": float(w),
      "low": lo,
      "high": hi,
    }

  def get_prediction_data(self, low_closed, high_closed):
    return {
      "prediction": {
      "kind": "multi",
      "d": sorted(
        [
          self.format_logistic_for_api(x0, dx, w, low_closed, high_closed)
          for x0, dx, w in zip(self.x0s, self.dxs, self.ws)
        ],
          key=lambda l: -l["w"],
          ),
        },
        "void": False,
      }