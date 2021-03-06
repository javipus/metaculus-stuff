"""Fit recalibrated ECDF to logistic mixture using scipy."""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy.optimize import curve_fit

import ergo

from utils import get_histogram, is_open
from model import LogisticMixture
from KEYS import USERNAME, PASSWORD

# def logit(x, x0, dx):
#   _x = (x-x0)/dx
#   return 1 / (1 + np.exp(-_x))

# def logistic_mixture(
#   x,
#   x01, x02, x03,# x04, x05,
#   dx1, dx2, dx3,# dx4, dx5,
#   w1, w2, w3,# w4,
# ):
#   x0s = x01, x02, x03#, x04, x05
#   dxs = dx1, dx2, dx3#, dx4, dx5
#   ws = w1, w2, w3#, w3, w4, 1-w1-w2-w3-w4
#   if sum(ws) != 1: ws = [w / sum(ws) for w in ws]
#   return sum([w * logit(x, x0, dx) for (w, x0, dx) in zip(ws, x0s, dxs)])

# class LogisticMixture:

#   def __init__(self, xdata=None, func=logistic_mixture):
#     self.func = func
#     self.xdata = xdata or np.linspace(0, 1, 201)

#   def fit(self, y, x=None, **kwds):
#     self.opt, self.cov = curve_fit(
#       f=self.func,
#       xdata=x or self.xdata,
#       ydata=y,
#       p0=[.5, .5, .5, .5, .5, .5, 1/3, 1/3, 1/3],
#       bounds=([0, 0, 0, 1e-2, 1e-2, 1e-2, .001, .001, .001], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
#       **kwds
#     )

#   def predict(self, x):
#     return self.func(x, *self.opt)

#   @property
#   def x0s(self):
#     return self.opt[:3]

#   @property
#   def dxs(self):
#     return self.opt[3:-3]

#   @property
#   def ws(self):
#     return self.opt[-3:]

#   @classmethod
#   def format_logistic_for_api(cls, x0, dx, w, low_closed, high_closed):
    
#     lo = 0 if low_closed else np.clip(logit(0, x0, dx), min_open_lo, max_open_lo)
#     hi = 1 if high_closed else np.clip(logit(1, x0, dx), min_open_hi + lo, max_open_hi)
    
#     return {
#       "kind": "logistic",
#       "x0": float(x0),
#       "s": float(dx),
#       "w": float(w),
#       "low": lo,
#       "high": hi,
#     }

#   def get_prediction_data(self, low_closed, high_closed):
#     return {
#       "prediction": {
#       "kind": "multi",
#       "d": sorted(
#         [
#           self.format_logistic_for_api(x0, dx, w, low_closed, high_closed)
#           for x0, dx, w in zip(self.x0s, self.dxs, self.ws)
#         ],
#           key=lambda l: -l["w"],
#           ),
#         },
#         "void": False,
#       }

# def get_histogram(question):
#   """Metaculus and community pdfs over normalized range."""
#   h = pd.DataFrame(question.prediction_histogram)
#   h.columns = ['x', 'm', 'c']
#   return h

def submit_recalibrated(qid, ir_model, session=None, username=USERNAME, password=PASSWORD, debug=False):
  if session is None:
    metaculus = ergo.Metaculus()
    metaculus.login_via_username_and_password(username=username, password=password)
  else:
    metaculus = session
  question = metaculus.get_question(qid)
  comm_cdf = get_histogram(question.data)['c'].cumsum() / 200
  cdf = ir_model.predict(comm_cdf)
  lm = LogisticMixture()
  lm.fit(cdf)
  low_closed, high_closed = [not question.possibilities.get(side)=='tail' for side in ('low', 'high')]
  data = lm.get_prediction_data(low_closed, high_closed)
  
  if not debug and is_open(question):
    print(f"\t{question.name}")
    metaculus.predict(q_id=str(qid), data=data)

  return lm.func(np.linspace(0,1,len(cdf)), *lm.opt), cdf, comm_cdf.squeeze().values

# TODO test this!

def diagnose(a, b, c, ax=None):
  assert len(a) == len(b) == len(c)
  df = pd.DataFrame([a, b, c], index=['fit', 'ecdf', 'comm_cdf']).T  
  ax = sns.lineplot(data=df, ax=ax)
  return ax

if __name__=='__main__':

  DEBUG = False
  CATS = [
    "series--maximum-likelihood",
    "series--hill-climbing",
    ]
  N_PAGES = 2
  
  metaculus = ergo.Metaculus()
  metaculus.login_via_username_and_password(username=USERNAME, password=PASSWORD)
  
  # qs = []
  # for cat in CATS:
  #   qs.extend(metaculus.get_questions(cat=cat, pages=N_PAGES))

  # qids = [q.id for q in qs]

  with open("db/ir_model.pkl", "rb") as f:
    model = pickle.load(f)

  with open("qids.csv", "r") as f:
    qids = [int(qid) for qid in f.readlines()]

  r = []
  fig, ax = plt.subplots(8,9)
  for (qid, _ax) in zip(qids, ax.flatten()):
    print(f"{qid}") #": {[q.name for q in qs if q.id==qid][0]}")
    
    try:
      new = (qid, *submit_recalibrated(qid, model, session=metaculus, debug=DEBUG))
      r.append(new)
      diagnose(*r[-1][1:], ax=_ax)
      _ax.set_title(str(r[-1][0]))
    except Exception as e:
      # raise
      print(e)

  ax = diagnose(*r[0][1:])
  fig.show()