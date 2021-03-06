#!/usr/bin/env python
# coding: utf-8

# TODO update only if new prediction is significantly different from old one as measured by KS test

try:
  import ergo
except ModuleNotFoundError:
  print("Ergo could not be imported")
import datetime, pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rv_continuous

from KEYS import USERNAME, PASSWORD

import warnings
warnings.filterwarnings("ignore")

# Questions I've forecasted manually - should be skipped
MANUAL = [
  5903,
  # TODO remove below this
  # 5958,
  # 5888,
  # 5951,
  # 5960,
  # 5961,
  # 5965,
  # 5901,
  # 5897,
  # 5899,
  # 5900,
]

# Parameters
IR_MODEL_PATH = "db/ir_model.pkl"

CATS = [
    "series--maximum-likelihood",
    "series--hill-climbing",
]

# 20 qs/page, so this assumes <40 questions per round, in line with the "approximately 30" stated in the rules
# change accordingly if it fails
N_PAGES = 2

# kinda slow but it's worth it
N_SAMPLES = int(10e3)

# Functions
is_continuous = lambda q: q.data['possibilities']['type'] == 'continuous'
is_binary = lambda q: q.data['possibilities']['type'] == 'binary'
is_open = lambda q: (datetime.datetime.strptime(q.data['publish_time'], '%Y-%m-%dT%H:%M:%SZ') <\
                    datetime.datetime.utcnow() <\
                    datetime.datetime.strptime(q.data['close_time'], '%Y-%m-%dT%H:%M:%SZ')) and \
                    (q.data['resolution'] is None)

def default_to_community_cont(question, n_samples):
    """Defaults to community distribution for continuous questions"""
    samples = np.array([question.sample_community() for _ in range(n_samples)])
    question.submit_from_samples(samples)

def default_to_community_bin(question):
    """Defaults to community prediction for binary questions"""
    p = question.get_community_prediction()
    question.submit(p)

def default_to_community(question, n_samples=None):
    if is_open(question):
        if is_binary(question):
            default_to_community_bin(question)
        elif is_continuous:
            if n_samples is None:
                raise ValueError("n_samples can't be None for continuous questions")
            default_to_community_cont(question, n_samples)
        else:
            print(f"Question {question.id} is of type {q.data['possibilities'].get('type')} and cannot be predicted")
    else:
        print(f"Question {question.id} is not open!")

# simple rejection sampling, h/t https://stackoverflow.com/a/55772867
def _sample_cdf(x0, x1, cdf, n_samples=None, n_points=101, max_loops=np.inf):
  # TODO maybe I can replace this with ergo code and it'd be faster?
  # they have to use the histogram somehow and build the community distribution from that...
  samples = []
  M = max(cdf(x) for x in np.linspace(x0, x1, n_points))
  n_loops = 0
  p_below, p_above = cdf(x0), 1-cdf(x1)
  p_in = 1-p_below-p_above
  ps = [p_below, p_in, p_above]
  while len(samples) < n_samples and n_loops < max_loops:
    interval = 1 #np.random.choice([0, 1, 2], p=ps)
    if interval==0:
      samples.append(x0)
    elif interval==2:
      samples.append(x1)
    else:
      x = np.random.uniform(low=x0, high=x1)
      p = cdf(x)
      assert 0 <= p <= 1
      if np.random.uniform(low=0, high=1) <= p/M:
        samples.append(x)
      n_loops += 1
  return np.array(samples)

class TruncatedRV(rv_continuous):

  def __init__(self, x0, x1, n_points=1001, cdf=None, pdf=None, *args, **kwds):
    super().__init__(*args, **kwds)
    if pdf is not None and cdf is not None: raise ValueError("pdf XOR cdf")
    self.x0, self.x1 = x0, x1
    if cdf is not None: self._cdf = cdf
    if pdf is not None: self._pdf = pdf
    self.n_points = n_points
    self.pmax = max(self._pdf(x) for x in np.linspace(self.x0, self.x1, self.n_points))

  def _rvs(self, size, **kwds):
    p_below, p_above = self.cdf(self.x0), 1-self.cdf(self.x1)
    p_in = 1-p_below-p_above
    ps = [p_below, p_in, p_above]
    k = 0
    samples = []
    while k<size:
      i = np.random.choice([0, 1, 2], p=ps)
      if i==0:
        samples.append(self.x0)
      elif i==2:
        samples.append(self.x1)
      else:
        x = np.random.uniform(self.x0, self.x1)
        y = np.random.uniform(0, 1)*self.pmax
        if y<=self.pdf(x):
          samples.append(x)
        else:
          continue
      k += 1
    return np.array(samples)

def sample_cdf(x0, x1, cdf, n_samples):
  rv = TruncatedRV(x0=x0, x1=x1, cdf=cdf)
  return rv._rvs(size=n_samples)

def sample_pdf(x0, x1, pdf, n_samples):
  rv = TruncatedRV(x0=x0, x1=x1, pdf=pdf)
  return rv._rvs(size=n_samples)

def load_ir_model(path=IR_MODEL_PATH):
  with open(path, 'rb') as f:
    model = pickle.load(f)
  return model

def get_histogram(question) -> pd.DataFrame:
  """Metaculus and community pdfs over normalized range."""
  h = pd.DataFrame(question.prediction_histogram)
  h.columns = ['x', 'm', 'c']
  return h

def get_community_cdf(question):
  h = get_histogram(question)
  cdf = h['c'].cumsum().values / (h.shape[0] - 1)
  return cdf

def sample_recalibrated(question, model, n_samples=N_SAMPLES):
  ccdf = get_community_cdf(question)
  model.fit(np.linspace(0, 1, len(ccdf)), ccdf)
  rcdf = lambda x: model.predict(x) if hasattr(x, '__len__') else model.predict([x])[0]
  scale = question.data['possibilities']['scale']
  return (scale['max'] - scale['min']) * sample_cdf(x0=0, x1=1, cdf=rcdf, n_samples=n_samples) + scale['min']

def submit_recalibrated(question, model=IR_MODEL_PATH, n_samples=N_SAMPLES, dry_run=True):
  if is_open(question) and is_continuous(question):
    print("\tLoading model...")
    if isinstance(model, str):
      model = load_ir_model(model)
    else:
      assert hasattr(model, 'predict')

    print("\tSampling...")
    samples = sample_recalibrated(question, model, n_samples)
    
    if not dry_run:
      print("\tSubmitting")
      question.submit_from_samples(samples)
    else:
      comm_samples = np.array([question.sample_community() for _ in range(n_samples)])
      print("\tSaving to file...")
      np.save(f"samples/{question.id}", np.vstack([samples, comm_samples]).T)


if __name__=='__main__':

  # Login and load questions
  metaculus = ergo.Metaculus()
  metaculus.login_via_username_and_password(
    username=USERNAME,
    password=PASSWORD
)

  print("Fetching questions...")
  qs = []
  for cat in CATS:
      qs.extend(metaculus.get_questions(cat=cat, pages=N_PAGES))
  print("Done!\n")

  # Submit
  for q in qs:
    print(f"{q.id}: {q.name}")
    if int(q.id) in MANUAL:
      print("\tSkipping...")
    try:
        submit_recalibrated(q, model=IR_MODEL_PATH, n_samples=N_SAMPLES, dry_run=False)
    except Exception as e:
        # print(e)
        # continue
        raise