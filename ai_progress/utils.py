import json, pickle, datetime
import numpy as np
import pandas as pd

def formatter(question: dict):
  """Format continuous date questions"""
  fmt = question.get('possibilities', {}).get('format')
  if fmt == 'date':
    return pd.to_datetime
  elif fmt == 'num':
    return lambda x: x
  else:
    raise ValueError(f"Question format {fmt} unknown")

def get_range(question: dict):
  fmt = formatter(question)
  lo, hi = map(fmt, [question['possibilities']['scale'][k] for k in ('min', 'max')])
  return lo, hi

def get_resolution(question: dict):
  fmt = formatter(question)
  x_star = fmt(question['resolution'])
  return x_star

def get_norm_resolution(question: dict):
  lo, hi = get_range(question)
  x_star = get_resolution(question)
  return (x_star - lo) / (hi - lo)

def get_histogram(question: dict) -> pd.DataFrame:
  """Metaculus and community pdfs over normalized range."""
  h = pd.DataFrame(question['prediction_histogram'])
  h.columns = ['x', 'm', 'c']
  return h

def get_community_cdf(question: dict):
  h = get_histogram(question)
  cdf = h['c'].cumsum().values / (h.shape[0] - 1)
  return cdf

def get_uniform(question: dict):
  n_bins = len(get_histogram(question)) - 1
  p_above = 0 if is_high_closed(question) else .15
  p_below = 0 if is_low_closed(question) else .15
  return (1-p_above-p_below) / n_bins

def json_save(ob, fpath):
  with open(fpath, 'w') as f:
    json.dump(ob, f)

def json_load(fpath):
  with open(fpath, 'r') as f:
    d = json.load(f)
  return d

def pickle_save(ob, fpath):
  with open(fpath, 'wb') as f:
    pickle.dump(ob, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(fpath):
  with open(fpath, 'rb') as f:
    ob = pickle.load(f)
  return ob

def interpolate(y, x, closed):
  # TODO this function should be aware of open/closed bounds and return:
  # - y[0] (y[-1]) if closed and x < 0 (x > len(y)-1)
  # - p_below, p_above if open
  x_lo, x_hi = map(lambda f: int(f(x)), (np.floor, np.ceil))
  if x_hi > len(y)-1:
    return y[-1] if closed else 1-sum(y)/(len(y)-1)-y[0]
  if x_lo < 0:
    return y[0] if closed else 1-sum(y)/(len(y)-1)-y[-1]
  return y[x_lo] + x * (y[x_hi] - y[x_lo]) # / (x_hi - x_lo) = 1

# Question type
is_continuous = lambda q: q['possibilities']['type'] == 'continuous'
is_binary = lambda q: q.data['possibilities']['type'] == 'binary'

# Question open (not upcoming or closed or resolved)
is_open = lambda q:\
    datetime.datetime.strptime(q.data['publish_time'], '%Y-%m-%dT%H:%M:%SZ') <\
    datetime.datetime.utcnow() <\
    datetime.datetime.strptime(q.data['close_time'], '%Y-%m-%dT%H:%M:%SZ')

# Open or closed range
is_low_closed = lambda q: q.possibilities.get('low')!='tail'
is_high_closed = lambda q: q.possibilities.get('high')!='tail'

# Question resolved already and did so non-amgiguously
resolved_ok = lambda q: q['resolution'] not in (None, -1)

# I wanna train the model using only non-ambiguously resolved continuous questions
training_q = lambda q: is_continuous(q) and resolved_ok(q)

# TODO filter by date