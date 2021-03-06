import json, copy, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ergo

from KEYS import USERNAME, PASSWORD

N_PAGES = 2

search_queries = [
  "series--maximum-likelihood",
  "series--hill-climbing",
]

def get_histogram(question):
  """Metaculus and community pdfs over normalized range."""
  h = pd.DataFrame(question['prediction_histogram'])
  h.columns = ['x', 'm', 'c']
  return h

def recalibrate_question(ergo_q, fitted_ir):
  # save old histogram in question - you're gonna overwrite it later and don't wanna lose it
  setattr(ergo_q, 'community_histogram', ergo_q.data['prediction_histogram'])
  h = get_histogram(ergo_q.data)
  interpolate = lambda x: [(x0 + x1) / 2 for x0, x1 in zip(x[:-1], x[1:])]
  pdf = [ergo_q.p_below] + h['c'].tolist() + [ergo_q.p_above]
  dx = 1 / (h.shape[0] - 1)
  cdf = np.cumsum(pdf)*dx
  new_cdf = fitted_ir.predict(cdf)
  new_pdf = interpolate(np.diff(new_cdf) / dx) # assumes p_below (p_above) concentrated in -dx (1+dx)
  # h_new = np.array([h['x'].values, h['m'].values, new_pdf]).T.tolist()
  h['c'] = new_pdf
  ergo_q.set_data('prediction_histogram', h.values.tolist())
  return ergo_q

def get_pc_and_p(recalibrated_question):
  return pd.DataFrame([
    np.array(recalibrated_question.community_histogram).T[2].tolist(),
    np.array(recalibrated_question.prediction_histogram).T[2].tolist(),
  ],
  index = ['c', 'p']
  ).T

def fetch_questions(search_query, qid_path=None, username=USERNAME, password=PASSWORD, n_pages=N_PAGES):
  
  metaculus = ergo.Metaculus()
  metaculus.login_via_username_and_password(
    username=username,
    password=password
  )

  if qid_path is not None:
    with open(qid_path, "r") as f:
      qids = [int(qid) for qid in f.readlines()]
    return [metaculus.get_question(qid) for qid in qids]
  
  qs = []
  
  for cat in search_query:
    qs.extend(metaculus.get_questions(cat=cat, pages=n_pages))
  return qs

with open("db/ir_model.pkl", "rb") as f:
  model = pickle.load(f)

is_resolved = lambda q: q.data['resolution'] is not None
is_open = lambda q: q.data.get('prediction_histogram') is not None

print("Fetching questions...")
qs = [recalibrate_question(q, model) for q in fetch_questions(search_queries, qid_path="qids.csv") if is_open(q)]
print("Done!")

fig, ax = plt.subplots(6, 6)
rng = np.linspace(0, 1, 201)
for (ax_, q) in zip(ax.flatten(), qs):
  x = np.array(q.community_histogram).T[2]
  y = np.array(q.prediction_histogram).T[2]
  ax_.plot(rng, x)
  ax_.plot(rng, y)
  ax_.set_title(q.id)
fig.show()