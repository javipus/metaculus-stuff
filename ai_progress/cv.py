"""Cross-validation of my recalibration procedure."""

# import json, pickle
# import numpy as np
# import pandas as pd
# import seaborn as sns

from utils import json_load

# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_validate, cross_val_score
# from sklearn.isotonic import IsotonicRegression

# def formatter(question):
#   fmt = question.get('possibilities', {}).get('format')
#   if fmt == 'date':
#     return pd.to_datetime
#   elif fmt == 'num':
#     return lambda x: x
#   else:
#     raise ValueError(f"Question format {fmt} unknown")

# def get_range(question):
#   fmt = formatter(question)
#   lo, hi = map(fmt, [question['possibilities']['scale'][k] for k in ('min', 'max')])
#   return lo, hi

# def get_resolution(question):
#   fmt = formatter(question)
#   x_star = fmt(question['resolution'])
#   return x_star

# def get_norm_resolution(question):
#   lo, hi = get_range(question)
#   x_star = get_resolution(question)
#   return (x_star - lo) / (hi - lo)

# def interpolate(y, x, closed):
#   # TODO this function should be aware of open/closed bounds and return:
#   # - y[0] (y[-1]) if closed and x < 0 (x > len(y)-1)
#   # - p_below, p_above if open
#   x_lo, x_hi = map(lambda f: int(f(x)), (np.floor, np.ceil))
#   if x_hi > len(y)-1:
#     return y[-1] if closed else 1-sum(y)/(len(y)-1)-y[0]
#   if x_lo < 0:
#     return y[0] if closed else 1-sum(y)/(len(y)-1)-y[-1]
#   return y[x_lo] + x * (y[x_hi] - y[x_lo]) # / (x_hi - x_lo) = 1

# def get_histogram(question):
#   """Metaculus and community pdfs over normalized range."""
#   h = pd.DataFrame(question['prediction_histogram'])
#   h.columns = ['x', 'm', 'c']
#   return h

# def scoring_(p_star, pc_star, pu, N):
#   B = 20 * np.log2(1 + N / 30)
#   A = 40 + B
#   abs_sc = A * np.log(max(p_star / pu, .02))
#   rel_sc = B * np.log(max(p_star / pc_star, .02))
#   return abs_sc + rel_sc

# def score(question, p, tol=1e-3):
#   # TODO pc is added to pu then renormalized, see https://discord.com/channels/694850840200216657/694850840707596320/798616689918738453
#   # TODO pc is actually calculated by logistic best fit
#   # Normalized resolution
#   x_norm = get_norm_resolution(question)

#   # Is question range open or closed?
#   pc = get_histogram(question)['c'].values.tolist()
#   assert len(pc) == len(p)
#   n_bins = len(p) - 1
#   dx = 1 / n_bins
#   closed = abs(sum(pc)*dx - 1) < tol # for lack of anything better

#   # Community distribution
#   pc_star = interpolate(pc, x_norm, closed)

#   # Uniform distribution
#   pu = 1 / n_bins if closed else (1-.15) / n_bins
  
#   # Own prediction
#   p_star = interpolate(p, x_norm, closed)

#   # # predictions
#   N = question['number_of_predictions']

#   my_sc = scoring_(p_star, pc_star, pu, N)
#   comm_sc = scoring_(pc_star, pc_star, pu, N)
  
#   return my_sc - comm_sc

# def json_load(fpath):
#   with open(fpath, 'r') as f:
#     d = json.load(f)
#   return d

def generate_cv_splits(k, f, n_samples, seed=0, save_paths={}):
  np.random.seed(seed)
  idx = list(range(n_samples))
  choose = lambda idx, s: np.random.choice(idx, replace=False, size=s)
  train_idx = np.array([choose(idx, int(f*n_samples)) for _ in range(k)])
  test_idx = np.array([[i for i in idx if i not in row] for row in train_idx])
  if save_paths:
    np.save(save_paths['train_idx'], train_idx)
    np.save(save_paths['test_idx'], test_idx)
  return train_idx, test_idx

# def F_star(question):
#   """CDF evaluated at resolution"""
#   h = get_histogram(question)
#   return h[h['x'] <= question['resolution']]['c'].sum() / h.shape[0]

# def P_hat(questions):
#   """Empirical probability that resolution is indeed below CDF_community(resolution)"""
#   return lambda p: len([q for q in (questions.values() if isinstance(questions, dict) else questions) if F_star(q) <= p]) / len(questions)

# def calibration_dataset(questions):
#   """Generate pairs (CDF_community(resolution), empirical_prob(resolution <= CDF_community(resolution)) to run regression on"""
#   d = []
#   p_hat = P_hat(questions)
#   for q in (questions.values() if isinstance(questions, dict) else questions):
#     try:
#       fstar = F_star(q)
#       d.append((fstar, p_hat(fstar)))
#     except Exception as e:
#       print(f"Could not add question {q['id']} to calibration dataset")
#       print(e)
#   return d

# def fit(qs, save_paths={}):
#   cd = calibration_dataset(qs)
#   df = pd.DataFrame(cd, columns=['x', 'y'])
  
#   if save_paths:
#     np.save(save_paths['calibration'], cd)
  
#   model = IsotonicRegression(
#     y_min=0,
#     y_max=1,
#     increasing=True,
#     out_of_bounds='clip'
#     ).fit(df['x'], df['y'])
  
#   if save_paths:
#     with open(save_paths['model'], 'wb') as f:
#       pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

#   return model

# def predict(model, q):
#   return model.predict(get_histogram(q)['c'].values.tolist())

is_continuous = lambda q: q['possibilities']['type'] == 'continuous'
resolved_ok = lambda q: q['resolution'] not in (None, -1)
training_q = lambda q: is_continuous(q) and resolved_ok(q)
# TODO filter by date

def run_cv(qs_path, save_paths, filter_=training_q, k=5, f=.8, seed=0):
  # fmt_save_paths = {name: path.format(f=f, s=seed) for name, path in save_paths.items()}
  print("Loading questions...")
  qs = [q for q in json_load(qs_path) if filter_(q)]
  print("Done!\n")
  print("Generating splits...")
  n_samples = len(qs)
  train_idx, test_idx = generate_cv_splits(k, f, n_samples, seed) #, save_paths)
  print("Done!\n")
  print("Cross-validating...")
  for (i, (train_set, test_set)) in enumerate(zip(train_idx, test_idx)):
    print(f"Fold {i+1} of {k}...")
    curr_paths = {name: path.format(f=f, k=i, s=seed) for name, path in save_paths.items()}
    print("  Fitting model...")
    qs_train = [qs[idx] for idx in train_set]
    qs_test = [qs[idx] for idx in test_set]
    model = fit(qs_train, save_paths=curr_paths)
    print("  Predicting...")
    preds = [predict(model, q) for q in qs_test]
    print("  Scoring...")
    scores = [score(q, pred) for q, pred in zip(qs_test, preds)]
    q1, q2, q3 = pd.Series(scores).quantile([.25, .5, .75])
    print(f"  Scores [median, IQR]: {q2} [{q1}, {q3}]")
    np.save(curr_paths['scores'], scores)

if __name__=='__main__':

  RUN = True
  k = 5
  f = .8
  seed = 0

  qs_path = "db/training_questions.json"

  save_paths = {
    "calibration" : "cv/calibration_k_{k}_f_{f}_s_{s}.npy",
    "model"       : "cv/model_k_{k}_f_{f}_s_{s}.pkl",
    "train_idx"   : "cv/train_idx_k_{k}_f_{f}_s_{s}.npy",
    "test_idx"    : "cv/test_idx_k_{k}_f_{f}_s_{s}.npy",
    "scores"      : "cv/scores_k_{k}_f_{f}_s_{s}.npy",
  }

  if RUN:
    run_cv(qs_path, save_paths)

  scores = pd.DataFrame(np.array([np.load(save_paths['scores'].format(k=i, f=f, s=seed)) for i in range(k)])).T