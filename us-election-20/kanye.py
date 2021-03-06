import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import dirichlet
import pycountry
from functools import reduce

d = pd.read_csv("538/presidential_state_toplines_2020.csv")
d = d[(d.model=='polls-plus') & (d.modeldate=="10/30/2020")]

shares = d[['state']+[c for c in d.columns if 'voteshare' in c and ('hi' not in c and 'lo' not in c)]].dropna(axis=1).set_index('state')

p = lambda alpha0, n_samples: shares.apply(lambda row: sum(map(lambda r: abs(r[0]-r[1]) < r[2], dirichlet(alpha0*row).rvs(n_samples)))/n_samples, axis=1)

p_any = lambda alpha0, n_samples: 1-reduce(lambda x, y: x*y, 1-p(alpha0, n_samples))

# TODO (D, R) ~ Normal in 2D with [10, 90] percentiles defined in transverse and longitudinal directions by simple geometry taking a rectangle of vertices (x_lo, y_lo), (x_lo, y_hi), etc.