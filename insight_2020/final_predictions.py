# coding: utf-8
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
ps = np.array([85, 75, 77, 78, 43, 85, 75, 93, 90, 33, 70, 15, 9, 20]) / 100
pd.Series(np.array([bernoulli(p).rvs(int(1e6)) == np.round(p) for p in ps]).sum(axis=0)).describe() # 11 right on average