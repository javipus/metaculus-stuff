import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import dirichlet
import pycountry

VAP = 257605088

d = pd.read_csv("538/presidential_national_toplines_2020.csv")
t = d[(d.model=='polls-plus') & (d.modeldate=="10/30/2020")][[c for c in d.columns if 'turnout' in c]]

print(100*t / VAP)