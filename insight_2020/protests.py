# coding: utf-8
import numpy as np
from scipy.stats import lognorm

# see https://www.metaculus.com/news/2020/10/20/results-of-2020-us-election-risks-survey/

# May-August deaths in protests
# * 2/3 since 2/3 the time b/w Nov 3 and Jan 21
data = (2/3) * np.array([23, 4, 1])

# estimators assuming data produced from a lognormal
scale = lambda data: (np.log((np.std(data) / np.mean(data)) ** 2 + 1))**(1/2)
loc = lambda data: np.log(np.mean(data)) - scale(data)**2 / 2

lognorm(s=1, loc=loc(data), scale=scale(data)).cdf(10)