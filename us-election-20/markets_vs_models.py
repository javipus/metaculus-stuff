import pandas as pd
from itertools import product
from numpy import log2, vectorize

xlogy = vectorize(lambda x, y: 0 if (x==y==0) else x*log2(y))
brier_score = lambda p, r: (p-r)**2
log_score = lambda p, r: -(xlogy(r, p) + xlogy(1-r, 1-p))

score = log_score

d = pd.read_csv('pivs358vsecon.csv')
r = d['result']
d = d.drop(columns=['result'])

print('Best guess:')
print(d.apply(lambda p: score(p,r)).mean(axis=0) * 100)

outstanding = 'PA', 'GA', 'NC', 'MI', 'WI', 'NV', 'AZ'
scenarios = list(product((0,1), repeat=len(outstanding)))

briers = []
for scenario in scenarios:
    _result = r.copy()
    _result[outstanding] = scenario
    briers += [d.apply(lambda p: score(p,_result)).mean(axis=0)]

print('')
print('Average over all scenarios [uniform distribution]:')
print(pd.concat(briers, axis=1).mean(axis=1) * 100)