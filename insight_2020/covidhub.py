import numpy as np
import pandas as pd

current = 8.86e6
days = 25 + 30 + 2
cutoff = 28 * (12e6 - current) / days
n_samples = int(1e6)

get_range = lambda vs, lo, hi: [(lo, vs[0])] + [(a,b) for a,b in zip(vs[:-1], vs[1:])] + [(vs[-1], hi)] # [lo] + [list(range(a,b)) for a, b in zip(vs[:-1], vs[1:])] + [hi]
get_pdf = lambda qs: np.diff([0] + qs + [1])

date = "2020-10-26"
url = f"https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-processed/COVIDhub-ensemble/{date}-COVIDhub-ensemble.csv"
df = pd.read_csv(url)

def sample_week(df, n_week):
    d = df[(df.target==f'{n_week} wk ahead inc case') & (df.type == 'quantile') & (df.location == 'US')][['quantile', 'value']].sort_values(by='quantile')

    qs, vs = d['quantile'].tolist(), d.value.tolist()
    rg = get_range(vs, lo=0, hi=None)
    pdf = get_pdf(qs)
    assert len(rg) == len(pdf), f"{len(rg)} != {len(pdf)}"

    return [np.random.randint(*rg[idx]) for idx in np.random.choice(len(rg), p=pdf, size=n_samples)]

samples = np.array([sample_week(df, wk) for wk in (1,2,3,4)])

# prob each week is above threshold ranges from 90 to 96 % so im taking 93
# print(f"p = {100*samples.apply(lambda x: x>cutoff).sum() / n_samples}%")