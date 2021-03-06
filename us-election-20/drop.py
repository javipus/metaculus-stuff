# coding: utf-8
import pandas as pd
d = pd.read_csv('pres_pollaverages_1968-2016.csv')
dn = d[d.state == 'National']
dn.modeldate = pd.to_datetime(dn.modeldate)
dn = dn[dn.modeldate.apply(lambda d: d.month) >= 6]
ret = {}
for k, v in dn.groupby(['cycle', 'candidate_name']):
    ret[k] = v.pct_estimate.rolling(window=15, center=True).apply(lambda x: (x.iloc[0] - x.min())).max() > 5

print(f"{sum(ret.values())} / {len(ret)} = {sum(ret.values()) / len(ret)}")