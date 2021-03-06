import pandas as pd
import seaborn as sns
import pycountry

e = pd.read_csv('economist/site_data/state_averages_and_predictions_topline.csv')[['state', 'projected_win_prob']].set_index('state')

d = pd.read_csv('538/presidential_state_toplines_2020.csv')
f = d[(d.model=='polls-plus') & (d.modeldate=='10/17/2020')][['state', 'winstate_chal']].set_index('state')

e['state'] = [pycountry.subdivisions.get(code=f"US-{code.split('-')[0]}").name for code in e.index]
e = e.set_index('state')

d = e.join(f)
d.columns = ['econ', '538']

print("Brier:")
print(((d-d.round())**2).sum())

ax = sns.scatterplot(data=d, x='538', y='econ')
ax.plot([0,1], [0,1], ls='--', c='gray')
ax.plot([0.5,0.5], [0,1], ls='--', c='gray')
ax.plot([0,1], [0.5,0.5], ls='--', c='gray')
ax.get_figure().show()