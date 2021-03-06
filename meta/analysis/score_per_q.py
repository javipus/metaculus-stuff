# TL;DR: no correlation b/w rank and points per question, but strong correlation b/w rank and # predictions
# suggests top (say) 10 performers aren't better than the next 90, only more engaged

import pandas as pd
import seaborn as sns
import os, sys
sys.path.append(os.path.abspath('..'))

from scrape_user_stats import safe_json_load
sns.set_style('darkgrid')

# Load data
df = pd.read_csv('../db/top_users.csv')
stats = safe_json_load('../db/stats.json')
users = safe_json_load('../db/users.json')

# Arrange data
top_ids = [k for k, v in users.items() if v['username'] in df.Player.values]
tq = pd.Series({users[tid]['username']: stats[tid]['predictions']['resolved'] for tid in top_ids})
ts = df[['Player', 'Score']].set_index('Player').squeeze()
tq = tq.rename('Questions')
d = pd.DataFrame([ts,tq]).T

d['Score']
d['Score'].astype(int)
d.Score = d['Score'].astype(int)
d.Questions
d.Questions = d.Questions.astype(int)
ax = sns.regplot(data=d, x='Questions', y='Score')
ax.get_figure().show()

# plt.show()
# sns.regplot(data=d, x='Questions', y='Score')
# plt.show()
# get_ipython().run_line_magic('pinfo', 'sns.regression')
# (d.Score / d.Questions).describe()
# df
# df.set_index('Player')
# pd.Series(df.Player.index, index=df.Player)
# pd.Series(df.Player.index, index=df.Player).to_dataframe
# pd.Series(df.Player.index, index=df.Player).to_DataFrame
# rk = pd.Series(df.Player.index, index=df.Player)
# d
# d['Rank'] = rk
# d
# d['score_per_q'] = d.Score / d.Questions
# sns.scatterplot(data=d, x='Rank', y='score_per_q')
# plt.show()
# sns.scatterplot(data=d, x='Rank', y='score_per_q')
# plt.tight_layout()
# plt.show()
