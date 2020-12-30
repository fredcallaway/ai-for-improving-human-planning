import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib as plt
import itertools as it
from analysis_utils import *

%load_ext rpy2.ipython

# %% ==================== SETUP ====================

VERSION = 'RTT-1.1'
figs = Figures(path=f'figs/{VERSION}')
figure = figs.plot
show = figs.show
write_tex = TeX(path=f'stats/{VERSION}').write
fb_order = ['none', 'action', 'meta']

def write_percent(name, x):
    write_tex(name, f'{x * 100:.1f}\\%')

def read(stage, name):
    loader = pd.read_csv if name.endswith('csv') else pd.read_json
    df = loader(f'../data/stage{stage}/{VERSION}/{name}')
    df = drop_nan_cols(df).set_index('wid')
    parse_json(df)
    return df

participants = read(1, 'participants.csv').rename(columns={'completed': 'completed_stage1'})
survey = read(1, 'survey.csv')
stage1 = read(1, 'trials.csv')
stage2 = read(2, 'trials.json')
bonus = read(2, 'bonus.csv').bonus

# Drop participants who didn't finish both sections
# Note: they are still in the participants dataframe for computing retention rates
participants['completed_stage2'] = (stage2.reset_index().wid.value_counts() == 8)
participants.completed_stage2 = participants.completed_stage2.fillna(False)
pdf = participants.query('completed_stage2').copy()
keep = list(pdf.index)
stage1 = stage1.query('wid == @keep').copy()
stage2 = stage2.query('wid == @keep').copy()

stage1['feedback'] = pdf.feedback
stage2['feedback'] = pdf.feedback
pdf['bonus'] = bonus
stage2['n_click'] = stage2.reveals.apply(len)
pdf['stage1_n_click'] = stage1.groupby('wid').n_clicks.mean()
pdf['stage2_n_click'] = stage2.groupby('wid').n_click.mean()


# %% ==================== DEMOGRAPHICS ====================

def regularize_gender(s):
    d = {
        'man': 'male',
        'woman': 'female',
        'f': 'female',
        'm': 'male',
    }
    s = s.lower().strip()
    return d.get(s, s)

# maybe need for age


def get_demographics():
    s = survey.loc[survey.responses.apply(len) == 3]
    age = s.responses.apply(get('Q1')).apply(excepts(ValueError, int, lambda _: None))
    gender = s.responses.apply(get('Q2')).apply(regularize_gender)
    return age, gender

age, gender = get_demographics()

write_tex('mean-age', f'{age.mean():.2f}')
write_tex('min-age', str(age.min()))
write_tex('max-age', str(age.max()))
write_tex("N-female", str(gender.value_counts()['female']))
write_tex("N-total", len(participants))

N = participants.query('completed_stage1').groupby(['completed_stage2', 'feedback']).apply(len)

for fb in fb_order:
    write_tex(f'N-drop-{fb}', N[False, fb])
    write_percent(f'drop-rate-{fb}', N[False, fb] / N[:, fb].sum())
write_percent('drop-rate', N[False].sum() / N.sum())
write_tex('return-N', N[True].sum())

write_tex('mean-bonus', f'\\${bonus.loc[pdf.index].mean():.2f}')

N.sum()
# %% ==================== TRANSFER PERFORMANCE ====================

def plot_transfer(outcome):
    sns.swarmplot('feedback', outcome, data=pdf, palette=palette, alpha=0.5, order=fb_order)
    sns.pointplot('feedback', outcome, data=pdf, palette=palette, order=fb_order, 
                  scale=1, capsize=0.1, markers='o')
    plt.xlabel('Feedback')
    plt.ylabel(nice_names[outcome])
    # test = 'Test' if EXPERIMENT == 1 else 'Transfer'
    reformat_labels()

figure(outcome='bonus')(plot_transfer)
figure(outcome='stage2_n_click')(plot_transfer)

# %% ==================== BACKWARD PLANNING ====================

leaves = {3,4,7,8,11,12}

def stage1_backward(clicks):
    if not clicks:
        return False
    first = clicks[0]
    return first in leaves

stage1['backward'] = stage1.clicks.apply(stage1_backward)

pdf['stage1_backward'] = stage1.groupby('wid').backward.mean()
pdf.groupby('feedback').stage1_backward.mean()

# %% --------

def learning_curve(var):
    df = stage1.copy()
    df.trial_index += 1
    sns.lineplot('trial_index', var, hue='feedback', 
                 data=df, hue_order=fb_order, palette=palette)
    plt.ylabel(nice_names[var])
    plt.xlabel('Trial Number')
    plt.gca().grid(axis='x')
    plt.xticks([1, *range(5, 31, 5)])
    plt.xlim(df.trial_index.min()-0.5, df.trial_index.max()+0.5)
    reformat_legend()

figure(var='backward')(learning_curve)

# %% ====================  ====================


import networkx as nx

def make_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def get_goal(G):
    goal = [k for (k, v) in G.out_degree if v == 0]
    assert len(goal) == 1
    return goal[0]

def click_distance(row):
    d = nx.shortest_path_length(row.graph.to_undirected(), source=row.goal)
    return [d[i] for (i, v) in row.reveals]


stage2['graph'] = stage2.edges.apply(make_graph)
stage2['goal'] = stage2.graph.apply(get_goal)
stage2['click_distance'] = stage2.apply(click_distance, axis=1)
stage2['click_near_goal'] = stage2.click_distance.apply(lambda x: x[0] == 1 if x else False)
pdf['stage2_backward'] = stage2.groupby('wid').click_near_goal.mean()
pdf.groupby('feedback').stage2_backward.mean()
# %% --------

@figure()
def transfer_backward():
    sns.scatterplot('stage1_backward', 'stage2_backward', data=pdf, hue='feedback', 
                    hue_order=fb_order, palette=palette)
    plt.xlabel('Backward Planning in Training Task')
    plt.ylabel('Backward Planning in Transfer Task')
    reformat_legend()
    plt.xlim(-0.05,1.05); plt.ylim(-0.05, 1.05)

figs.browse()
# %% --------

practice = stage2.bonus.groupby('wid').apply(lambda x: x.iloc[0])
stage2.bonus.groupby('wid').sum() - practice


# %% --------
%%R -i pdf
library(lme4)
library(lmerTest)
m = lmer(bonus ~ n_click + transfer_backward + (1|wid), data=pdf)
summary(m)
# %% --------



bonus = pd.read_csv(f'data/processed/{version}/bonus.csv').set_index('wid').bonus
pdf = pd.DataFrame(bonus)
for k in 'n_click click_near_goal'.split():
    pdf[k] = df.groupby('wid')[k].mean()
print(len(pdf), 'participants')
savefig = Figures(f'figs/{version}', dpi=200).savefig




