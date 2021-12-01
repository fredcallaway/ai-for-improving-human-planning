import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib as plt
import itertools as it
from new_analysis_utils import *
from figure import Figures
import os

%load_ext rpy2.ipython

# %% ==================== SETUP ====================

VERSION = 'RTT-2.1'
# VERSION = 'RTT-3.4'
tmp = f'tmp/{VERSION}'
os.makedirs(tmp, exist_ok=True)
figs = Figures(path=f'figs/{VERSION}')
figure = figs.figure
show = figs.show
figs.watch()

write_tex = TeX(path=f'stats/{VERSION}').write
fb_order = ['none', 'action', 'meta']

def write_percent(name, x):
    write_tex(name, f'{x * 100:.1f}\\%')

def read(stage, name):
    loader = pd.read_csv if name.endswith('csv') else pd.read_json
    if VERSION in ('final', 'RTT-2.1'):
        df = loader(f'../data/5/stage{stage}/{name}')
    else:
        df = loader(f'../data/pilot/stage{stage}/{VERSION}/{name}')
    df = drop_nan_cols(df).set_index('wid')
    parse_json(df)
    return df

participants = read(1, 'participants.csv').rename(columns={'completed': 'completed_stage1'})
survey = read(1, 'survey.csv')
stage1 = read(1, 'trials.csv')
stage2 = read(2, 'trials.json')

# %% --------
bonus = read(2, 'bonus.csv')
ids = read(2, 'identifiers.csv')
try:
    stage2_start = read(2, 'params.json').start_time
    # participants['stage2_ontime'] = stage2_start.dt.day == 27
    participants['stage2_ontime'] = participants.stage2_ontime.astype(bool)
    print(participants.query('~stage2_ontime').feedback.value_counts())
except AttributeError:
    pass
# Drop participants who didn't finish both sections
# Note: they are still in the participants dataframe for computing retention rates
participants['completed_stage2'] = (stage2.reset_index().wid.value_counts() == 8)
participants.completed_stage2 = participants.completed_stage2.fillna(False)
# pdf = participants.query('completed_stage2 and stage2_ontime').copy()
pdf = participants.query('completed_stage2').copy()

keep = list(pdf.index)
stage1 = stage1.query('wid == @keep').copy()
stage2 = stage2.query('wid == @keep').copy()

stage1['feedback'] = pdf.feedback
stage2['feedback'] = pdf.feedback
pdf['bonus'] = bonus.bonus
stage2['n_click'] = stage2.reveals.apply(len)
pdf['stage1_n_click'] = stage1.groupby('wid').n_clicks.mean()
pdf['stage2_n_click'] = stage2.groupby('wid').n_click.mean()
# %% --------
participants.groupby('feedback').completed_stage1.mean()
participants.groupby('feedback').completed_stage2.mean()


participants.query('feedback == "video"').completed_stage2.sum()
participants.query('feedback == "video"').completed_stage1.sum()

# %% ==================== SCRATCH ====================

stage2['no_click'] = stage2.n_click == 0

sns.lineplot('trial_index', 'no_click', hue='feedback', data=stage2, palette=palette)
show()

from statsmodels.formula.api import ols
stage2['time_cost'] = (350 - stage2.route_cost) / 1000 - stage2.bonus
model = ols('time_cost ~ n_click', data=stage2).fit()

cost_per_click = model.params.n_click
stage2.time_cost

(stage2.bonus * 1000).mean()

# %% --------
sns.lineplot('n_click', 'time_cost', data=stage2)
plt.plot(np.arange(0, 10), model.predict(pd.DataFrame({'n_click': np.arange(0, 10)})), c='r')
show()
# %% --------
stage2['no_click'] = stage2.n_click == 0
# rdf = stage2[['feedback', 'no_click']]
pdf['no_click'] = stage2.groupby('wid').no_click.mean()
pdf.feedback.value_counts()
stage2.groupby('feedback').no_click.mean().round(2)
# %% --------
%%R -i pdf
pdf$feedback = relevel(factor(pdf$feedback), ref="none")
summary(lm(bonus ~ feedback, data=pdf))

# %% ==================== TRANSFER PERFORMANCE ====================

def plot_transfer(outcome):
    sns.swarmplot('feedback', outcome, data=pdf, palette=palette, alpha=0.5, order=fb_order)
    sns.pointplot('feedback', outcome, data=pdf, palette=palette, order=fb_order, 
                  scale=1, capsize=0.1, markers='o')
    plt.xlabel('Feedback')
    plt.ylabel(nice_names[outcome])
    # test = 'Test' if EXPERIMENT == 1 else 'Transfer'
    reformat_labels()

@figure()
def plot_bonus():
    plot_transfer('bonus')

@figure()
def plot_stag2_n_click():
    plot_transfer('stage2_n_click')


# %% ==================== BACKWARD PLANNING ====================

leaves = {3,4,7,8,11,12}

def stage1_backward(clicks):
    if not clicks:
        return False
    first = clicks[0]
    return first in leaves

def stage1_ever_goal(clicks):
    if not clicks:
        return False
    first = clicks[0]
    return first in leaves

# stage1.clicks.apply(lambda cs: any(c in leaves for c in cs)).mean()
stage1['backward'] = stage1.clicks.apply(stage1_backward)
stage1['ever_goal'] = stage1.clicks.apply(stage1_backward)
pdf['stage1_backward'] = stage1.groupby('wid').backward.mean()
pdf.groupby('feedback').stage1_backward.mean()

stage1.query('trial_index == 0').backward.mean()
# %% --------

import networkx as nx

def make_graph(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def get_goals(G):
    goal = [k for (k, v) in G.out_degree if v == 0]
    return set(goal)

def click_goal_first(row):
    if not row.reveals:
        return False
    else:
        goals = get_goals(row.graph)
        return row.reveals[0][0] in goals

def ever_goal(row):
    if not row.reveals:
        return False
    else:
        goals = get_goals(row.graph)
        return any(x[0] in goals for x in row.reveals)

def num_goal(row):
    if not row.reveals:
        return 0
    else:
        goals = get_goals(row.graph)
        return sum(x[0] in goals for x in row.reveals)

stage2['graph'] = stage2.edges.apply(make_graph)
stage2['backward'] = stage2.apply(click_goal_first, axis=1)
stage2['ever_goal'] = stage2.apply(ever_goal, axis=1)
stage2['num_goal'] = stage2.apply(num_goal, axis=1)

pdf['stage2_backward'] = stage2.groupby('wid').backward.mean()
pdf['stage2_ever_goal'] = stage2.groupby('wid').ever_goal.mean()
pdf['stage2_num_goal'] = stage2.groupby('wid').num_goal.mean()

# %% --------

pdf['bonus'] = bonus.bonus
bonus.bonus

pdf.bonus
pdf.query('stage2_backward >= 0.75').bonus


# %% --------
stage2['no_reveal'] = stage2.n_click == 0
X = stage2.groupby('no_reveal').route_cost.mean()
X[True] - X[False]
# %% --------
non_airport = np.mean([30, 35, 40, 45])
airport = np.mean([40, 80, 120, 160])

# %% --------
import time

def all_paths(row):
    def rec(path):
        children = row.graph[path[-1]]
        if not children:
            yield path
        for i in children:
            yield from rec(path + [i])
    yield from rec([0])

def path_value(row, path):
    revealed = {i: v for i, v in row.reveals}
    goals = get_goals(row.graph)
    cost = 0
    for i in path[1:]:
        if i in revealed:
            cost += revealed[i]
        elif i in goals:
            cost += airport
        else:
            cost += non_airport
    return cost

def expected_path_costs(row):
    for p in all_paths(row):
        yield path_value(row, p)

def expected_cost(row):
    return path_value(row, row.route)

def action_loss(row):
    return expected_cost(row) - min(expected_path_costs(row))

def inverse_action_loss(row):
    return expected_cost(row) - max(expected_path_costs(row))

def pick_worst_path(row):
    return expected_cost(row) == max(expected_path_costs(row))

def choice_quantile(row):
    # sorted(list(expected_path_costs(row)))
    chosen_cost = expected_cost(row)
    return np.mean([cost >= chosen_cost for cost in expected_path_costs(row)])

stage2['choice_quantile'] = stage2.apply(choice_quantile, axis=1)

def inverse_choice_quantile(row):
    # sorted(list(expected_path_costs(row)))
    chosen_cost = expected_cost(row)
    return np.mean([cost <= chosen_cost for cost in expected_path_costs(row)])

stage2['inverse_choice_quantile'] = stage2.apply(inverse_choice_quantile, axis=1)


pv = stage2.apply(lambda row: np.var(list(expected_path_costs(row))), axis=1)
(pv == 0).mean()

# stage2['act_loss'] = stage2.apply(action_loss, axis=1)
# stage2['inv_act_loss'] = stage2.apply(inverse_action_loss, axis=1)
# stage2['picked_worst'] = stage2.apply(pick_worst_path, axis=1)
# %% --------

stage2.groupby('wid').choice_quantile.mean().sort_values().plot()
plt.xticks([]); plt.xlabel('participant'); plt.ylabel('Pr(cost ≥ chosen_cost)'); show()

stage2.groupby('wid').inverse_choice_quantile.mean().sort_values().plot()
plt.xticks([]); plt.xlabel('participant'); plt.ylabel('Pr(cost ≤ chosen_cost)'); show()

stage2.groupby('wid').act_loss.mean().sort_values().plot()
plt.xticks([]); plt.xlabel('participant'); plt.ylabel('choice loss'); show()

stage2.groupby('wid').inv_act_loss.mean().sort_values().plot(); show()
stage2.groupby('wid').picked_worst.sum().sort_values().plot(); 
plt.xticks([]); plt.xlabel('participant'); plt.ylabel('N choose worst')
show()

pdf['picked_worst'] = stage2.groupby('wid').picked_worst.mean()
sns.catplot('feedback', 'picked_worst', data=pdf, kind='violin'); show()

# %% --------

stage2.groupby('trial_index').choice_quantile.mean().plot(); show()
stage2.groupby('trial_index').picked_worst.mean().plot(); show()
# %% --------
d = stage2.groupby('wid').choice_quantile.mean() -\
   stage2.groupby('wid').inverse_choice_quantile.mean()

d.sort_values().plot()
plt.xticks([]); plt.xlabel('participant')
plt.ylabel('Pr(cost ≥ chosen_cost) -\nPr(cost ≤ chosen_cost)')
plt.axhline(0, c="black")
(d < 0).mean()

show()
# %% --------

def expected_cost(row):
    cost = 0
    goals = get_goals(row.graph)
    revealed = {i: v for i, v in row.reveals}
    for i in row.route[1:]:
        if i in revealed:
            cost += revealed[i]
        elif i in goals:
            cost += airport
        else:
            cost += non_airport
    return cost

stage2 = stage2.dropna()
stage2['expected_cost'] = stage2.apply(expected_cost, axis=1)

stage2['pv0'] = pv == 0


# %% --------
pdf['expected_cost'] = stage2.groupby('wid').expected_cost.mean()
pdf['route_cost'] = stage2.groupby('wid').route_cost.mean()

pdf.route_cost.std()
pdf.expected_cost.std()

# %% --------
stage2['strategy'] = stage2.apply(lambda x: 
    'no plan' if x.n_click == 0 else
    'backward' if x.backward else
    'lots' if x.n_click > 3 else
    'other', axis = 1
)
X = stage2.groupby('feedback').strategy.value_counts().unstack().T
(X / X.sum()).T

# %% --------
sns.pointplot(stage2.n_click, stage2.route_cost - stage2.expected_cost)
show()
# %% --------
%%R -i pdf
pdf$feedback = relevel(factor(pdf$feedback), ref="none")
# summary(lm(expected_cost ~ feedback, data=pdf))
summary(lm(route_cost ~ feedback, data=pdf))

# %% --------
pdf.groupby('feedback').stage2_backward.mean().round(2)

pdf['stage2_backward_one'] = stage2.query('trial_index == 0').backward
pdf.groupby('feedback').stage2_backward_one.mean().round(2)

pdf.groupby('feedback').stage2_num_goal.mean()
pdf.groupby('feedback').stage2_ever_goal.mean()

pdf.groupby('feedback').stage2_num_goal.mean()
pdf.groupby('feedback').stage2_ever_goal.mean()

# %% --------
%R -i pdf
%R print(summary(lm(stage2_num_goal ~ feedback, data=pdf)));
x

# %% --------

def learning_curve(var):
    # df = {1: stage1, 2: stage2}[stage].copy()
    df = stage2.copy().query('not no_reveal')
    df.trial_index += 1
    sns.lineplot('trial_index', var, hue='feedback', 
                 data=df, hue_order=fb_order, palette=palette)
    plt.ylabel(nice_names[var])
    plt.xlabel('Trial Number')
    plt.gca().grid(axis='x')
    plt.xticks([1, *range(5, 31, 5)])
    plt.xlim(df.trial_index.min()-0.5, df.trial_index.max()+0.5)
    reformat_legend()

# figure(stage=1, var='backward')(learning_curve)
figure(var='backward')(learning_curve)

# %% --------
print(pdf[['stage1_backward', 'stage2_backward']].corr())
stage2.to_csv(f'{tmp}/stage2.csv')
rdf.to_csv(f'{tmp}/participants.csv', index=False)


# %% --------
g = stage2.groupby(['feedback', 'trial_index']).backward.mean()
plt.plot(g['meta'] - g['none'])
show()

# %% --------
for w, d in stage2.backward.groupby('wid'):
    plt.plot(list(d), color='k', alpha=0.3)
show()

np.diff([1, 1, 0, 0])
stage2.backward.groupby('wid').apply(lambda d: sum(np.diff(d.astype(int)))).value_counts()
# %% --------
b = stage2.backward.astype(int)
diff = b - b.shift(1)
stage2['DB'] = diff
stage2.loc[lambda row: row.trial_index == 0, 'DB'] = np.nan

stage2['D1'] = stage2.DB == 1
stage2['D0'] = stage2.DB == 0
stage2['DN'] = stage2.DB == -1
sns.lineplot('trial_index', 'DN', data=stage2.query('trial_index > 0'), label='Backward → Forward')
sns.lineplot('trial_index', 'D1', data=stage2.query('trial_index > 0'), label='Forward → Backward')
# sns.lineplot('trial_index', 'D0', data=stage2, label='No Change')
plt.xlabel('Trial')
plt.ylabel('Proportion')
show()
(stage2.DB.dropna() != 0).groupby('wid').apply(sum).value_counts()

# %% --------
def first_backward(d):
    if not any(d):
        return -1
    return np.argmax(list(d))

pdf['first_backward'] = stage2['backward'].groupby('wid').apply(first_backward)
pdf.groupby('feedback').first_backward.value_counts().sort_index()   

for v, d in pdf.groupby('feedback'):
    d.first_backward.value_counts().sort_index().plot.bar(label=v)
plt.legend()
show()

x = {fb: d.pivot(None, 'trial_index', 'backward').values
 for fb, d in stage2.groupby('feedback')}
pd.to_pickle(x, 'backward.pkl')

# %% --------
from matplotlib.ticker import FormatStrFormatter

x = {fb: d.pivot(None, 'trial_index', 'backward').values
 for fb, d in stage2.groupby('feedback')}

@figure()
def bas_plot():
    fig,ax=plt.subplots(figsize=(6,6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    for i,key in enumerate(x.keys()):
        y = x[key][np.sum(x[key],axis=1)>0]
        print(key,(len(x[key])-len(y))/len(x[key]))
        ax.bar(x=np.arange(1,9)+i/4-1/4,height=np.bincount(np.argmax(y,axis=1),minlength=8)/len(y),
               width=1/4,label=key, color=palette[key])
    ax.set_xticks(range(1,9))
    ax.set_xlim([0.5,8.5])
    ax.set_xlabel('First backwards\nplanning trial')
    ax.set_ylabel('Proportion')
    plt.grid()
    ax.legend()
# %% ====================  ====================

@figure()
def transfer_backward(pdf=pdf):
    pdf = pdf.copy()
    pdf.stage1_backward += 0.05* (np.random.rand(len(pdf)) - 0.5)
    pdf.stage2_backward += 0.05* (np.random.rand(len(pdf)) - 0.5)
    sns.scatterplot('stage1_backward', 'stage2_backward', data=pdf, hue='feedback', 
                    hue_order=fb_order, palette=palette, alpha=0.5)
    
    g = pdf.groupby('feedback')
    fbs = g.stage1_backward.mean().index
    plt.scatter(g.stage1_backward.mean(), g.stage2_backward.mean(), 
        color=[palette[fb] for fb in fbs], s=100)


    plt.xlabel('Backward Planning in Training Task')
    plt.ylabel('Backward Planning in Transfer Task')
    reformat_legend()
    plt.legend(loc=(1,0.5))
    plt.xlim(-0.05,1.05); plt.ylim(-0.05, 1.05)

# %% --------
@figure()
def backward_bonus():
    sns.regplot('stage2_backward', 'bonus', data=pdf)
    plt.xlabel('Backward Planning in Transfer Task')


@figure()
def backward_bonus_trial():
    sns.barplot('backward', 'bonus', data=stage2)

# %% ==================== STATS ====================
rdf = stage2.reset_index()[['wid', 'feedback', 'no_reveal']]
rdf.no_reveal = rdf.no_reveal.astype(int)
# %% --------
%%R -i rdf
library(lme4)
m = glmer(no_reveal ~ feedback + (1|wid), family=binomial, data=rdf)
summary(m)
# %% --------
pdf['no_reveal'] = stage2.no_reveal.groupby('wid').mean()
rdf = pdf.reset_index()


# %% --------
%R -i rdf
%R summary(lm(bonus ~ no_reveal, data=rdf)) %>% print;
%R library(dplyr)
%R rdf$feedback = relevel(factor(rdf$feedback), ref="none")

%R summary(lm(bonus ~ feedback, data=rdf)) %>% print;
%R summary(lm(stage2_backward ~ feedback, data=rdf)) %>% print;
%R summary(glm(stage2_backward_one ~ feedback, family=binomial, data=rdf)) %>% print;

%R print(summary(lm(bonus ~ stage2_n_click + stage2_backward, data=rdf)));
# summary(m)

# %% --------

bonus = pd.read_csv(f'data/processed/{version}/bonus.csv').set_index('wid').bonus
pdf = pd.DataFrame(bonus)
for k in 'n_click click_near_goal'.split():
    pdf[k] = df.groupby('wid')[k].mean()
print(len(pdf), 'participants')
savefig = Figures(f'figs/{version}', dpi=200).savefig




