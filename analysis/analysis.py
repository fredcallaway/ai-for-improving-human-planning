import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib as plt
import itertools as it
from analysis_utils import *
import os

%load_ext rpy2.ipython

# %% ==================== SETUP ====================

VERSION = 'RTT-2.1'
tmp = f'tmp/{VERSION}'
os.makedirs(tmp, exist_ok=True)
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


# %% --------
# export = pd.read_csv('/Users/fred/heroku/tutor/data/prolific/export_RTT-2.0A.csv')
# export.participant_id = export.participant_id.apply(hash_id)
# export.set_index('participant_id', inplace=True)
# hours = export.time_taken / 3600
# hours = hours.loc[total.index]
# total = bonus.bonus + 4
# wage = (total / hours)
# %% --------

bonus = read(2, 'bonus.csv')
ids = read(2, 'identifiers.csv')
try:
    stage2_start = read(2, 'params.json').start_time
    participants['stage2_ontime'] = stage2_start.dt.day == 27
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
# write_tex("N-female", str(gender.value_counts()['female']))
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




