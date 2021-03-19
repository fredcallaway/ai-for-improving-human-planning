import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

from utils import *
import shutil

# ---------- Data wrangling ---------- #

def mostly_nan(col):
    try:
        return col.apply(np.isnan).mean() > 0.5
    except:
        return False

def drop_nan_cols(df):
    return df[[name for name, col in df.iteritems()
               if not mostly_nan(col)]]

def query_subset(df, col, subset):
    idx = df[col].apply(lambda x: x in subset)
    return df[idx].copy()

def rowapply(df, f):
    return [f(row) for i, row in df.iterrows()]

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'[.:\/]', '_', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def to_camel_case(snake_str):
    return ''.join(x.title() for x in snake_str.split('_'))

def reformat_name(name):
    return re.sub('\W', '', to_snake_case(name))

class Labeler(object):
    """Assigns unique integer labels."""
    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label


# ---------- Loading data ---------- #

from glob import glob
import json
import ast
def parse_json(df):
    def can_eval(x):
        try:
            ast.literal_eval(x)
            return True
        except:
            return False


    to_eval = df.columns[df.iloc[0].apply(can_eval)]
    for col in to_eval:
        try:
            df[col] = df[col].apply(ast.literal_eval)
        except:
            pass

def get_data(version, data_path='../data'):
    data = {}
    for file in glob('{}/{}/*.csv'
                     .format(data_path, version)):
        name = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        parse_json(df)
        data[name] = drop_nan_cols(df)


    # data['trials']['clicks'] = data['trials'].clicks.apply(ast.literal_eval)
    # data['trials']['n_clicks'] = data['trials'].clicks.apply(len)

    return data



# ---------- Statistics ---------- #

try:
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    # from rpy2.robjects.conversion import ri2py
except:
    print("Error importing rpy2")
else:
    def r2py(results, p_col=None):
        tbl = ri2py(results)
        tbl = tbl.rename(columns=reformat_name)
        if p_col:
            tbl['signif'] = tbl[reformat_name(p_col)].apply(pval)
        return tbl

def df2r(df, cols):
    df = df[cols].copy()
    for name, col in df.iteritems():
        if col.dtype == bool:
            df[name] = col.astype(int)
    return df

def pval(p, sig=3):
    assert sig == 3
    min_p = 10 ** -3
    if p < min_p:
        return f'p < {min_p:.3f}'.replace('0.', '.')
    else:
        return f'p = {p:.3f}'.replace('0.', '.')

from statsmodels.stats.weightstats import ttest_ind
def t_test(x, y):
    t, p, df = ttest_ind(x, y)
    print(f'{x.mean():0.1f} vs. {y.mean():0.1f}, t({int(df)}) = {t:.2f}, {pval(p)}')

# ---------- Saving results ---------- #
class TeX(object):
    """Saves tex files."""
    def __init__(self, path='stats', clear=False):
        self.path = path
        if clear:
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


    def write(self, name, tex):
        file = f"{self.path}/{name}.tex"
        with open(file, "w+") as f:
            f.write(str(tex) + r"\unskip")
        print(f'wrote "{tex}" to "{file}"')
      
# ---------- Plotting ---------- #

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.4)
sns.set_palette('deep', color_codes=True)

class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs/', formats=['pdf']):
        self.path = path
        self.formats = formats
        os.makedirs(path, exist_ok=True)


    def savefig(self, name):
        name = name.lower()
        for fmt in self.formats:
            path = os.path.join(self.path, name + '.' + fmt)
            print(path)
            plt.savefig(path, bbox_inches='tight')

    def plot(self, **kwargs1):
        """Decorator that calls a plotting function and saves the result."""
        def decorator(func):
            def wrapped(*args, **kwargs):
                kwargs.update(kwargs1)
                params = [v for v in kwargs1.values() if v is not None]
                param_str = '_' + str_join(params).rstrip('_') if params else ''
                name = func.__name__ + param_str
                if name.startswith('plot_'):
                    name = name[len('plot_'):]
                func(*args, **kwargs)
                self.savefig(name)
            wrapped()
            return wrapped

        return decorator


sns.set_style('whitegrid')
blue, orange, green = sns.color_palette('tab10')[:3]
gray = (0.5,)*3
red = (1, 0.2, 0.3)
yellow = (0.9, 0.85, 0)


palette = {
    'video': green,
    'none': gray,
    'action': blue,
    'meta': orange,
    'info_only': red,
    'reward_only': yellow,
    'both': orange,
}

nice_names = {
    'meta': 'Metacognitive',
    'action': 'Action',
    'none': 'None',
    'video': 'Video',
    'feedback': 'Feedback',
    'info_only': 'Information\nOnly',
    'reward_only': 'Delay Penalty\nOnly',
    'both': 'Information &\nDelay Penalty',
    'score': 'Average Score',
    'backward': 'Proportion Planning Backward',
    'bonus': 'Bonus Amount ($)',
    'stage2_n_click': 'Average # Clicks in Transfer'
}

def reformat_labels(ax=None):
    ax = ax or plt.gca()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    new_labels = [nice_names.get(lab, lab) for lab in labels]
    ax.set_xticklabels(new_labels)
    
def reformat_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[nice_names.get(l, l).replace('\n', ' ') 
                                       for l in labels])
    
def plot_block_changes():
    block_changes = mdf.loc[1].block.apply(Labeler()).diff().reset_index().query('block == 1').index
    for t in block_changes:
        plt.axvline(t-0.5, c='k', ls='--')

# from datetime import datetime
# # os.makedirs(f'stats/{EXPERIMENT}/', exist_ok=True)
# def result_file(name, ext='tex'):
#     file = f'stats/{EXPERIMENT}-{name}.{ext}'
# #     with open(file, 'w+') as f:
# #         timestamp = datetime.now().strftime('Created on %m/%d/%y at %H:%M:%S\n\n')
# #         f.write(timestamp)
#     return file

