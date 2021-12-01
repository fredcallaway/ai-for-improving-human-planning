import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

from utils import *

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



class Tex:
    chi2 = r"$\chi^2({df:.0f})={chisq:.2f},\ {signif}$"

class Variables():
    """Saves variables for use in external documents."""
    def __init__(self, path='.'):
        # os.makedirs(path, exist_ok=True)
        self.path = path
        self.csv_file = os.path.join(path, 'variables.csv')
        self.sed_file = os.path.join(path, 'variables.sed')
        self.tex_file = os.path.join(path, 'variables.tex')
        self.read()

    def read(self):
        try:
            self.series = pd.Series.from_csv(self.csv_file)
        except (OSError, pd.io.common.EmptyDataError):
            self.reset()

    def reset(self):
        self.series = pd.Series()
        self.save()

    def write(self, key, val):
        self.read()
        self.series[key] = val
        self.series.to_csv(self.csv_file)
        print('{} = {}'.format(key, val))

    def save(self):
        self.series.to_csv(self.csv_file)
        with open(self.sed_file, 'w+') as f:
            for key, val in self.series.items():
                val = str(val).replace('\\', '\\\\').replace('&', '\&')
                f.write('s/`{}`/{}/g'.format(key, val) + '\n')

        with open(self.tex_file, 'w+') as f:
            for key, val in self.series.items():
                key = to_camel_case(key)
                f.write(r'\newcommand{\%s}{%s}' % (key, val) + '\n')


    def save_analysis(self, table, tex, name='', idx='{index}', display_tex=True):
        if display_tex:
            from IPython.display import Latex, display

        for i, row in table.iterrows():
            row['index'] = i
            n = name
            if idx is not None:
                n += '_' + (idx(row) if callable(idx) else idx)
            n = reformat_name(n.format_map(row)).upper()

            t = tex(row) if callable(tex) else tex
            t = t.format_map(row)

            self.write(n, t)
            if display_tex:
                display(Latex(t))

        self.save()


    def write_lm(self, model, var, name):
        beta = np.round(model.params[var], 2)
        se = np.round(model.bse[var], 2)
        p = model.pvalues[var]
        p_desc = pval(p)

        self.write_var(
            '{}_RESULT'.format(name),
            r'$\\beta = %s,\\ \\text{SE} = %s,\\ %s$' % (beta, se, p_desc)
        )

def get_rtable(results, p_col=None):
    tbl = ri2py(results)
    tbl = tbl.rename(columns=reformat_name)
    if p_col:
        tbl['signif'] = tbl[reformat_name(p_col)].apply(pval)
    return tbl


class Results(object):
    """Writes results to files"""
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def file(self, name):
        # mode = 'a+' if append else 'w+'
        file = os.path.join(self.path, name + '.txt')
        with open(file, 'w+') as f:
            timestamp = datetime.now().strftime('Created on %m/%d/%y at %H:%M:%S\n\n')
            f.write(timestamp)
        return open(file, 'a')



# ---------- Plotting ---------- #

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.4)
sns.set_palette('deep', color_codes=True)



class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs/', formats=['eps']):
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

def get_stars(p):
    if p < .001:
        return '***'
    elif p < .01:
        return '**'
    elif p < .05:
        return '*'
    else:
        return None

    