import scipy.sparse as sp
import scipy as sp
import numpy as np
import warnings
import itertools
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split



def cholesky(A, sparse=True, verbose=False):
    if not sp.sparse.issparse(A) and sparse:
        msg = 'Matrix not sparse. Using dense cholesky.'
        if verbose:
            warnings.warn(msg)

    if sp.sparse.issparse(A):
        A = A.toarray()
    try:
        L = sp.linalg.cholesky(A, lower=False)
        return L
    except:
        msg = 'Matrix not positive definite. Adding jitter.'
        if verbose:
            warnings.warn(msg)

        jitter = np.eye(A.shape[0]) * np.abs(np.mean(np.diag(A))) * 1e-8
        return cholesky(A + jitter, sparse=False, verbose=verbose)

def batch_pairs_to_dataframe(vars_list, func):
    """
    For each unordered pair (x, y) in vars_list, call func(x, y) and collect the outputs in a DataFrame.

    :param vars_list: Iterable of variables (e.g., names, IDs, pandas Series, or any hashable objects)
    :param func: Function that accepts two variables and returns a tuple (a, b)
    :return: pandas DataFrame with columns ['var_1', 'var_2', 'a->b', 'b->a']
    """
    rows = []
    # Iterate over all unique pairs of variables
    for v1, v2 in itertools.combinations(vars_list, 2):
        a, b = func(v1, v2)
        rows.append({
            'var_1': v1,
            'var_2': v2,
            'a->b':  a,  # first value returned by func(v1, v2)
            'b->a':  b   # second value returned by func(v1, v2)
        })
    return pd.DataFrame(rows)

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph

def collapse_classes(df: pd.DataFrame,
                     new_val: int,
                     class_col: str = "class",
                     keep_val: int = 0,
                     inplace: bool = False) -> pd.DataFrame:
    """
    combined other class values with a new value.

    Example: original column class = [0, 11, 12, 13, 14, 15] keep value = 0, new val = 1 -> class = [0, 1]

    :param df: pd.DataFrame
    :param new_val: new int values
    :param class_col: which column to collapse
    :param keep_val: class value to keep.
    :return: pandas DataFrame
    """

    if not inplace:
        _df = df.copy()

    mask = _df[class_col]!= keep_val

    if not mask.any():
        return _df

    _df.loc[mask, class_col] = new_val
    return _df

def min_sample_retention(df: pd.DataFrame,
                         test_size: int,
                         random_state: int
                         )-> pd.DataFrame:
    _df = df.copy()
    _test_size = test_size
    _random_state = random_state

    _, sample = train_test_split(_df,
                                 test_size=_test_size,
                                 stratify=_df['class'],
                                 shuffle=True,
                                 random_state=_random_state)

    return sample