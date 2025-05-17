import scipy.sparse as sp
import scipy as sp
import numpy as np
import warnings
import itertools
import pandas as pd


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
