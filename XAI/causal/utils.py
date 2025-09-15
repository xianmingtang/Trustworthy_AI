import scipy.sparse as sp
import scipy as sp
import numpy as np
import warnings
import itertools
import pandas as pd
import pydot
import graphviz
from sklearn.model_selection import train_test_split
from collections import OrderedDict



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

import numpy as np

def dot_to_adj(dot_input, desired_order, use_edge_label=False):
    """
    change <class 'pydot.core.Dot'> to numpy.ndarray
    Define：adj[i, j] = 1 represent j → i (column is parent，row is child)

    parameters
    ----
    dot_graph : pydot.Dot
    desired_order : list of ordered node names

    return
    ----
    adj_matrix : np.ndarray
    """

    def _s(x):
        if x is None: return None
        x = str(x)
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            return x[1:-1]
        return x

    dot = pydot.graph_from_dot_data(dot_input)[0] if isinstance(dot_input, str) else dot_input

    name_to_label = OrderedDict()
    for n in dot.get_nodes():
        nm = _s(n.get_name())
        if nm in ("graph", "node", "edge"):
            continue
        lb = n.get_attributes().get("label")
        lb = _s(lb if lb is not None else n.get_label())
        name_to_label.setdefault(nm, lb if lb not in (None, "") else nm)

    edges = dot.get_edges()
    for e in edges:
        s = _s(e.get_source());
        t = _s(e.get_destination())
        if s not in (None, "graph", "node", "edge"): name_to_label.setdefault(s, s)
        if t not in (None, "graph", "node", "edge"): name_to_label.setdefault(t, t)

    detected_labels = [name_to_label[nm] for nm in name_to_label.keys()]
    labels = list(desired_order) if desired_order else detected_labels
    if desired_order:
        missing = set(labels) - set(detected_labels)
        if missing:
            raise ValueError(f"Can not find labels: {missing}")

    idx_by_label = {lb: i for i, lb in enumerate(labels)}
    idx_by_name = {nm: idx_by_label[lb] for nm, lb in name_to_label.items() if lb in idx_by_label}

    adj_matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for e in edges:
        s = _s(e.get_source());
        t = _s(e.get_destination())
        if s in ("graph", "node", "edge") or t in ("graph", "node", "edge"):
            continue
        if s not in idx_by_name or t not in idx_by_name:
            continue

        attrs = {k: _s(v) for k, v in e.get_attributes().items()}
        dir_attr = attrs.get("dir")  # forward/back/both/none/None
        arrowhead = attrs.get("arrowhead")  # normal/none/None
        arrowtail = attrs.get("arrowtail")  # normal/none/None

        forward = (dir_attr in (None, "forward", "both")) and (arrowhead != "none")
        reverse = (dir_attr in ("back", "both")) and (arrowtail != "none")

        if use_edge_label:
            w_attr = attrs.get("label")
            try:
                w = float(w_attr) if w_attr not in (None, "") else 1.0
            except ValueError:
                w = 1.0
        else:
            w = 1.0

        parent = idx_by_name[s]
        child = idx_by_name[t]
        if forward:
            adj_matrix[child, parent] = w
        if reverse:
            adj_matrix[parent, child] = w

    return adj_matrix