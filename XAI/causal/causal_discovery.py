import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.search.PermutationBased.gst import GST;
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
import pandas as pd
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import networkx as nx

def pc_algorithm(
        df: pd.DataFrame,
        indep_test_func: callable,
        alpha: float,
        uc_rule: int,
        max_k: int,
        background_knowledge,
        node_names: list[str]
):
    cg = pc(
        df.values,
        indep_test_func=indep_test_func,
        alpha=alpha,
        uc_rule=uc_rule,
        max_k=max_k,
        background_knowledge=background_knowledge,
        node_names=node_names
    )
    return cg


def fci_algorithm(
        df: pd.DataFrame,
        indep_test_func: callable,
        alpha: float,
        depth: int,
        max_path_length: int,
        verbose: bool,
        background_knowledge,
        show_progress: bool,
        node_names: list[str]
):
    g, edges = fci(
        df.values,
        indep_test_func=indep_test_func,
        alpha=alpha,
        depth=depth,
        max_path_length=max_path_length,
        verbose=verbose,
        background_knowledge=background_knowledge,
        show_progress=show_progress,
        node_names=node_names
    )
    return g, edges

def notears_linear(
        X,
        lambda1,
        loss_type,
        max_iter=500,
        h_tol=1e-8,
        rho_max=1e+16,
        w_threshold=0.03
):
    """
    solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    def _enforce_leafnode(W, u):
        W2 = W.copy()
        W2[:, u] = 0.0
        # W2[u, :] = 0.0
        return W2

    def _project_to_dag(W, zero_tol=1e-12):
        """Remove the smallest-|w| edge on a cycle, repeatedly, until DAG."""
        W2 = W.copy()
        W2[np.abs(W2) < max(zero_tol, 0.)] = 0.0
        d_ = W2.shape[0]
        G = nx.DiGraph(((j, i, {'w': W2[j, i]}) for j in range(d_) for i in range(d_) if i != j and W2[j, i] != 0.0))
        try:
            cycle = nx.find_cycle(G, orientation='original')
        except nx.NetworkXNoCycle:
            cycle = None

        while cycle:
            u, v = min(((u, v) for u, v, _ in cycle), key=lambda e: abs(W2[e[0], e[1]]))
            W2[u, v] = 0.0
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            try:
                cycle = nx.find_cycle(G, orientation='original')
            except nx.NetworkXNoCycle:
                break
        return W2

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    W_est = _enforce_leafnode(W_est, 7)

    ensure_dag = True

    if ensure_dag:
        W_est = _project_to_dag(W_est, zero_tol=max(1e-12, 0.1 * w_threshold))

    return W_est


def boss(
        X: np.ndarray,
        score_func: str = "local_score_marginal_general",
        parameters: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = True,
        node_names: Optional[List[str]] = None,
) -> GeneralGraph:
    """
    Perform a best order score search (BOSS) algorithm

    node_names: ['sendTime', 'senderPseudo', 'messageID', 'pos', 'spd', 'acl', 'hed', 'class']

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BIC_from_cov', 'local_score_BDeu')).
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.
    verbose : whether to print the time cost and verbose output of the algorithm.

    Returns
    -------
    G : learned causal graph, where G.graph[j,i] = 1 and G.graph[i,j] = -1 indicates i --> j, G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j.
    """

    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    if score_func == "local_score_CV_general":
        # % k-fold negative cross validated likelihood based on regression in RKHS
        if parameters is None:
            parameters = {
                "kfold": 10,  # 10 fold cross validation
                "lambda": 0.01,
            }  # regularization parameter
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters
        )
    elif score_func == "local_score_marginal_general":
        # negative marginal likelihood based on regression in RKHS
        parameters = {}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters
        )
    elif score_func == "local_score_CV_multi":
        # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {
                "kfold": 10,
                "lambda": 0.01,
                "dlabel": {},
            }  # regularization parameter
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters
        )
    elif score_func == "local_score_marginal_multi":
        # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters
        )
    elif score_func == "local_score_BIC":
        # SEM BIC score
        warnings.warn("Using 'local_score_BIC_from_cov' instead for efficiency")
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BIC_from_cov":
        # SEM BIC score
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BDeu":
        # BDeu score
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )
    else:
        raise Exception("Unknown function!")

    score = localScoreClass
    gsts = [GST(i, score) for i in range(p)]

    node_names = [("X%d" % (i + 1)) for i in range(p)] if node_names is None else node_names
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)

    runtime = time.perf_counter()

    order = [v for v in range(p)]
    print(f'order:{order}')
    leaf_idx = node_names.index("class")
    print(f'leaf_idx:{leaf_idx}')

    gsts = [GST(v, score) for v in order]
    print(f'gsts:{gsts}')

    # New >>> Added forbidden node to gst, class will not be a parent node to any other nodes.
    for gst in gsts:
        if gst.vertex != leaf_idx:
            gst.forbidden.append(leaf_idx)
    for i, gst in enumerate(gsts):
        print(f"=== GST #{i} ===")
        print("vertex        :", gst.vertex)
        print("forbidden     :", gst.forbidden)
        print("required      :", gst.required)
        print("root.grow_score :", gst.root.grow_score)
        print("root.shrink_score:", gst.root.shrink_score)
    # New <<<

    parents = {v: [] for v in order}
    print(f'parents:{parents}')

    variables = [v for v in order]

    # New >>> never
    # variables = [v for v in order if v != leaf_idx]
    # print(variables)
    # New <<<

    while True:
        improved = False
        random.shuffle(variables)
        print(variables)
        if verbose:
            for i, v in enumerate(order):
                parents[v].clear()
                gsts[v].trace(order[:i], parents[v])
            sys.stdout.write("\rBOSS edge count: %i    " % np.sum([len(parents[v]) for v in range(p)]))
            sys.stdout.flush()

        for v in variables:
            improved |= better_mutation(v, order, gsts)
        if not improved: break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    runtime = time.perf_counter() - runtime

    if verbose:
        sys.stdout.write("\nBOSS completed in: %.2fs \n" % runtime)
        sys.stdout.flush()

    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)

    return G


def reversed_enumerate(iter, j):
    for w in reversed(iter):
        yield j, w
        j -= 1


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]: best = j

    if scores[i] + 1e-6 > scores[best]: return False
    order.remove(v)
    order.insert(best - int(best > i), v)

    return True

# New NOTEARS Algorithm

# def notears_linear(X, lambda1, loss_type, mask_l2, mask_logit, mask_pois, max_iter=500, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
#     """
#     solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
#
#     Args:
#         X (np.ndarray): [n, d] sample matrix
#         lambda1 (float): l1 penalty parameter
#         loss_type (str): l2, logistic, poisson
#         max_iter (int): max num of dual ascent steps
#         h_tol (float): exit if |h(w_est)| <= htol
#         rho_max (float): exit if rho >= rho_max
#         w_threshold (float): drop edge if |weight| < threshold
#
#     Returns:
#         W_est (np.ndarray): [d, d] estimated DAG
#     """
#
#     n, d = X.shape
#     use_mixed = (mask_l2 is not None) or (mask_logit is not None) or (mask_pois is not None)
#     if use_mixed:
#         mask_l2 = np.zeros(d, bool) if mask_l2 is None else np.asarray(mask_l2, dtype=bool)
#         mask_logit = np.zeros(d, bool) if mask_logit is None else np.asarray(mask_logit, dtype=bool)
#         mask_pois = np.zeros(d, bool) if mask_pois is None else np.asarray(mask_pois, dtype=bool)
#         assert mask_l2.size == mask_logit.size == mask_pois.size == d, "掩码长度必须等于列数 d"
#         # 互斥
#         assert not np.any(mask_l2 & mask_logit) and not np.any(mask_l2 & mask_pois) and not np.any(
#             mask_logit & mask_pois), \
#             "mask_l2/logit/pois 需两两互斥"
#
#     def _sigmoid(z):
#         return 1.0 / (1.0 + np.exp(-z))
#
#     def _loss(W):
#         M = X @ W  # n x d
#         if use_mixed:
#             loss = 0.0
#             G_loss = np.zeros_like(W)  # d x d
#
#             # L2
#             if mask_l2.any():
#                 J = np.where(mask_l2)[0]
#                 R = X[:, J] - M[:, J]  # n x |J|
#                 loss += 0.5 / n * np.sum(R * R)
#                 G_loss[:, J] += - (X.T @ R) / n  # d x |J|
#
#             # Logistic
#             if mask_logit.any():
#                 J = np.where(mask_logit)[0]
#                 Mj = M[:, J];
#                 Xj = X[:, J]
#                 P = _sigmoid(Mj)
#                 loss += np.sum(np.logaddexp(0.0, Mj) - Xj * Mj) / n
#                 G_loss[:, J] += (X.T @ (P - Xj)) / n
#
#             # Poisson
#             if mask_pois.any():
#                 J = np.where(mask_pois)[0]
#                 Mj = M[:, J];
#                 Xj = X[:, J]
#                 S = np.exp(Mj)
#                 loss += np.sum(S - Xj * Mj) / n
#                 G_loss[:, J] += (X.T @ (S - Xj)) / n
#
#             return loss, G_loss
#
#         else:
#             if loss_type == 'l2':
#                 R = X - M
#                 loss = 0.5 / n * (R ** 2).sum()
#                 G_loss = - (X.T @ R) / n
#             elif loss_type == 'logistic':
#                 loss = (np.logaddexp(0, M) - X * M).sum() / n
#                 G_loss = (X.T @ (_sigmoid(M) - X)) / n
#             elif loss_type == 'poisson':
#                 S = np.exp(M)
#                 loss = (S - X * M).sum() / n
#                 G_loss = (X.T @ (S - X)) / n
#             else:
#                 raise ValueError("unknown loss type (or provide masks for mixed loss)")
#             return loss, G_loss
#
#     # def _loss(W):
#     #     """Evaluate value and gradient of loss."""
#     #     M = X @ W
#     #     if loss_type == 'l2':
#     #         R = X - M
#     #         loss = 0.5 / X.shape[0] * (R ** 2).sum()
#     #         G_loss = - 1.0 / X.shape[0] * X.T @ R
#     #     elif loss_type == 'logistic':
#     #         loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
#     #         G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
#     #     elif loss_type == 'poisson':
#     #         S = np.exp(M)
#     #         loss = 1.0 / X.shape[0] * (S - X * M).sum()
#     #         G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
#     #     else:
#     #         raise ValueError('unknown loss type')
#     #     return loss, G_loss
#
#     def _h(W):
#         """Evaluate value and gradient of acyclicity constraint."""
#         E = slin.expm(W * W)  # (Zheng et al. 2018)
#         h = np.trace(E) - d
#         #     # A different formulation, slightly faster at the cost of numerical stability
#         #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
#         #     E = np.linalg.matrix_power(M, d - 1)
#         #     h = (E.T * M).sum() - d
#         G_h = E.T * W * 2
#         return h, G_h
#
#     def _adj(w):
#         """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
#         return (w[:d * d] - w[d * d:]).reshape([d, d])
#
#     def _func(w):
#         """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
#         W = _adj(w)
#         loss, G_loss = _loss(W)
#         h, G_h = _h(W)
#         obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
#         G_smooth = G_loss + (rho * h + alpha) * G_h
#         g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
#         return obj, g_obj
#
#     def _enforce_leafnode(W, u):
#         W2 = W.copy()
#         W2[:, u] = 0.0
#         # W2[u, :] = 0.0
#         return W2
#
#     w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
#     bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
#     if loss_type == 'l2':
#         X = X - np.mean(X, axis=0, keepdims=True)
#     for _ in range(max_iter):
#         w_new, h_new = None, None
#         while rho < rho_max:
#             sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
#             w_new = sol.x
#             h_new, _ = _h(_adj(w_new))
#             if h_new > 0.25 * h:
#                 rho *= 10
#             else:
#                 break
#         w_est, h = w_new, h_new
#         alpha += rho * h
#         if h <= h_tol or rho >= rho_max:
#             break
#     W_est = _adj(w_est)
#     W_est[np.abs(W_est) < w_threshold] = 0
#     W_est = _enforce_leafnode(W_est, 7)
#     return W_est