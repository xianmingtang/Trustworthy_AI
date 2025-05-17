import warnings
import webbrowser
import os
import pandas as pd
import numpy as np
import shap
from npeet.entropy_estimators import cmi
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import xgboost as xgb
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


def add_vector_magnitude_column(df, cols, new_col_name):
    """
    Adds a new column to the DataFrame containing the Euclidean norm (vector magnitude)
    calculated from three specified columns (e.g., x, y, z).

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        cols (list of str): A list of exactly 3 column names, e.g., ['x', 'y', 'z'].
        new_col_name (str): The name of the new column to store the computed magnitudes.

    Returns:
        pandas.DataFrame: The original DataFrame with the new magnitude column added.
    """
    if len(cols) != 3:
        raise ValueError("The 'cols' argument must be a list of exactly 3 column names, e.g., ['x', 'y', 'z']")

    df[new_col_name] = np.linalg.norm(df[cols].values, axis=1)
    return df

def compute_first_order_cmi_npeet(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5
) -> dict[str, pd.Series]:
    """
    Compute first-order conditional mutual information I (xi; y | xj) using NPEET.

    Principle:
      - NPEET uses k-nearest neighbor distances to estimate the entropy of continuous
        or discrete variables without binning.
      - Conditional Mutual Information (CMI) I(X; Y | Z) = H(X, Z) + H(Y, Z) – H(Z) – H(X, Y, Z),
        all estimated via kNN.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_samples × n_features) may contain continuous and/or discrete columns.
    y : pd.Series
        Binary target (length n_samples), values 0/1 or any two distinct values.
    k : int, default=5
        Number of neighbors for kNN estimation.

    Returns
    -------
    result_dict : dict of pd.Series
      Keys: feature names xi.
      Values: pd.Series indexed by conditioning feature xj, values are I (xi; y | xj),
              sorted descending.
    """
    y_vals = y.values.reshape(-1)
    feature_names = X.columns.tolist()
    result_dict = {}
    # print(y_vals)
    # print(feature_names)

    # Loop over each feature xi
    for xi in feature_names:
        xi_vals = X[xi].values.reshape(-1)
        scores = {}
        print(f"Computing first-order conditional mutual information (CMI) for {xi} ...")
        # Compute I(xi; y | xj) for every other feature xj
        for xj in feature_names:
            if xj == xi:
                continue
            zj_vals = X[xj].values.reshape(-1, 1)  # conditioning variable must be 2D
            # print(f'zj = {zj_vals}')
            cmi_val = cmi(xi_vals, y_vals, zj_vals, k=k)
            # print(f'cmi = {cmi_val}')
            scores[xj] = max(0, cmi_val)
        # print(scores)
        # Sort and store as pandas Series
        result_dict[xi] = pd.Series(scores).sort_values(ascending=False)
        # print('*-'*50)

    return result_dict

def compute_second_order_cmi_npeet(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5
) -> dict[str, pd.Series]:
    """
    Compute second-order conditional mutual information I(x_i; y | x_j, x_k) for each feature x_i.

    Principle:
      - Uses k-nearest neighbor (kNN) entropy estimators from NPEET to estimate
        conditional mutual information directly on mixed continuous/discrete data.
      - Second-order CMI: I(X; Y | Z1, Z2) = H(X,Z1,Z2) + H(Y,Z1,Z2) - H(Z1,Z2) - H(X,Y,Z1,Z2).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape (n_samples, n_features), containing 8 columns.
    y : pd.Series or single-column pd.DataFrame
        Binary target variable (0/1) of length n_samples.
    k : int, default=5
        Number of nearest neighbors for kNN-based entropy estimation.

    Returns
    -------
    result_dict : dict of pd.Series
      Keys are each feature name x_i.
      Values are pd.Series indexed by ordered feature pairs (x_j, x_k),
      values are estimated I(x_i; y | x_j, x_k), sorted descending.
    """

    # Ensure y is a flat 1D array
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y_vals = y.values.reshape(-1)

    feature_names = X.columns.tolist()
    result_dict = {}

    # For each target feature xi
    for xi in feature_names:
        xi_vals = X[xi].values.reshape(-1)
        scores = {}
        print(f"Computing second-order conditional mutual information (CMI) for {xi} ...")

        # All other features as ordered pairs (xj, xk)
        others = [f for f in feature_names if f != xi]
        for xj in others:
            for xk in others:
                if xk == xj:
                    continue
                # Build conditioning array Z of shape (n_samples, 2)
                Z = X[[xj, xk]].values
                # Estimate I(xi; y | [xj, xk])
                cmi_val = cmi(xi_vals, y_vals, Z, k=k)
                scores[(xj, xk)] = cmi_val

        # Sort the scores descending and store in dictionary
        result_dict[xi] = pd.Series(scores).sort_values(ascending=False)

    return result_dict


def plot_cmi(
    cmi_dict: dict[str, pd.Series],
    order: int = 1
):
    """
    Plot conditional mutual information (CMI) results.

    Parameters
    ----------
    cmi_dict : dict[str, pd.Series]
        Dictionary mapping each feature xi to a pandas Series of CMI values.
        For first-order (order=1), Series.index are single feature names xj.
        For second-order (order=2), Series.index are tuples (xj, xk).
    order : int, default=1
        Specify 1 to plot first-order CMI I(xi; y | xj),
        or 2 to plot second-order CMI I(xi; y | xj, xk).

    Behavior
    --------
    - Creates one bar chart per xi.
    - For first-order, x-axis labels are xj.
    - For second-order, x-axis labels are "xj,xk".

    Example usage:
    --------
    - plot_cmi(first_order_cmi_dict, order=1)
    - plot_cmi(second_order_cmi_dict, order=2)

    """
    features = list(cmi_dict.keys())
    n = len(features)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for idx, xi in enumerate(features):
        ax = axes[idx]
        series = cmi_dict[xi]
        if order == 1:
            labels = series.index.tolist()
        else:
            # order 2: index is tuple (xj, xk)
            labels = [f"{j},{k}" for j, k in series.index]

        ax.bar(labels, series.values)
        title = f"{order}st-order CMI for {xi}" if order == 1 else f"{order}nd-order CMI for {xi}"
        ax.set_title(title)
        ax.set_ylabel("Conditional Mutual Information")
        ax.set_xlabel(" and ".join(["xj"] if order == 1 else ["xj,xk"]))
        ax.tick_params(axis='x', rotation=90)

    # Remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    '''
    1. 
        Data Preprocessing
    '''

    Current_path = os.getcwd()
    path = './Dataset/veremi_extension_simple.csv'
    # data_origin = pd.read_csv(path, nrows=500000)
    data_origin = pd.read_csv(path, nrows=500000)
    print(data_origin.head(5))
    # Observed data, "type", "Attack", "Attack_type", will be dropped. Keep the class feature value only 0(Normal) and 15(Traffic congestion sybil).
    drop_column = ['type','Attack','Attack_type']
    data = data_origin.drop(drop_column,axis=1)
    data = data[data['class'].isin([0, 15])]

    # Check the non-numerical rows, delete non-numeric rows
    data = data.apply(pd.to_numeric, errors='coerce')
    data_processed = data.dropna(axis=0, how='any')

    # Combine axis related data such as pos, spd etc. by using M = \sqrt{X^2 + Y^2 + Z^2}
    data_processed = add_vector_magnitude_column(data_processed, ['posx', 'posy', 'posz'], 'pos')
    data_processed = add_vector_magnitude_column(data_processed, ['spdx', 'spdy', 'spdz'], 'spd')
    data_processed = add_vector_magnitude_column(data_processed, ['aclx', 'acly', 'aclz'], 'acl')
    data_processed = add_vector_magnitude_column(data_processed, ['hedx', 'hedy', 'hedz'], 'hed')
    data_processed.drop(columns=['posx', 'posy', 'posz', 'spdx', 'spdy', 'spdz', 'aclx', 'acly', 'aclz', 'hedx', 'hedy', 'hedz'], inplace=True)
    features = data_processed.drop(columns=['class'])
    target = data_processed['class']

    # Standardize
    scaler = StandardScaler()
    standardized_array = scaler.fit_transform(features)
    data_processed = pd.DataFrame(standardized_array, columns=features.columns)


    print(len(data_processed))
    data_processed['class'] = target.values

    with pd.option_context('display.max_columns', None):
        print(data_processed)

    count_0 = (data_processed['class'] == 0).sum()
    count_15 = (data_processed['class'] == 15).sum()
    print(f'normal: {count_0}, sybil attack: {count_15}')
    print('*-'*50)

    X = data_processed[features.columns]     # 8 features
    y = data_processed['class']      # Target variable

    '''
    2. 
        Clarify the causal relationship between the features and the target variable
    '''
    # Calculate mutual information scores for each feature
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    mi_series.plot.bar()
    plt.title("Single-Variable Mutual Information I(X; Y)")
    plt.ylabel("Mutual Information")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    print(f'mutual info:\n{mi_series})')
    print('*-'*50)


    '''
    3. 
        Independence test features based on mutual information (Conditional Mutual Information)
        This step is only use to define the causal relationship between the features and the target variable. There is no directional causal relationship.
    '''
    # 3.1 skeleton creation: Computing the rank 1 and rank2 features that provide given by y, example: I(Xi,Y|Xj), I(Xi,Y|Xj,Xk) where (j != i, k != i, and j != k)
    cmi_result_first_order = compute_first_order_cmi_npeet(X, y)
    cmi_result_second_order = compute_second_order_cmi_npeet(X, y)
    print(cmi_result_first_order)
    print('*-'*50)
    print(cmi_result_second_order)
    print('*-'*50)
    plot_cmi(cmi_result_first_order, order=1)
    plot_cmi(cmi_result_second_order, order=2)


    # 3.2
    # 3.3
    # 3.4
    # 3.5
    # 3.6


    '''4. Granges-Friedman test features based on mutual information(Focus on timestamp and sender)'''
    '''5. Create a SEM model, clarify the direct or indirect causal relationship'''
    '''6. Intervention (Evaluate each features impact on the target variable)'''
    '''7. Causal Graph Generation'''


    # SHAP Values for XGBOOST
    # y = data_processed['class'].map({0: 0, 15: 1})
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    # model_xgb.fit(X_train, y_train)
    # y_pred = model_xgb.predict(X_test)
    #
    # print("✅ Model Evaluation Metrics:")
    # print("Accuracy:      ", accuracy_score(y_test, y_pred))
    # print("Precision:     ", precision_score(y_test, y_pred))
    # print("Recall:        ", recall_score(y_test, y_pred))
    # print("F1 Score:      ", f1_score(y_test, y_pred))
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("\nDetailed Report:\n")
    # print(classification_report(y_test, y_pred))
    # print('*-'*50)

    # explainer = shap.TreeExplainer(model_xgb)
    # shap_values = explainer.shap_values(X_test)
    # shap_importance = np.abs(shap_values).mean(axis=0)
    # shap_series = pd.Series(shap_importance, index=X.columns).sort_values(ascending=False)
    # print(f'shap values for xgb:\n{shap_series})')

    # explainer_xgb = shap.Explainer(model_xgb)
    # shap_values = explainer_xgb(X_test)
    # shap.summary_plot(shap_values, X_test)
    # shap_importance = np.abs(shap_values.values).mean(axis=0)
    # shap_series = pd.Series(shap_importance, index=X.columns).sort_values(ascending=False)
    # print(f'shap values for xgb:\n{shap_series})')
    # print('*-'*50)

    # 1. PC Algorithm
    'we are using fisher z algorithm as the independence test method since we have standard data match normal distribution'

    # data_ndarray = data_processed.to_numpy()
    # cg = pc(data_ndarray,
    #         alpha = 0.1,
    #         indep_test = 'fisherz',
    #         stable = True,
    #         uc_rule = 1,
    #         uc_priority = 4,
    #         mvpc = False,
    #         background_knowledge = None,
    #         verbose = True,
    #         show_progress = True,
    #         )
    #
    # 'visualization using pydot'
    # cg.draw_pydot_graph(labels=['sendTime', 'sender', 'senderPseudo', 'messageID', 'class', 'pos', 'spd', 'acl', 'hed'])
    #
    # # 2. GES with the BIC score or generalized score
    # Record = ges(data_ndarray)
    # pyd = GraphUtils.to_pydot(Record['G'])
    # tmp_png = pyd.create_png(f="png")
    # fp = io.BytesIO(tmp_png)
    # img = mpimg.imread(fp, format='png')
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()

