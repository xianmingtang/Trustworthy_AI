import warnings
import webbrowser
import os
import pandas as pd
import numpy as np
import shap
import networkx as nx
import inspect
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci
from causal import preprocess
from causal import restriction
from causal import causal_discovery as cd
from causal import visualization as vis
from causal import utils
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.FCMBased import lingam
from dowhy import CausalModel
import itertools
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import scipy as sp

if __name__ == '__main__':

    '''
    1. Data Loading and Preprocessing
    '''

    path = '../Dataset/veremi_extension_simple.csv'
    data_origin = pd.read_csv('./Dataset/veremi_extension_simple.csv')
    data_origin = data_origin.sample(n=50000, random_state=42)

    # filter dos and normal data
    # data_origin = data_origin[data_origin['class'].isin([0, 11, 12])]

    # filter sybil and normal data
    # data_origin = data_origin[data_origin['class'].isin([0, 14, 15, 16, 17])]
    print(data_origin.head(5))
    print('*-' * 50)

    # Data Cleaning:
    drop_column = ['type', 'Attack', 'Attack_type']
    data_processed = preprocess.clean(data_origin, drop_column=drop_column, drop_na=True, data_numerical=True)

    # Standardize features, target keep same as original data_processed:
    data_processed = preprocess.standardize(data_processed, ['class', 'sendTime', 'sender', 'senderPseudo'])

    # Combine axis related data such as pos, spd etc. by using M = \sqrt{X^2 + Y^2 + Z^2}
    data_processed = preprocess.add_vector_magnitude_column(data_processed, ['posx', 'posy', 'posz'], 'pos')
    data_processed = preprocess.add_vector_magnitude_column(data_processed, ['spdx', 'spdy', 'spdz'], 'spd')
    data_processed = preprocess.add_vector_magnitude_column(data_processed, ['aclx', 'acly', 'aclz'], 'acl')
    data_processed = preprocess.add_vector_magnitude_column(data_processed, ['hedx', 'hedy', 'hedz'], 'hed')
    data_processed.drop(
        columns=['posx', 'posy', 'posz', 'spdx', 'spdy', 'spdz', 'aclx', 'acly', 'aclz', 'hedx', 'hedy', 'hedz'],
        inplace=True
    )

    dummies_sender = pd.get_dummies(data_processed['sender'], prefix='sender')
    dummies_senderPseudo = pd.get_dummies(data_processed['senderPseudo'], prefix='senderPseudo')

    df_onehot = pd.concat([data_processed.drop(columns=['sender', 'senderPseudo']),
                           dummies_sender,
                           dummies_senderPseudo],
                          axis=1)
    data_processed = df_onehot

    with pd.option_context('display.max_columns', None):
        print(data_processed)
        print(type(data_processed))
    print('*-' * 50)