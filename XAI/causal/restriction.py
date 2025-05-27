"""
    Restriction for each method of causal discovery. In other words, the restriction is the background knowledge.
    Such as prevent the target node being a parent node etc.
"""


from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
import pandas as pd
import numpy as np


bk = BackgroundKnowledge()

def PC_BGKnowledge(
        df: pd.DataFrame,
        features: pd.DataFrame,
        target: str
):
    """
    Background knowledge based on PC algorithm.

    Y is a target, must be a child node, the final causal graph must be X->Y or X->Z->Y, cannot be Y->Z

    Args:
        df: whole dataframe
        features: features dataframe
        target: label of target node

    Returns:
        background knowledge
    """
    node_table = {}
    for name in df.columns:
        n = GraphNode(name)
        node_table[name] = n
    cls_node = node_table[target]
    for feat in features.columns:
        bk.add_forbidden_by_node(cls_node, node_table[feat])
    print(type(bk))
    return bk

def DirectLiNGAM_BGKnowledge(
        features: list,
        target: str) -> pd.DataFrame:
    """
        Background knowledge based on DirectLiNGAM algorithm.

        Y is a target, must be a child node, the final causal graph must be X->Y or X->Z->Y, cannot be Y->Z

        Args:
            features: list of features.
            target: label of target node.

        Returns:
            9*9 matrix background knowledge
        """

    name_to_idx = {name: idx for idx, name in enumerate(features)}
    p = len(features)

    bk = np.zeros((p, p), dtype=int)  # 0 = unspecified
    class_idx = name_to_idx[target]
    bk[class_idx, :] = -1  # forbid outgoing from class
    bk[class_idx, class_idx] = 0

    return bk