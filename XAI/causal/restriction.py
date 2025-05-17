"""
    Restriction for each method of causal discovery. In other words, the restriction is the background knowledge.
    Such as prevent the target node being a parent node etc.
"""


from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
import pandas as pd


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
        No return
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