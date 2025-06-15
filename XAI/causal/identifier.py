from dowhy import CausalModel
from causal import utils
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define Causal Model

def estimate(
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        method_params: dict,
        method_name: str,
        graph: str,
        confidence_intervals: bool = True,
        target_units: str = 'ate',
        test_significance: bool = True
):
    """
        return an estimand result based on the treatement, outcome and graph.

        :param df: pd.DataFrame
        :param treatment: features that need to be treated
        :param outcome: target variable
        :param method_params: method families for data estimation (Gaussian for Multinomial)
        :param method_name: name of the method (Front Door/Back Door/Instrument Variables)
        :param graph: <class 'graphviz.graphs.Digraph'>
        :return: estimand result
    """

    # Model define
    graph_dot = utils.str_to_dot(graph.source)
    model=CausalModel(
            data = df,
            treatment=treatment,
            outcome=outcome,
            graph=graph_dot
    )

    # Identification
    estimand = model.identify_effect(proceed_when_unidentifiable=False)

    # Estimation
    estimate = model.estimate_effect(
        estimand,
        method_params=method_params,
        method_name=method_name,
        confidence_intervals=confidence_intervals,
        test_significance=test_significance,
        target_units=target_units
    )


    return model, estimand, estimate
