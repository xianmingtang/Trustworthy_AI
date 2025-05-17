"""
    Handle the data preprocessing.
        Data Clean
        Standardize
        Columns Refactor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Union, Sequence

def clean(
        data: pd.DataFrame,
        drop_row: list[str] | None = None,
        drop_column: list[str] | None = None,
        drop_na: bool = True,
        data_numerical: bool = True,
) -> pd.DataFrame:
    """
        Clean a DataFrame with optional numeric coercion, row/column drops, and NA removal.

        Parameters
        ----------
        data : pd.DataFrame
            Original data.
        drop_row : list[str], optional
            Row labels to drop.
        drop_column : list[str], optional
            Column labels to drop.
        drop_na : bool, default True
            If True, drop any row with at least one NA.
        data_numerical : bool, default True
            If True, coerce entire frame to numeric (non-coercible → NaN).

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.
        """

    df = data.copy()

    # 1) Numeric coercion
    if data_numerical:
        try:
            df = df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            raise ValueError(f"[clean] Failed to coerce to numeric: {e}") from e

    # 2) Drop specified rows
    if drop_row is not None:
        # check existence first
        missing = [r for r in drop_row if r not in df.index]
        if missing:
            raise KeyError(f"[Preprocess:clean] Cannot drop rows, not found in index: {missing}")
        try:
            df = df.drop(drop_row, axis=0)
        except Exception as e:
            raise RuntimeError(f"[Preprocess:clean] Error dropping rows {drop_row}: {e}") from e

    # 3) Drop specified columns
    if drop_column is not None:
        missing = [c for c in drop_column if c not in df.columns]
        if missing:
            raise KeyError(f"[Preprocess:clean] Cannot drop columns, not found: {missing}")
        try:
            df = df.drop(drop_column, axis=1)
        except Exception as e:
            raise RuntimeError(f"[Preprocess:clean] Error dropping columns {drop_column}: {e}") from e

    # 4) Drop NA rows
    if drop_na:
        try:
            df = df.dropna(axis=0, how='any')
        except Exception as e:
            raise RuntimeError(f"[Preprocess:clean] Error dropping NA rows: {e}") from e

    return df

# def standardize(
#         df: pd.DataFrame,
#         exempt_cols: str | list[str] | None = None,
# ) -> pd.DataFrame:
#     """
#         Return a DataFrame where all columns except `exempt_col` are
#         zero‐mean/unit‐variance standardized. `exempt_col` is left unchanged.
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             Input feature matrix. Must contain only numeric columns and no missing values.
#
#         exempt_col : str | None, default None
#             Input string, the column name of the exempt column. If None, no exempt column is used.
#
#         Returns
#         -------
#         pd.DataFrame
#             A DataFrame of the same shape, with standardized values and the original
#             index and column labels preserved.
#
#         Raises
#         ------
#         TypeError
#             If `features` is not a pandas DataFrame, or if it contains non-numeric columns.
#         ValueError
#             If `features` is empty or contains any missing values.
#         RuntimeError
#             If the underlying scaling operation fails.
#
#     Args:
#         exempt_cols:
#         """
#
#     # 1) Type check
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError(f"[standardize] Expected a pandas DataFrame, got {type(df)}")
#
#     # 2) Non-empty check
#     if df.shape[0] == 0 or df.shape[1] == 0:
#         raise ValueError("[standardize] Input DataFrame must have at least one row and one column")
#
#     # 3) Missing-value check
#     if df.isnull().any().any():
#         raise ValueError("[standardize] Input contains missing values; please impute or remove them first")
#
#     # 4) Numeric-type check
#     non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
#     if non_numeric_cols:
#         raise TypeError(
#             f"[standardize] The following columns are non-numeric and cannot be standardized: {non_numeric_cols}")
#
#
#     # 5) If no exempt column given, standardize all columns
#     if exempt_cols is None:
#         # Ensure all columns are numeric
#         non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
#         if non_num:
#             raise TypeError(f"[standardize_except] Cannot standardize non-numeric columns: {non_num}")
#         scaler = StandardScaler()
#         arr = scaler.fit_transform(df.values)
#         return pd.DataFrame(arr, index=df.index, columns=df.columns)
#
#     # 6) Otherwise, standardize all *except* exempt_col
#     if exempt_cols not in df.columns:
#         raise KeyError(f"[standardize_except] Exempt column '{exempt_cols}' not found in DataFrame.")
#
#     # 7) Split off exempt_col
#     exempt = df[[exempt_cols]]
#     to_scale = df.drop(columns=[exempt_cols])
#
#     # 8) Numeric check on the rest
#     non_num = to_scale.select_dtypes(exclude=[np.number]).columns.tolist()
#     if non_num:
#         raise TypeError(f"[standardize_except] Cannot standardize non-numeric columns: {non_num}")
#
#     # 9) Scale
#     scaler = StandardScaler()
#     arr = scaler.fit_transform(to_scale.values)
#     scaled = pd.DataFrame(arr, index=to_scale.index, columns=to_scale.columns)
#
#     # 10) Reassemble
#     return pd.concat([exempt, scaled], axis=1)


def standardize(
        df: pd.DataFrame,
        exempt_cols: Union[str, Sequence[str], None] = None
) -> pd.DataFrame:
    """
    Z-score–standardize every numeric column in *df* **except** those
    listed in *exempt_cols*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
        All columns to be standardized must be numeric and contain no NaNs.
    exempt_cols : str | Sequence[str] | None, default None
        • None   → standardize *all* columns.
        • str    → keep this single column unchanged.
        • list / tuple / set → keep all listed columns unchanged.

    Returns
    -------
    pd.DataFrame
        Data frame whose non-exempt columns are standardized
        (zero mean, unit variance); exempt columns are returned untouched,
        and original column order is preserved.
    """
    # --- Basic Sanity Checks --------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[standardize] Expected a pandas DataFrame, got {type(df)}"
        )

    if df.empty:
        raise ValueError("[standardize] Input DataFrame is empty")

    if df.isnull().any().any():
        raise ValueError(
            "[standardize] Input contains missing values; "
            "please clean or impute them first"
        )

    # --- Normalize *exempt_cols* into a set -----------------------------------
    if exempt_cols is None:
        exempt_set = set()
    elif isinstance(exempt_cols, str):
        exempt_set = {exempt_cols}
    else:                                  # list / tuple / set / np.ndarray …
        exempt_set = set(exempt_cols)

    # Verify that every exempt column exists in df
    missing = exempt_set.difference(df.columns)
    if missing:
        raise KeyError(
            f"[standardize] Exempt column(s) not found in DataFrame: {missing}"
        )

    # --- Split DataFrame into “exempt” and “to-scale” parts -------------------
    exempt_df = df[list(exempt_set)].copy()          # may be empty
    to_scale  = df.drop(columns=exempt_set)

    # All columns to be scaled must be numeric
    non_numeric = to_scale.select_dtypes(
        exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"[standardize] Non-numeric columns cannot be standardized: "
            f"{non_numeric}"
        )

    # --- Apply StandardScaler -------------------------------------------------
    if not to_scale.empty:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(to_scale.values)
        scaled_df = pd.DataFrame(
            scaled_array, index=to_scale.index, columns=to_scale.columns
        )
    else:
        # nothing to scale → create an empty frame with correct index
        scaled_df = pd.DataFrame(index=df.index)

    # --- Reassemble result, preserving original column order -----------------
    result = pd.concat([exempt_df, scaled_df], axis=1)
    result = result[df.columns]            # restore original ordering
    return result


def add_vector_magnitude_column(
        df: pd.DataFrame,
        cols: list[str],
        new_col_name: str) -> pd.DataFrame:
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


def feature_filter_bin(
        data: pd.DataFrame,
        feat_name: str,
        feat_val_1: int,
        feat_val_2: int,
        n_samples: int,
        random_state: int | None = None) -> pd.DataFrame:
    """
        Stratified sampling on exactly two categorical values so that
        the returned DataFrame

          • always contains rows of both feat_val_1 and feat_val_2
          • approximates the original class ratio
          • has size == n_samples (or as close as possible if a class is scarce)

        Parameters
        ----------
        data : pd.DataFrame
            Original data set.
        feat_name : str
            Column to stratify on.
        feat_val_1, feat_val_2 : hashable
            Two values of `feat_name` that must be kept.
        n_samples : int
            Target number of rows to return.
        random_state : int | None
            Seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Sampled data, row order is shuffled.
        """


    return None