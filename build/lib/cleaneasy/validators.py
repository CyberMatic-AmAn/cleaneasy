import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Union

def check_missing_proportion(df: pd.DataFrame) -> Dict[str, float]:
    """Return the proportion of missing values per column."""
    return df.isnull().mean().to_dict()

def check_normality(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, float]:
    """Perform Shapiro-Wilk test for normality on numeric columns."""
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    results = {}
    for col in columns:
        if col in df.columns:
            stat, p = stats.shapiro(df[col].dropna())
            results[col] = p
    return results

def check_unique_values(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, int]:
    """Return the number of unique values per column."""
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.columns
    return {col: df[col].nunique() for col in columns if col in df.columns}

def check_skewness(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, float]:
    """Compute skewness for numeric columns."""
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    return {col: df[col].skew() for col in columns if col in df.columns}

def check_correlation(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, method: str = 'pearson') -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    return df[columns].corr(method=method)