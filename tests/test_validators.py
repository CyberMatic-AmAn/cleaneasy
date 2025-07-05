import pandas as pd
import numpy as np
import pytest
from cleaneasy.validators import check_missing_proportion, check_normality, check_unique_values, check_skewness, check_correlation

def test_check_missing_proportion():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 'y', None]})
    result = check_missing_proportion(df)
    assert result['A'] == 1/3
    assert result['B'] == 2/3

def test_check_normality():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['x', 'y', 'z', 'w', 'v']})
    result = check_normality(df, columns='A')
    assert isinstance(result['A'], float)

def test_check_unique_values():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': ['x', 'y', 'x']})
    result = check_unique_values(df)
    assert result['A'] == 2
    assert result['B'] == 2

def test_check_skewness():
    df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': ['x', 'y', 'z', 'w']})
    result = check_skewness(df, columns='A')
    assert isinstance(result['A'], float)

def test_check_correlation():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 4, 6]})
    result = check_correlation(df)
    assert isinstance(result, pd.DataFrame)
    assert result.loc['A', 'B'] == 1.0