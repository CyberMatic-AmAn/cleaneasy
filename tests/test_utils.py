import pandas as pd
import numpy as np
import pytest
from cleaneasy.utils import get_column_types, setup_logger, convert_to_dataframe

def test_get_column_types():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    types = get_column_types(df)
    assert types == {'A': 'int64', 'B': 'object'}

def test_setup_logger():
    logger = setup_logger('DEBUG')
    assert logger.level == 10  # DEBUG level

def test_convert_to_dataframe():
    # Test DataFrame
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = convert_to_dataframe(df)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)

    # Test NumPy array
    arr = np.array([[1, 2], [3, 4]])
    result = convert_to_dataframe(arr, columns=['A', 'B'])
    assert list(result.columns) == ['A', 'B']

    # Test list of dicts
    data = [{'A': 1, 'B': 'x'}, {'A': 2, 'B': 'y'}]
    result = convert_to_dataframe(data)
    assert list(result.columns) == ['A', 'B']

    # Test list of lists
    data = [[1, 'x'], [2, 'y']]
    result = convert_to_dataframe(data, columns=['A', 'B'])
    assert list(result.columns) == ['A', 'B']

    with pytest.raises(ValueError):
        convert_to_dataframe(123)  # Unsupported type