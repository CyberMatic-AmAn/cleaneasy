import pandas as pd
import numpy as np
import logging
from typing import Dict, Union, Optional, List

def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return a dictionary of column names and their data types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def setup_logger(log_level: str = 'INFO') -> logging.Logger:
    """Set up a logger with the specified log level."""
    logger = logging.getLogger('CleanEasy')
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def convert_to_dataframe(data: Union[pd.DataFrame, np.ndarray, List, Dict, str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert various data structures to a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, np.ndarray):
        if columns is None:
            columns = [f'col_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)
    elif isinstance(data, list):
        if not data:
            raise ValueError("Empty list provided")
        if isinstance(data[0], dict):
            return pd.DataFrame(data)
        else:
            if columns is None:
                columns = [f'col_{i}' for i in range(len(data[0]) if isinstance(data[0], (list, tuple)) else 1)]
            return pd.DataFrame(data, columns=columns)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    elif isinstance(data, str):
        try:
            return pd.read_csv(data)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {data}: {e}")
    else:
        raise ValueError("Unsupported data type. Must be pandas DataFrame, NumPy array, list, dict, or CSV file path")