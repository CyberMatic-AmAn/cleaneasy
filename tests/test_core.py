import pandas as pd
import numpy as np
import pytest
from cleaneasy import CleanEasy

@pytest.fixture
def sample_data():
    data = {
        'name': ['John@Doe', 'Jane Smith!', None, 'Alice'],
        'age': [25, 30, 1000, None],
        'salary': [50000, None, 60000, 55000],
        'date': ['2023-01-01', '2023-02-02', 'invalid', '2023-03-03'],
        'category': ['A', 'B', 'A', 'C']
    }
    return pd.DataFrame(data)

def test_init(sample_data):
    cleaner = CleanEasy(sample_data)
    assert isinstance(cleaner.get_cleaned_data(), pd.DataFrame)

def test_impute_knn(sample_data):
    cleaner = CleanEasy(sample_data)
    cleaner.impute_knn(columns=['age', 'salary'], n_neighbors=3)
    assert cleaner.get_cleaned_data()['age'].isnull().sum() == 0
    assert cleaner.get_cleaned_data()['salary'].isnull().sum() == 0

def test_remove_outliers_isolation_forest(sample_data):
    cleaner = CleanEasy(sample_data)
    initial_rows = len(cleaner.get_cleaned_data())
    cleaner.remove_outliers_isolation_forest(columns=['age'], contamination=0.2)
    assert len(cleaner.get_cleaned_data()) <= initial_rows

def test_tokenize_text(sample_data):
    cleaner = CleanEasy(sample_data)
    cleaner.tokenize_text(columns=['name'])
    assert 'name_tokens' in cleaner.get_cleaned_data().columns

def test_extract_day_of_week(sample_data):
    cleaner = CleanEasy(sample_data)
    cleaner.parse_dates(columns=['date'])
    cleaner.extract_day_of_week(columns=['date'], return_numeric=True)
    assert 'date_dayofweek' in cleaner.get_cleaned_data().columns

def test_frequency_encode(sample_data):
    cleaner = CleanEasy(sample_data)
    cleaner.frequency_encode(columns=['category'])
    assert 'category_freq' in cleaner.get_cleaned_data().columns

def test_check_skewness(sample_data):
    cleaner = CleanEasy(sample_data)
    result = cleaner.check_skewness(columns=['age'])
    assert isinstance(result, dict)
    assert 'age' in result

def test_remove_highly_correlated(sample_data):
    cleaner = CleanEasy(sample_data)
    cleaner.remove_highly_correlated(threshold=0.8)
    assert len(cleaner.get_cleaned_data().columns) <= len(sample_data.columns)