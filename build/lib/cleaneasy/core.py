import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Union, Literal, List, Dict
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .utils import get_column_types, setup_logger, convert_to_dataframe

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class CleanEasy:
    """A comprehensive data cleaning toolkit for various data structures."""
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray, List, Dict, str], columns: Optional[List[str]] = None, log_level: str = 'INFO'):
        """Initialize with a pandas DataFrame, NumPy array, list, dict, or file path."""
        self.logger = setup_logger(log_level)
        self.cleaning_steps = []
        self.encoders = {}
        self.results = {}
        self.df = convert_to_dataframe(data, columns)
        self.logger.info(f"Initialized CleanEasy with data type: {type(data).__name__}")
    
    def impute_knn(self, columns: Optional[Union[str, List[str]]] = None, n_neighbors: int = 5, weights: Literal['uniform', 'distance'] = 'uniform') -> 'CleanEasy':
        """Impute missing values using KNN imputation for numeric columns."""
        columns = self._get_columns(columns, numeric=True)
        if columns:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            self.df[columns] = imputer.fit_transform(self.df[columns])
            self.logger.info(f"Imputed {columns} with KNN (n_neighbors={n_neighbors}, weights={weights})")
            self.results['knn_imputation'] = {'columns': columns, 'n_neighbors': n_neighbors, 'weights': weights}
        self.cleaning_steps.append(f"Imputed missing values with KNN (weights={weights})")
        return self

    def remove_outliers_isolation_forest(self, columns: Optional[Union[str, List[str]]] = None, contamination: float = 0.1, random_state: Optional[int] = 42) -> 'CleanEasy':
        """Remove outliers using Isolation Forest for numeric columns."""
        columns = self._get_columns(columns, numeric=True)
        if columns:
            iso = IsolationForest(contamination=contamination, random_state=random_state)
            X = self.df[columns].dropna()
            labels = iso.fit_predict(X)
            mask = pd.Series(labels, index=X.index) == 1
            initial_rows = len(self.df)
            self.df = self.df.loc[self.df.index.isin(X.index[mask])]
            self.logger.info(f"Removed {initial_rows - len(self.df)} outliers from {columns} using Isolation Forest (contamination={contamination})")
            self.results['isolation_forest'] = {'columns': columns, 'outliers_removed': initial_rows - len(self.df)}
        self.cleaning_steps.append("Removed outliers using Isolation Forest")
        return self

    def tokenize_text(self, columns: Optional[Union[str, List[str]]] = None, lowercase: bool = True) -> 'CleanEasy':
        """Tokenize text columns using NLTK."""
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            self.df[f"{col}_tokens"] = self.df[col].apply(
                lambda x: word_tokenize(str(x).lower() if lowercase else str(x)) if pd.notnull(x) else []
            )
            self.logger.info(f"Tokenized text in {col} (lowercase={lowercase})")
            self.results[f"{col}_tokens"] = self.df[f"{col}_tokens"].tolist()
        self.cleaning_steps.append("Tokenized text columns")
        return self

    def extract_day_of_week(self, columns: Optional[Union[str, List[str]]], new_column_suffix: str = '_dayofweek', return_numeric: bool = False) -> 'CleanEasy':
        """Extract day of week from datetime columns (name or number)."""
        columns = self._get_columns(columns, dtype='datetime64[ns]')
        for col in columns:
            new_col = f"{col}{new_column_suffix}"
            if return_numeric:
                self.df[new_col] = self.df[col].dt.dayofweek
            else:
                self.df[new_col] = self.df[col].dt.day_name()
            self.logger.info(f"Extracted day of week from {col} to {new_col} (numeric={return_numeric})")
        self.cleaning_steps.append("Extracted day of week from datetime columns")
        return self

    def frequency_encode(self, columns: Optional[Union[str, List[str]]] = None, normalize: bool = True) -> 'CleanEasy':
        """Apply frequency encoding to categorical columns."""
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            freq = self.df[col].value_counts(normalize=normalize)
            new_col = f"{col}_freq"
            self.df[new_col] = self.df[col].map(freq)
            self.logger.info(f"Frequency encoded {col} to {new_col} (normalize={normalize})")
            self.results[f"{col}_freq"] = freq.to_dict()
        self.cleaning_steps.append("Applied frequency encoding")
        return self

    def check_skewness(self, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, float]:
        """Compute skewness for numeric columns."""
        columns = self._get_columns(columns, numeric=True)
        results = {}
        for col in columns:
            skew = self.df[col].skew()
            results[col] = skew
            self.logger.info(f"Skewness for {col}: {skew:.4f}")
        self.results['skewness'] = results
        self.cleaning_steps.append("Checked skewness")
        return results

    def remove_highly_correlated(self, threshold: float = 0.8, method: Literal['pearson', 'spearman', 'kendall'] = 'pearson') -> 'CleanEasy':
        """Drop numeric columns with correlation above threshold."""
        corr_matrix = self.df.select_dtypes(include=[np.number]).corr(method=method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        initial_cols = len(self.df.columns)
        self.df = self.df.drop(columns=to_drop)
        self.logger.info(f"Dropped {initial_cols - len(self.df.columns)} highly correlated columns (method={method}, threshold={threshold})")
        self.results['correlated_columns_dropped'] = to_drop
        self.cleaning_steps.append(f"Removed highly correlated columns (threshold={threshold})")
        return self

    def impute_mean(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                value = self.df[col].mean()
                self.df[col].fillna(value, inplace=True)
                self.logger.info(f"Imputed {col} with mean: {value}")
        self.cleaning_steps.append("Imputed missing values with mean")
        return self

    def impute_median(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                value = self.df[col].median()
                self.df[col].fillna(value, inplace=True)
                self.logger.info(f"Imputed {col} with median: {value}")
        self.cleaning_steps.append("Imputed missing values with median")
        return self

    def impute_mode(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns)
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                value = self.df[col].mode()[0]
                self.df[col].fillna(value, inplace=True)
                self.logger.info(f"Imputed {col} with mode: {value}")
        self.cleaning_steps.append("Imputed missing values with mode")
        return self

    def impute_constant(self, columns: Optional[Union[str, List[str]]], value: any) -> 'CleanEasy':
        columns = self._get_columns(columns)
        for col in columns:
            self.df[col].fillna(value, inplace=True)
            self.logger.info(f"Imputed {col} with constant: {value}")
        self.cleaning_steps.append(f"Imputed missing values with constant {value}")
        return self

    def impute_forward_fill(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns)
        for col in columns:
            self.df[col].fillna(method='ffill', inplace=True)
            self.logger.info(f"Imputed {col} with forward fill")
        self.cleaning_steps.append("Imputed missing values with forward fill")
        return self

    def impute_backward_fill(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns)
        for col in columns:
            self.df[col].fillna(method='bfill', inplace=True)
            self.logger.info(f"Imputed {col} with backward fill")
        self.cleaning_steps.append("Imputed missing values with backward fill")
        return self

    def impute_interpolate(self, columns: Optional[Union[str, List[str]]] = None, method: str = 'linear') -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            self.df[col] = self.df[col].interpolate(method=method)
            self.logger.info(f"Imputed {col} with interpolation ({method})")
        self.cleaning_steps.append(f"Imputed missing values with interpolation ({method})")
        return self

    def drop_missing_rows(self, columns: Optional[Union[str, List[str]]] = None, threshold: float = 0.5) -> 'CleanEasy':
        initial_rows = len(self.df)
        if columns:
            columns = self._get_columns(columns)
            self.df = self.df.dropna(subset=columns)
        else:
            self.df = self.df.dropna(thresh=int(len(self.df.columns) * (1 - threshold)))
        self.logger.info(f"Dropped {initial_rows - len(self.df)} rows with missing values")
        self.cleaning_steps.append("Dropped rows with missing values")
        return self

    def drop_missing_columns(self, threshold: float = 0.5) -> 'CleanEasy':
        initial_cols = len(self.df.columns)
        self.df = self.df.loc[:, self.df.isnull().mean() < threshold]
        self.logger.info(f"Dropped {initial_cols - len(self.df.columns)} columns with high missing values")
        self.cleaning_steps.append("Dropped columns with high missing values")
        return self

    def remove_outliers_iqr(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_rows = len(self.df)
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            self.logger.info(f"Removed {initial_rows - len(self.df)} outliers from {col} using IQR")
        self.cleaning_steps.append("Removed outliers using IQR")
        return self

    def cap_outliers_iqr(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            self.logger.info(f"Capped outliers in {col} using IQR")
        self.cleaning_steps.append("Capped outliers using IQR")
        return self

    def remove_outliers_zscore(self, columns: Optional[Union[str, List[str]]] = None, threshold: float = 3) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            mask = self.df[col].notnull()
            if len(z_scores) == len(self.df):
                mask = z_scores < threshold
            initial_rows = len(self.df)
            self.df = self.df[mask]
            self.logger.info(f"Removed {initial_rows - len(self.df)} outliers from {col} using Z-score")
        self.cleaning_steps.append("Removed outliers using Z-score")
        return self

    def cap_outliers_zscore(self, columns: Optional[Union[str, List[str]]] = None, threshold: float = 3) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            self.logger.info(f"Capped outliers in {col} using Z-score")
        self.cleaning_steps.append("Capped outliers using Z-score")
        return self

    def remove_outliers_dbscan(self, columns: Optional[Union[str, List[str]]] = None, eps: float = 0.5, min_samples: int = 5) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        if columns:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            X = self.df[columns].dropna()
            labels = db.fit_predict(X)
            mask = labels != -1
            initial_rows = len(self.df)
            self.df = self.df.loc[self.df.index.isin(X.index[mask])]
            self.logger.info(f"Removed {initial_rows - len(self.df)} outliers from {columns} using DBSCAN")
        self.cleaning_steps.append("Removed outliers using DBSCAN")
        return self

    def lowercase_text(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            self.df[col] = self.df[col].str.lower()
            self.logger.info(f"Converted {col} to lowercase")
        self.cleaning_steps.append("Converted text to lowercase")
        return self

    def remove_special_chars(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        def clean_text(text):
            return re.sub(r'[^a-zA-Z0-9\s]', '', str(text)) if pd.notnull(text) else text
        for col in columns:
            self.df[col] = self.df[col].apply(clean_text)
            self.logger.info(f"Removed special characters from {col}")
        self.cleaning_steps.append("Removed special characters from text")
        return self

    def trim_whitespace(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            self.df[col] = self.df[col].str.strip()
            self.logger.info(f"Trimmed whitespace from {col}")
        self.cleaning_steps.append("Trimmed whitespace from text")
        return self

    def replace_text(self, columns: Optional[Union[str, List[str]]], pattern: str, replacement: str) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            self.df[col] = self.df[col].str.replace(pattern, replacement, regex=True)
            self.logger.info(f"Replaced pattern {pattern} with {replacement} in {col}")
        self.cleaning_steps.append(f"Replaced text pattern {pattern}")
        return self

    def remove_numbers(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            self.df[col] = self.df[col].str.replace(r'\d+', '', regex=True)
            self.logger.info(f"Removed numbers from {col}")
        self.cleaning_steps.append("Removed numbers from text")
        return self

    def lemmatize_text(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        lemmatizer = WordNetLemmatizer()
        for col in columns:
            self.df[f"{col}_lemmatized"] = self.df[col].apply(
                lambda x: [lemmatizer.lemmatize(word) for word in word_tokenize(str(x))] if pd.notnull(x) else []
            )
            self.logger.info(f"Lemmatized text in {col}")
        self.cleaning_steps.append("Lemmatized text columns")
        return self

    def parse_dates(self, columns: Optional[Union[str, List[str]]], format: Optional[str] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], format=format, errors='coerce')
                self.logger.info(f"Parsed {col} to datetime")
            except Exception as e:
                self.logger.warning(f"Failed to parse {col} as datetime: {e}")
        self.cleaning_steps.append("Parsed date columns")
        return self

    def extract_year(self, columns: Optional[Union[str, List[str]]], new_column_suffix: str = '_year') -> 'CleanEasy':
        columns = self._get_columns(columns, dtype='datetime64[ns]')
        for col in columns:
            new_col = f"{col}{new_column_suffix}"
            self.df[new_col] = self.df[col].dt.year
            self.logger.info(f"Extracted year from {col} to {new_col}")
        self.cleaning_steps.append("Extracted year from datetime columns")
        return self

    def extract_month(self, columns: Optional[Union[str, List[str]]], new_column_suffix: str = '_month') -> 'CleanEasy':
        columns = self._get_columns(columns, dtype='datetime64[ns]')
        for col in columns:
            new_col = f"{col}{new_column_suffix}"
            self.df[new_col] = self.df[col].dt.month
            self.logger.info(f"Extracted month from {col} to {new_col}")
        self.cleaning_steps.append("Extracted month from datetime columns")
        return self

    def extract_quarter(self, columns: Optional[Union[str, List[str]]], new_column_suffix: str = '_quarter') -> 'CleanEasy':
        columns = self._get_columns(columns, dtype='datetime64[ns]')
        for col in columns:
            new_col = f"{col}{new_column_suffix}"
            self.df[new_col] = self.df[col].dt.quarter
            self.logger.info(f"Extracted quarter from {col} to {new_col}")
        self.cleaning_steps.append("Extracted quarter from datetime columns")
        return self

    def standardize_date_format(self, columns: Optional[Union[str, List[str]]], output_format: str = '%Y-%m-%d') -> 'CleanEasy':
        columns = self._get_columns(columns, dtype='datetime64[ns]')
        for col in columns:
            self.df[col] = self.df[col].dt.strftime(output_format)
            self.logger.info(f"Standardized {col} to format {output_format}")
        self.cleaning_steps.append(f"Standardized dates to {output_format}")
        return self

    def label_encode(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col].astype(str))
            self.encoders[col] = encoder
            self.logger.info(f"Label encoded {col}")
        self.cleaning_steps.append("Applied label encoding")
        return self

    def one_hot_encode(self, columns: Optional[Union[str, List[str]]] = None, drop_first: bool = True) -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=drop_first)
        self.logger.info(f"One-hot encoded {columns}")
        self.cleaning_steps.append("Applied one-hot encoding")
        return self

    def merge_rare_categories(self, columns: Optional[Union[str, List[str]]], threshold: float = 0.05, other: str = 'Other') -> 'CleanEasy':
        columns = self._get_columns(columns, dtype=object)
        for col in columns:
            freq = self.df[col].value_counts(normalize=True)
            rare = freq[freq < threshold].index
            self.df[col] = self.df[col].replace(rare, other)
            self.logger.info(f"Merged rare categories in {col} to {other}")
        self.cleaning_steps.append("Merged rare categories")
        return self

    def drop_duplicates(self, columns: Optional[Union[str, List[str]]] = None, keep: Literal['first', 'last', False] = 'first') -> 'CleanEasy':
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=columns, keep=keep)
        self.logger.info(f"Dropped {initial_rows - len(self.df)} duplicate rows")
        self.cleaning_steps.append("Dropped duplicate rows")
        return self

    def identify_duplicates(self, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        duplicates = self.df[self.df.duplicated(subset=columns, keep=False)]
        self.logger.info(f"Identified {len(duplicates)} duplicate rows")
        return duplicates

    def standardize_numeric(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.logger.info(f"Standardized {columns}")
        self.cleaning_steps.append("Standardized numeric columns")
        return self

    def normalize_numeric(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.logger.info(f"Normalized {columns}")
        self.cleaning_steps.append("Normalized numeric columns")
        return self

    def bin_numeric(self, columns: Optional[Union[str, List[str]]], bins: int = 5) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            self.df[f"{col}_binned"] = pd.cut(self.df[col], bins=bins, labels=False)
            self.logger.info(f"Binned {col} into {bins} bins")
        self.cleaning_steps.append(f"Binned numeric columns into {bins} bins")
        return self

    def log_transform(self, columns: Optional[Union[str, List[str]]] = None) -> 'CleanEasy':
        columns = self._get_columns(columns, numeric=True)
        for col in columns:
            self.df[col] = np.log1p(self.df[col])
            self.logger.info(f"Applied log transformation to {col}")
        self.cleaning_steps.append("Applied log transformation")
        return self

    def check_data_types(self) -> Dict[str, str]:
        types = get_column_types(self.df)
        self.logger.info("Checked data types")
        return types

    def check_missing_proportion(self) -> Dict[str, float]:
        missing = self.df.isnull().mean().to_dict()
        self.logger.info("Checked missing value proportions")
        return missing

    def check_normality(self, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, float]:
        columns = self._get_columns(columns, numeric=True)
        results = {}
        for col in columns:
            stat, p = stats.shapiro(self.df[col].dropna())
            results[col] = p
            self.logger.info(f"Normality test for {col}: p-value = {p}")
        self.cleaning_steps.append("Performed normality test")
        return results

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df

    def get_cleaning_log(self) -> List[str]:
        return self.cleaning_steps

    def get_results(self) -> Dict:
        """Return stored results from cleaning operations."""
        return self.results

    def _get_columns(self, columns: Optional[Union[str, List[str]]], numeric: bool = False, dtype: any = None) -> List[str]:
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            if numeric:
                columns = self.df.select_dtypes(include=[np.number]).columns
            elif dtype:
                columns = self.df.select_dtypes(include=[dtype]).columns
            else:
                columns = self.df.columns
        return [col for col in columns if col in self.df.columns]

    def auto_clean(self, impute_method: Literal["mean", "median", "mode", "ffill", "bfill", "knn", "interpolate", "none"] = "median",
                   outlier_method: Literal["iqr", "zscore", "isolation_forest", "dbscan", "none"] = "iqr",
                   text_clean: bool = True, date_parse: bool = True,
                   categorical_encode: Literal["label", "onehot", "frequency", "none"] = "none") -> pd.DataFrame:
        self.logger.info("Starting auto-clean process")
        if impute_method != "none":
            getattr(self, f"impute_{impute_method}")()
        if outlier_method != "none":
            method = "remove_outliers_" + outlier_method
            getattr(self, method)()
        if text_clean:
            self.lowercase_text().remove_special_chars().trim_whitespace().remove_numbers().tokenize_text().lemmatize_text()
        if date_parse:
            self.parse_dates().extract_year().extract_month().extract_day_of_week().extract_quarter()
        if categorical_encode != "none":
            getattr(self, f"{categorical_encode}_encode")()
        self.drop_duplicates()
        self.remove_highly_correlated()
        self.check_skewness()
        self.logger.info("Auto-clean process completed")
        return self.get_cleaned_data()