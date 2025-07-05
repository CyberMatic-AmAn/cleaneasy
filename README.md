![Alt text](https://pbs.twimg.com/media/GvFreezXAAEN0ie?format=jpg&name=900x900)

# CleanEasy

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.2-brightgreen)](https://github.com/yourusername/cleaneasy)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](https://pytest.org)

**CleanEasy** is a powerful, user-friendly Python library designed to simplify data cleaning and preprocessing for data scientists and analysts. Built on top of `pandas`, `numpy`, `scikit-learn`, and `nltk`, it provides a chainable API to handle common tasks like missing value imputation, outlier detection, text processing, date manipulation, and categorical encoding. With detailed logging and formatted output, CleanEasy makes data preparation intuitive, transparent, and visually appealing.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact and Support](#contact-and-support)
- [FAQ](#faq)
- [Roadmap](#roadmap)

## Introduction

`CleanEasy` streamlines the data cleaning process by offering a unified interface for a wide range of preprocessing tasks. Whether you're working with DataFrames, NumPy arrays, lists, dictionaries, or CSV files, CleanEasy handles data conversion, cleaning, and validation with ease. Its method-chaining API allows you to build complex cleaning pipelines in a readable, maintainable way, while detailed logs and formatted outputs (using `tabulate` and `colorama`) ensure clarity and usability.

Key highlights:
- Supports multiple data input formats.
- Extensive methods for imputation, outlier removal, text processing, and encoding.
- Built-in validation tools for skewness, normality, and correlations.
- Auto-cleaning pipeline for quick preprocessing.
- Pretty-printed output for easy interpretation.

## Features

`CleanEasy` offers a rich set of tools for data preprocessing:

### Data Input and Conversion
- Accepts `pandas.DataFrame`, `numpy.ndarray`, lists, dictionaries, or CSV file paths.
- Automatically converts inputs to a `pandas.DataFrame` using `convert_to_dataframe`.

### Missing Value Imputation
- **KNN Imputation**: `impute_knn` for numeric columns using k-nearest neighbors.
- **Statistical Imputation**: `impute_mean`, `impute_median`, `impute_mode`.
- **Time-Series Imputation**: `impute_forward_fill`, `impute_backward_fill`, `impute_interpolate`.
- **Constant Imputation**: `impute_constant` with a user-specified value.
- **Drop Missing**: `drop_missing_rows`, `drop_missing_columns` based on thresholds.

### Outlier Detection and Handling
- **Isolation Forest**: `remove_outliers_isolation_forest` for robust outlier removal.
- **IQR**: `remove_outliers_iqr` and `cap_outliers_iqr` for interquartile range-based handling.
- **Z-Score**: `remove_outliers_zscore` and `cap_outliers_zscore` for standard deviation-based handling.
- **DBSCAN**: `remove_outliers_dbscan` for clustering-based outlier detection.

### Text Processing
- **Tokenization**: `tokenize_text` using NLTK's word tokenizer.
- **Lemmatization**: `lemmatize_text` with WordNet lemmatizer.
- **Cleaning**: `lowercase_text`, `remove_special_chars`, `trim_whitespace`, `remove_numbers`, `replace_text`.

### Date and Time Processing
- **Parsing**: `parse_dates` to convert strings to datetime.
- **Feature Extraction**: `extract_year`, `extract_month`, `extract_quarter`, `extract_day_of_week`.
- **Formatting**: `standardize_date_format` for consistent date strings.

### Categorical Encoding
- **Frequency Encoding**: `frequency_encode` for value counts.
- **Label Encoding**: `label_encode` for ordinal categories.
- **One-Hot Encoding**: `one_hot_encode` with drop-first option.
- **Rare Categories**: `merge_rare_categories` to group infrequent categories.

### Data Validation
- **Skewness**: `check_skewness` for numeric columns.
- **Normality**: `check_normality` using Shapiro-Wilk test.
- **Missing Values**: `check_missing_proportion` for column-wise missing ratios.
- **Unique Values**: `check_unique_values` for distinct counts.
- **Correlations**: `check_correlation` and `remove_highly_correlated` for numeric columns.

### Other Utilities
- **Duplicates**: `drop_duplicates` and `identify_duplicates`.
- **Scaling**: `standardize_numeric` (z-score) and `normalize_numeric` (min-max).
- **Binning**: `bin_numeric` for discretizing numeric columns.
- **Log Transformation**: `log_transform` for handling skewed data.
- **Auto-Cleaning**: `auto_clean` for a customizable, one-step pipeline.

### Output and Logging
- Detailed logging of all operations with customizable log levels.
- Formatted console output with tables (`tabulate`) and colors (`colorama`).
- Results storage in `get_results()` for inspection.

## Installation

### Prerequisites
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Virtual Environment**: Recommended for dependency isolation
- **Terminal**: For running commands (e.g., Windows Terminal, VS Code, or bash)

### Step-by-Step Instructions
1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/CyberMatic-AmAn/cleaneasy.git
   cd cleaneasy
   ```

2. **Create a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install Dependencies**
   Install required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `pandas>=1.5.0`
   - `numpy>=1.23.0`
   - `scipy>=1.9.0`
   - `scikit-learn>=1.1.0`
   - `nltk>=3.7`
   - `pytest>=7.0.0`
   - `tabulate>=0.8.9`
   - `colorama>=0.4.4`

4. **Download NLTK Data**
   Some methods (e.g., `tokenize_text`, `lemmatize_text`) require NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('wordnet')
   ```

5. **Install CleanEasy as a Package**
   Install the `cleaneasy` package locally to make it importable:
   ```bash
   pip install .
   ```

## Usage

### Basic Example
The `main.py` script demonstrates a typical cleaning pipeline. It processes a sample dataset with missing values, outliers, text, dates, and categorical data, producing formatted output.

```python
import pandas as pd
import json
from tabulate import tabulate
from colorama import init, Fore, Style
from cleaneasy import CleanEasy

# Initialize colorama for colored output
init()

def format_dict(d, indent=0):
    """Pretty-print a dictionary with indentation for nested structures."""
    result = []
    for key, value in d.items():
        key_str = f"{Fore.CYAN}{key}{Style.RESET_ALL}"
        if isinstance(value, dict):
            result.append(f"{'  ' * indent}{key_str}:")
            result.append(format_dict(value, indent + 1))
        elif isinstance(value, list) and key == 'name_tokens':
            value_str = ', '.join([str(item) for item in value])
            result.append(f"{'  ' * indent}{key_str}: {value_str}")
        else:
            if isinstance(value, (np.floating, np.integer)):
                value = float(value) if isinstance(value, np.floating) else int(value)
            result.append(f"{'  ' * indent}{key_str}: {value}")
    return '\n'.join(result)

# Sample data
data = {
    'name': ['John@Doe', 'Jane Smith!', None, 'Alice'],
    'age': [25, 30, 1000, None],
    'salary': [50000, None, 60000, 55000],
    'date': ['2023-01-01', '2023-02-02', 'invalid', '2023-03-03'],
    'category': ['A', 'B', 'A', 'C']
}
df = pd.DataFrame(data)

# Initialize CleanEasy
cleaner = CleanEasy(df, log_level='INFO')

# Apply cleaning steps
cleaner.parse_dates(columns=['date'])
cleaner = (cleaner
    .impute_knn(columns=['age', 'salary'], n_neighbors=3, weights='distance')
    .remove_outliers_isolation_forest(columns=['age'], contamination=0.2, random_state=42)
    .tokenize_text(columns=['name'], lowercase=True)
    .extract_day_of_week(columns=['date'], return_numeric=True)
    .frequency_encode(columns=['category'], normalize=True)
)

# Store skewness results
skewness_results = cleaner.check_skewness(columns=['age', 'salary'])

# Continue method chain
cleaned_df = (cleaner
    .remove_highly_correlated(threshold=0.8, method='pearson')
    .get_cleaned_data()
)

# Display results
print(f"\n{Fore.GREEN}=== Cleaned DataFrame ==={Style.RESET_ALL}")
cleaned_df_display = cleaned_df.copy()
cleaned_df_display['name_tokens'] = cleaned_df_display['name_tokens'].apply(lambda x: ', '.join(x))
print(tabulate(cleaned_df_display, headers='keys', tablefmt='psql', showindex=True, floatfmt='.2f'))

print(f"\n{Fore.GREEN}=== Cleaning Steps ==={Style.RESET_ALL}")
for i, step in enumerate(cleaner.get_cleaning_log(), 1):
    print(f"{i}. {step}")

print(f"\n{Fore.GREEN}=== Skewness Results ==={Style.RESET_ALL}")
skewness_formatted = {k: float(v) for k, v in skewness_results.items()}
for col, value in skewness_formatted.items():
    print(f"{Fore.CYAN}{col}{Style.RESET_ALL}: {value:.4f}")

print(f"\n{Fore.GREEN}=== All Results ==={Style.RESET_ALL}")
results = cleaner.get_results()
for key, value in results.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            if isinstance(subvalue, (np.floating, np.integer)):
                results[key][subkey] = float(subvalue) if isinstance(subvalue, np.floating) else int(subvalue)
print(format_dict(results))
```

### Example Output
Running `python main.py` produces:

```
2025-07-05 12:35:10,417 - CleanEasy - INFO - Initialized CleanEasy with data type: DataFrame
2025-07-05 12:35:10,421 - CleanEasy - INFO - Parsed date to datetime
2025-07-05 12:35:10,425 - CleanEasy - INFO - Imputed ['age', 'salary'] with KNN (n_neighbors=3, weights=distance)
2025-07-05 12:35:10,540 - CleanEasy - INFO - Removed 1 outliers from ['age'] using Isolation Forest (contamination=0.2)
2025-07-05 12:35:10,610 - CleanEasy - INFO - Tokenized text in name (lowercase=True)
2025-07-05 12:35:10,637 - CleanEasy - INFO - Extracted day of week from date to date_dayofweek (numeric=True)
2025-07-05 12:35:10,641 - CleanEasy - INFO - Frequency encoded category to category_freq (normalize=True)
2025-07-05 12:35:10,641 - CleanEasy - INFO - Skewness for age: 1.7314
2025-07-05 12:35:10,642 - CleanEasy - INFO - Skewness for salary: 1.7314
2025-07-05 12:35:10,645 - CleanEasy - INFO - Dropped 1 highly correlated columns (method=pearson, threshold=0.8)

=== Cleaned DataFrame ===
+----+------------+-------+------------+-----------+------------------+--------------+-----------------+
|    | name       |   age | date       | category  | name_tokens      | date_dayofweek |   category_freq |
|----+------------+-------+------------+-----------+------------------+--------------+-----------------|
|  0 | John@Doe   | 25.00 | 2023-01-01 | A         | john, @, doe     |            6 |            0.33 |
|  1 | Jane Smith!| 30.00 | 2023-02-02 | B         | jane, smith, !   |            3 |            0.33 |
|  3 | Alice      | 512.50| 2023-03-03 | C         | alice            |            4 |            0.33 |
+----+------------+-------+------------+-----------+------------------+--------------+-----------------+

=== Cleaning Steps ===
1. Parsed date columns
2. Imputed missing values with KNN (weights=distance)
3. Removed outliers using Isolation Forest
4. Tokenized text columns
5. Extracted day of week from datetime columns
6. Applied frequency encoding
7. Checked skewness
8. Removed highly correlated columns (threshold=0.8)

=== Skewness Results ===
age: 1.7314
salary: 1.7314

=== All Results ===
knn_imputation:
  columns: ['age', 'salary']
  n_neighbors: 3
  weights: distance
isolation_forest:
  columns: ['age']
  outliers_removed: 1
name_tokens: [john, @, doe], [jane, smith, !], [alice]
category_freq:
  A: 0.3333333333333333
  B: 0.3333333333333333
  C: 0.3333333333333333
skewness:
  age: 1.7314295926231227
  salary: 1.7314295926231076
correlated_columns_dropped: ['salary']
```

### Auto-Cleaning Example
Use `auto_clean` for a one-step pipeline:
```python
cleaner = CleanEasy(df, log_level='INFO')
cleaned_df = cleaner.auto_clean(
    impute_method='knn',
    outlier_method='isolation_forest',
    text_clean=True,
    date_parse=True,
    categorical_encode='frequency'
)
print(f"\n{Fore.GREEN}=== Auto-Cleaned DataFrame ==={Style.RESET_ALL}")
print(tabulate(cleaned_df, headers='keys', tablefmt='psql', showindex=True, floatfmt='.2f'))
```

## Project Structure

```
cleaneasy/
├── cleaneasy/
│   ├── __init__.py         # Package initialization and exports
│   ├── core.py            # Core CleanEasy class with cleaning methods
│   ├── utils.py           # Utility functions (e.g., convert_to_dataframe)
│   ├── validators.py      # Validation functions (e.g., check_skewness)
├── tests/
│   ├── __init__.py        # Test package initialization
│   ├── test_core.py       # Tests for core.py
│   ├── test_utils.py      # Tests for utils.py
│   ├── test_validators.py # Tests for validators.py
├── docs/
│   ├── conf.py            # Sphinx documentation configuration
│   ├── index.rst          # Sphinx documentation index
├── main.py                # Example script demonstrating usage
├── pyproject.toml         # Project metadata and build configuration
├── requirements.txt       # Dependencies
├── README.md              # This file
├── LICENSE                # License file (MIT)
```

## Testing

`CleanEasy` includes a test suite using `pytest` to ensure reliability.

1. **Install pytest**
   ```bash
   pip install pytest
   ```

2. **Run Tests**
   ```bash
   cd cleaneasy
   pytest tests/
   ```

   Tests cover:
   - Initialization and data conversion (`test_core.py`, `test_utils.py`)
   - Cleaning methods (e.g., `impute_knn`, `remove_outliers_isolation_forest`)
   - Validation functions (e.g., `check_skewness`, `check_normality`)

## Contributing

We welcome contributions to `CleanEasy`! To contribute:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/CyberMatic-AmAn/cleaneasy.git
   cd cleaneasy
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Add new features or fix bugs in `cleaneasy/`.
   - Update tests in `tests/`.
   - Document changes in `docs/` if necessary.

4. **Run Tests**
   Ensure all tests pass: `pytest tests/`.

5. **Submit a Pull Request**
   - Push your branch: `git push origin feature/your-feature-name`.
   - Open a pull request on GitHub with a clear description of changes.

6. **Report Issues**
   - Use the GitHub Issues page to report bugs or suggest features.
   - Include detailed descriptions and reproduction steps.

## License

`CleanEasy` is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact and Support

- **Email**: exehyper999@gmail.com
- **GitHub Issues**: [github.com/CyberMatic-AmAn/cleaneasy/issues](https://github.com/CyberMatic-AmAn/cleaneasy/issues)
- **Documentation**: [https://github.com/CyberMatic-AmAn/cleaneasy](https://github.com/CyberMatic-AmAn/cleaneasy)

For support, open an issue on GitHub or contact the maintainer directly.

## FAQ

### Why do I get an NLTK error when using `tokenize_text`?
Ensure NLTK data is downloaded:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

### Why is the output not colored?
- Verify `colorama` is installed: `pip show colorama`.
- Ensure your terminal supports ANSI colors (e.g., Windows Terminal, VS Code).
- Check that `colorama.init()` is called in `main.py`.

### How do I add a new cleaning method?
- Add the method to `cleaneasy/core.py` in the `CleanEasy` class.
- Ensure it returns `self` for method chaining.
- Update tests in `tests/test_core.py`.
- Document the method in `docs/` and this `README.md`.

### Can I use `CleanEasy` with large datasets?
Yes, but performance depends on the methods used (e.g., `impute_knn` and `remove_outliers_isolation_forest` can be computationally intensive). Test with a sample first.

