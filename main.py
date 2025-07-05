import pandas as pd
import numpy as np
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
            # Format name_tokens as a comma-separated string for readability
            value_str = ', '.join([str(item) for item in value])
            result.append(f"{'  ' * indent}{key_str}: {value_str}")
        else:
            # Convert NumPy types to native Python types
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

# Apply requested functions
cleaner.parse_dates(columns=['date'])
cleaner = (cleaner
    .impute_knn(columns=['age', 'salary'], n_neighbors=3, weights='distance')
    .remove_outliers_isolation_forest(columns=['age'], contamination=0.2, random_state=42)
    .tokenize_text(columns=['name'], lowercase=True)
    .extract_day_of_week(columns=['date'], return_numeric=True)
    .frequency_encode(columns=['category'], normalize=True)
)

# Store skewness results separately
skewness_results = cleaner.check_skewness(columns=['age', 'salary'])

# Continue method chain
cleaned_df = (cleaner
    .remove_highly_correlated(threshold=0.8, method='pearson')
    .get_cleaned_data()
)

# Format and print results
print(f"\n{Fore.GREEN}=== Cleaned DataFrame ==={Style.RESET_ALL}")
# Convert name_tokens to string for better table display
cleaned_df_display = cleaned_df.copy()
cleaned_df_display['name_tokens'] = cleaned_df_display['name_tokens'].apply(lambda x: ', '.join(x))
print(tabulate(cleaned_df_display, headers='keys', tablefmt='psql', showindex=True, floatfmt='.2f'))

print(f"\n{Fore.GREEN}=== Cleaning Steps ==={Style.RESET_ALL}")
for i, step in enumerate(cleaner.get_cleaning_log(), 1):
    print(f"{i}. {step}")

print(f"\n{Fore.GREEN}=== Skewness Results ==={Style.RESET_ALL}")
# Convert NumPy floats to regular floats
skewness_formatted = {k: float(v) for k, v in skewness_results.items()}
for col, value in skewness_formatted.items():
    print(f"{Fore.CYAN}{col}{Style.RESET_ALL}: {value:.4f}")

print(f"\n{Fore.GREEN}=== All Results ==={Style.RESET_ALL}")
# Convert NumPy types in results
results = cleaner.get_results()
for key, value in results.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            if isinstance(subvalue, (np.floating, np.integer)):
                results[key][subkey] = float(subvalue) if isinstance(subvalue, np.floating) else int(subvalue)
print(format_dict(results))