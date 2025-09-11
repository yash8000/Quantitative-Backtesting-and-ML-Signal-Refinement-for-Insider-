import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def print_section_header(section):
    print(f"\n{'='*80}")
    print(f"=== {section} ===")
    print(f"{'='*80}\n")

# Read the non-derivatives dataset
print_section_header("Loading Non-Derivatives Data")
df = pd.read_csv('nse_non_derivatives_cleaned_final.csv')
print(f"Total records: {len(df):,}")

# Define common missing/inconsistent values
missing_values = [
    '-', "'", "'-'", "' - '", "'-", "-'",  # Various combinations of dashes and quotes
    'Nil', 'NIL', 'nil',
    'NA', 'N/A', 'n/a',
    'null', 'NULL',
    '',  # empty string
    ' ',  # space
    '–',  # en dash
    '—',  # em dash
]

print_section_header("Missing and Inconsistent Value Analysis")

for column in df.columns:
    print(f"\n=== {column} ===")
    
    # 1. Check for NULL/NaN values
    null_count = df[column].isna().sum()
    null_pct = (null_count / len(df) * 100).round(2)
    print(f"NULL/NaN values: {null_count:,} ({null_pct}%)")
    
    # 2. Check for common missing value indicators
    missing_counts = {}
    for val in missing_values:
        count = (df[column] == val).sum()
        if count > 0:
            missing_counts[val] = count
    
    if missing_counts:
        print("\nCommon missing value indicators found:")
        for val, count in missing_counts.items():
            pct = (count / len(df) * 100).round(2)
            print(f"'{val}': {count:,} ({pct}%)")
    
    # 3. Check for special characters
    
    
    # 4. Check for whitespace issues
    if df[column].dtype == 'object':
        leading_space = df[column].astype(str).str.startswith(' ').sum()
        trailing_space = df[column].astype(str).str.endswith(' ').sum()
        if leading_space > 0 or trailing_space > 0:
            print("\nWhitespace issues:")
            if leading_space > 0:
                print(f"Leading spaces: {leading_space:,} ({(leading_space/len(df)*100):.2f}%)")
            if trailing_space > 0:
                print(f"Trailing spaces: {trailing_space:,} ({(trailing_space/len(df)*100):.2f}%)")
    
    # 5. For numeric columns, check for suspicious values
    if pd.api.types.is_numeric_dtype(df[column]):
        zeros = (df[column] == 0).sum()
        negatives = (df[column] < 0).sum()
        if zeros > 0 or negatives > 0:
            print("\nSuspicious numeric values:")
            if zeros > 0:
                print(f"Zeros: {zeros:,} ({(zeros/len(df)*100):.2f}%)")
            if negatives > 0:
                print(f"Negative values: {negatives:,} ({(negatives/len(df)*100):.2f}%)")
    
    # 6. Check for mixed data types in string columns
    if df[column].dtype == 'object':
        # Try to convert to numeric
        numeric_mask = pd.to_numeric(df[column], errors='coerce').notna()
        numeric_count = numeric_mask.sum()
        if 0 < numeric_count < len(df):
            print(f"\nMixed data types: {numeric_count:,} numeric values in string column ({(numeric_count/len(df)*100):.2f}%)")
    
    print("-" * 80)

print_section_header("Summary of Data Quality Issues")
print("Columns with significant quality issues (>1% problematic values):")
for column in df.columns:
    issues = []
    total_problematic = 0
    
    # Count all types of issues
    null_count = df[column].isna().sum()
    missing_count = sum((df[column] == val).sum() for val in missing_values)
    total_problematic = null_count + missing_count
    
    if total_problematic/len(df) > 0.01:  # More than 1% problematic
        pct = (total_problematic/len(df)*100).round(2)
        print(f"\n{column}: {total_problematic:,} issues ({pct}%)")
        if null_count > 0:
            print(f"  - NULL/NaN: {null_count:,}")
        if missing_count > 0:
            print(f"  - Missing indicators: {missing_count:,}") 