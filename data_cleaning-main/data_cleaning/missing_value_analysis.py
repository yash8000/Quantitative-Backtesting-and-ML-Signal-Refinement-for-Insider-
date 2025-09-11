import pandas as pd
import numpy as np
import re # Added for detailed analysis of unknown categories

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read the deduplicated dataset
df = pd.read_csv('nse_cleaned.csv')

def print_section_header(section):
    print(f"\n{'='*80}")
    print(f"=== {section} ===")
    print(f"{'='*80}\n")

# 1. Basic Missing Value Statistics
print_section_header("1. Missing Values: Counts and Percents")
missing_counts = df.isna().sum()
missing_pct = df.isna().mean() * 100
missing_df = pd.concat([missing_counts, missing_pct], axis=1, keys=['MissingCount', 'MissingPct'])
missing_df_sorted = missing_df.sort_values('MissingCount', ascending=False)
print(missing_df_sorted)

# 2. Analyze Columns with Missing Values
cols_with_na = missing_df_sorted[missing_df_sorted.MissingCount > 0].index.tolist()

print_section_header("2. Unique Values in Columns With Missing Data")
for col in cols_with_na:
    print(f"\n--- Unique values in {col} ---")
    unique_vals = df[col].dropna().unique()[:30]  # show first 30 unique values
    print(f"Count of unique non-null values: {len(df[col].dropna().unique())}")
    print("Sample of values:")
    print(unique_vals)

# 3. Patterns of Missingness
print_section_header("3. Patterns of Missingness")

# Get transaction type column
transaction_type_col = [col for col in df.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]

# Create indicator columns for missing values
for col in cols_with_na:
    df[f'{col}_ISNA'] = df[col].isna()

# Check if missingness is related to transaction type
print("\nMissingness by Transaction Type:")
for col in cols_with_na:
    print(f"\n--- Missing {col} by Transaction Type ---")
    missing_by_type = df.groupby(transaction_type_col)[f'{col}_ISNA'].mean() * 100
    print("Percentage missing in each transaction type:")
    print(missing_by_type)

# 4. Numeric Columns Analysis
print_section_header("4. Numeric Columns Analysis")

num_cols = [
    'NO. OF SECURITY (PRIOR)',
    'NO. OF SECURITIES (ACQUIRED/DISPLOSED)',
    'NO. OF SECURITY (POST)',
    '% POST',
    '% SHAREHOLDING (PRIOR)',
    'VALUE OF SECURITY (ACQUIRED/DISPLOSED)',
    'NOTIONAL VALUE(BUY)',
    'NOTIONAL VALUE(SELL)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE (BUY)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE  (SELL)',
]

# Find actual numeric columns that exist in the dataset
existing_num_cols = [col for col in num_cols if any(c for c in df.columns if col in c)]

for col in existing_num_cols:
    matching_cols = [c for c in df.columns if col in c]
    for matched_col in matching_cols:
        print(f"\n--- Analysis of numeric column {matched_col} ---")
        print("Sample of unique values:")
        print(df[matched_col].unique()[:10])
        print(f"Count of empty strings: {(df[matched_col] == '').sum()}")
        print(f"Count of '-' values: {(df[matched_col] == '-').sum()}")
        print(f"Count of 'Nil' values: {(df[matched_col].astype(str).str.lower() == 'nil').sum()}")

# 5. Completely Missing Analysis
print_section_header("5. Complete Missingness Analysis")

print("\nFully missing columns:")
print(missing_df_sorted[missing_df_sorted.MissingPct == 100])

print("\nRows completely empty:")
empty_rows = df.isna().all(axis=1).sum()
print(f"Empty rows count: {empty_rows}")

# 6. Date Columns Analysis
print_section_header("6. Date Columns Analysis")

date_cols = [
    'DATE OF ALLOTMENT/ACQUISITION FROM',
    'DATE OF ALLOTMENT/ACQUISITION TO',
    'DATE OF INITMATION TO COMPANY',
    'BROADCASTE DATE AND TIME'
]

existing_date_cols = [col for col in date_cols if any(c for c in df.columns if col in c)]

for col in existing_date_cols:
    matching_cols = [c for c in df.columns if col in c]
    for matched_col in matching_cols:
        print(f"\n--- Analysis of date column {matched_col} ---")
        print(f"Null count: {df[matched_col].isna().sum()}")
        print("Sample of unique non-null values:")
        print(df[matched_col].dropna().unique()[:10])

# 7. Derivative Fields Analysis
print_section_header("7. Derivative Fields Analysis")

deriv_cols = [
    'DERIVATIVE TYPE SECURITY',
    'DERIVATIVE CONTRACT SPECIFICATION',
    'NOTIONAL VALUE(BUY)',
    'NOTIONAL VALUE(SELL)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE (BUY)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE  (SELL)',
]

# Find derivative type security column
deriv_type_col = [col for col in df.columns if 'DERIVATIVE TYPE SECURITY' in col]
if deriv_type_col:
    deriv_type_col = deriv_type_col[0]
    existing_deriv_cols = [col for col in deriv_cols if any(c for c in df.columns if col in c)]
    
    for col in existing_deriv_cols:
        matching_cols = [c for c in df.columns if col in c]
        for matched_col in matching_cols:
            print(f"\n--- Missing {matched_col} vs. Derivative Type ---")
            # First create the missing value indicator
            df[f'{matched_col}_ISNA'] = df[matched_col].isna()
            # Then group by derivative type and calculate percentage
            missing_by_deriv = df.groupby(deriv_type_col)[f'{matched_col}_ISNA'].mean() * 100
            print("Percentage missing by derivative type:")
            print(missing_by_deriv)
            # Clean up temporary column
            df.drop(columns=[f'{matched_col}_ISNA'], inplace=True)

# 8. Check for Duplicates in Missing Values
print_section_header("8. Duplicates Among Missing Values")

for col in cols_with_na:
    na_dupes = df[df[col].isna()].duplicated().sum()
    print(f"{col}: {na_dupes} duplicated rows among missing entries")

# Save indicator columns for future reference
print_section_header("9. Saving Missing Value Indicators")
missing_indicators = df[[col for col in df.columns if col.endswith('_ISNA')]]
missing_indicators.to_csv('missing_value_indicators.csv', index=False)
print("Missing value indicators saved to 'missing_value_indicators.csv'")

# Add new section for Security Type Analysis
print_section_header("10. Security Type Analysis")

# Get the exact column names
security_prior_col = [col for col in df.columns if 'TYPE OF SECURITY (PRIOR)' in col][0]
deriv_type_col = [col for col in df.columns if 'DERIVATIVE TYPE SECURITY' in col][0]

# First get rows where TYPE OF SECURITY (PRIOR) is empty
empty_prior_mask = df[security_prior_col].isna()
empty_prior_count = empty_prior_mask.sum()

# Among these, count how many have DERIVATIVE TYPE SECURITY not '-'
empty_prior_rows = df[empty_prior_mask]
deriv_not_dash_in_empty = (empty_prior_rows[deriv_type_col] != '-').sum()

print(f"\nTotal rows where {security_prior_col} is empty: {empty_prior_count:,}")
print(f"Among these, count where {deriv_type_col} is not '-': {deriv_not_dash_in_empty:,}")
print(f"Percentage: {(deriv_not_dash_in_empty/empty_prior_count)*100:.2f}%")

# Show distribution of derivative types in these cases
print("\nDistribution of DERIVATIVE TYPE SECURITY values where TYPE OF SECURITY (PRIOR) is empty:")
print(empty_prior_rows[deriv_type_col].value_counts())

# Show examples where both fields are empty/dash
print("\nExamples where both TYPE OF SECURITY (PRIOR) is empty and DERIVATIVE TYPE SECURITY is '-':")
both_empty = df[
    (df[security_prior_col].isna()) & 
    (df[deriv_type_col] == '-')
]
print("\nCount of such rows:", len(both_empty))
print("\nFirst few examples:")
# Show relevant columns for understanding these cases
print(both_empty.head())

# Add distribution analysis of MODE OF ACQUISITION
print("\nDistribution of MODE OF ACQUISITION where both security types are empty:")
mode_col = [col for col in df.columns if 'MODE OF ACQUISITION' in col][0]
mode_dist = both_empty[mode_col].value_counts()
print("\nCounts:")
print(mode_dist)
print("\nPercentages:")
print(mode_dist / len(both_empty) * 100)

# Show examples where all three fields are empty/dash
print("\nExamples where all fields are empty/dash (security types and mode of acquisition):")
all_empty = df[
    (df[security_prior_col].isna()) & 
    (df[deriv_type_col] == '-') &
    (df[mode_col] == '-')
]
print("\nCount of such rows:", len(all_empty))

# 11. Analysis of Missing Quantities
print_section_header("11. Analysis of Missing Quantities")

# Find the exact column name for quantities
qty_col = [col for col in df.columns if 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)' in col][0]

# Get rows with missing quantities
missing_qty = df[df[qty_col].isna()]
print("\nFirst 10 rows with missing quantities:")
print(missing_qty.head(10))

# Print some statistics about missing quantities
print(f"\nTotal rows with missing quantities: {len(missing_qty):,}")
print(f"Percentage of rows with missing quantities: {(len(missing_qty)/len(df))*100:.2f}%")

# Show distribution of transaction types for missing quantities
print("\nDistribution of transaction types where quantities are missing:")
print(missing_qty[transaction_type_col].value_counts())

# 12. Analysis of XBRL Missing Row
print_section_header("12. XBRL Missing Row Analysis")

# Find the XBRL column
xbrl_col = [col for col in df.columns if 'XBRL' in col][0]

# Get the row where XBRL is missing
xbrl_missing = df[df[xbrl_col].isna()]
print("\nRow where XBRL is missing:")
print(xbrl_missing)

# Show all column values for this specific row
print("\nDetailed view of the row with missing XBRL:")
for col in df.columns:
    print(f"\n{col}:")
    print(xbrl_missing[col].values[0])

# 13. Column Coverage Analysis
print_section_header("13. Column Coverage Analysis")

# Define all expected columns
expected_columns = [
    'SYMBOL',
    'COMPANY',
    'REGULATION',
    'NAME OF THE ACQUIRER/DISPOSER',
    'CATEGORY OF PERSON',
    'TYPE OF SECURITY (PRIOR)',
    'NO. OF SECURITY (PRIOR)',
    '% SHAREHOLDING (PRIOR)',
    'NO. OF SECURITIES (ACQUIRED/DISPLOSED)',
    'VALUE OF SECURITY (ACQUIRED/DISPLOSED)',
    'ACQUISITION/DISPOSAL TRANSACTION TYPE',
    'TYPE OF SECURITY (POST)',
    'NO. OF SECURITY (POST)',
    '% POST',
    'DATE OF ALLOTMENT/ACQUISITION FROM',
    'DATE OF ALLOTMENT/ACQUISITION TO',
    'DATE OF INITMATION TO COMPANY',
    'MODE OF ACQUISITION',
    'DERIVATIVE TYPE SECURITY',
    'DERIVATIVE CONTRACT SPECIFICATION',
    'NOTIONAL VALUE(BUY)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE (BUY)',
    'NOTIONAL VALUE(SELL)',
    'NUMBER OF UNITS/CONTRACT LOT SIZE  (SELL)',
    'EXCHANGE',
    'REMARK',
    'BROADCASTE DATE AND TIME',
    'XBRL'
]

print("Column Coverage Analysis:")
print("\nColumns present in dataset:")
for col in expected_columns:
    matching_cols = [c for c in df.columns if col in c]
    if matching_cols:
        print(f"✓ {col}")
        # Print missing value statistics for each column
        for matched_col in matching_cols:
            missing_count = df[matched_col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"   - Missing values: {missing_count:,} ({missing_pct:.2f}%)")
    else:
        print(f"✗ {col} (Not found in dataset)")

# 14. Analysis of Nil Prior Security Records
print_section_header("14. Analysis of Records with 'Nil' Prior Securities")

# Get exact column names
prior_security_col = [col for col in df.columns if 'NO. OF SECURITY (PRIOR)' in col][0]
transaction_type_col = [col for col in df.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]
mode_col = [col for col in df.columns if 'MODE OF ACQUISITION' in col][0]
post_security_col = [col for col in df.columns if 'NO. OF SECURITY (POST)' in col][0]
acquired_disposed_col = [col for col in df.columns if 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)' in col][0]
category_col = [col for col in df.columns if 'CATEGORY OF PERSON' in col][0]
prior_shareholding_col = [col for col in df.columns if '% SHAREHOLDING (PRIOR)' in col][0]

# 1. Transaction Type Analysis for Nil rows
print("\n1. Transaction Types for 'Nil' Prior Security Records:")
nil_trans_types = df.loc[df[prior_security_col] == "Nil", transaction_type_col].value_counts()
nil_trans_pct = (nil_trans_types / nil_trans_types.sum() * 100).round(2)
print("\nDistribution (with percentages):")
for trans_type, count in nil_trans_types.items():
    print(f"{trans_type}: {count} ({nil_trans_pct[trans_type]}%)")

# 2. Mode of Acquisition Analysis
print("\n2. Mode of Acquisition for 'Nil' Prior Security Records:")
nil_modes = df.loc[df[prior_security_col] == "Nil", mode_col].value_counts()
nil_modes_pct = (nil_modes / nil_modes.sum() * 100).round(2)
print("\nDistribution (with percentages):")
for mode, count in nil_modes.items():
    print(f"{mode}: {count} ({nil_modes_pct[mode]}%)")

# 3. Compare Post Security with Acquired/Disposed
print("\n3. Comparison of Post Security vs Acquired/Disposed (First 20 rows):")
comparison_df = df.loc[
    df[prior_security_col] == "Nil",
    [acquired_disposed_col, post_security_col, transaction_type_col]
].head(20)
print("\nDetailed view of sample rows:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(comparison_df.to_string())
