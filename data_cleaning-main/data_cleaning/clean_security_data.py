import pandas as pd
import numpy as np

# Read the deduplicated dataset
df = pd.read_csv('nse_deduplicated.csv')

def print_section_header(section):
    print(f"\n{'='*80}")
    print(f"=== {section} ===")
    print(f"{'='*80}\n")

# Get exact column names
security_acquired_col = [col for col in df.columns if 'TYPE OF SECURITY (ACQUIRED/DISPLOSED)' in col][0]
exchange_col = [col for col in df.columns if 'EXCHANGE' in col][0]
security_prior_col = [col for col in df.columns if 'TYPE OF SECURITY (PRIOR)' in col][0]
deriv_type_col = [col for col in df.columns if 'DERIVATIVE TYPE SECURITY' in col][0]
trans_type_col = [col for col in df.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]
mode_col = [col for col in df.columns if 'MODE OF ACQUISITION' in col][0]

category_col = [col for col in df.columns if 'CATEGORY OF PERSON' in col][0]
regulation_col = [col for col in df.columns if 'REGULATION' in col][0]
prior_security_col = [col for col in df.columns if 'NO. OF SECURITY (PRIOR)' in col][0]
acquired_disposed_col = [col for col in df.columns if 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)' in col][0]
transaction_type_col = [col for col in df.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]

print_section_header("1. Initial Data Stats")
print(f"Initial shape: {df.shape}")

# 1. Drop TYPE OF SECURITY (ACQUIRED/DISPLOSED)
print_section_header("2. Dropping Acquired Security Type")
print(f"Dropping column: {security_acquired_col}")
print(f"Missing percentage before drop: {df[security_acquired_col].isna().mean()*100:.2f}%")
df = df.drop(columns=[security_acquired_col])

# 2. Handle EXCHANGE missing values
print_section_header("3. Handling Exchange Missing Values")
print("Missing Exchange by Transaction Type:")
# Fix: Create missing indicator first, then group
df['exchange_missing'] = df[exchange_col].isna()
missing_exchange = df.groupby(trans_type_col)['exchange_missing'].mean() * 100
print(missing_exchange)
df = df.drop(columns=['exchange_missing'])

# Impute "OFF MARKET" for missing exchange values
df[exchange_col] = df[exchange_col].fillna("OFF MARKET")
print("\nAfter imputation, missing Exchange values:", df[exchange_col].isna().sum())

# 3. Handle TYPE OF SECURITY (PRIOR)
print_section_header("4. Handling Type of Security (Prior)")
missing_prior = df[df[security_prior_col].isna()]
print(f"Total rows with missing prior security type: {len(missing_prior)}")

# a. Handle derivative cases
deriv_missing = missing_prior[missing_prior[deriv_type_col] != '-']
print(f"\nDerivative cases among missing (deriv_type != '-'): {len(deriv_missing)}")

# b. Handle ESOP and market sale cases
esop_market_missing = missing_prior[
    (missing_prior[deriv_type_col] == '-') & 
    (missing_prior[mode_col].str.contains('ESOP|MARKET', case=False, na=False))
]
print(f"\nESOP/Market sale cases among missing: {len(esop_market_missing)}")

# Create a copy of the dataframe for cleaning
df_clean = df.copy()

# Apply the rules
# 1. For derivatives
mask_deriv = (df_clean[security_prior_col].isna()) & (df_clean[deriv_type_col] != '-')
df_clean.loc[mask_deriv, security_prior_col] = "Derivatives"

# 2. For ESOP/Market sales
mask_esop_market = (
    (df_clean[security_prior_col].isna()) & 
    (df_clean[deriv_type_col] == '-') & 
    (df_clean[mode_col].str.contains('ESOP|MARKET', case=False, na=False))
)
df_clean.loc[mask_esop_market, security_prior_col] = "Equity Shares"

# 3. Drop rows where almost everything is empty
mask_drop = (
    (df_clean[security_prior_col].isna()) & 
    (df_clean[deriv_type_col] == '-') & 
    ~(df_clean[mode_col].str.contains('ESOP|MARKET', case=False, na=False))
)
df_clean = df_clean[~mask_drop]

print_section_header("5. Final Statistics")
print("Initial shape:", df.shape)
print("Final shape:", df_clean.shape)
print(f"Rows dropped: {len(df) - len(df_clean)}")

print("\nMissing values in key columns:")
for col in [security_prior_col, exchange_col, deriv_type_col]:
    print(f"{col}: {df_clean[col].isna().sum():,} ({df_clean[col].isna().mean()*100:.2f}%)")

# Print summary of changes
print("\nSummary of changes made:")
print("1. Dropped column:", security_acquired_col)
print(f"2. Imputed {df[exchange_col].isna().sum():,} missing exchange values with 'OFF MARKET'")
print(f"3. Imputed {mask_deriv.sum():,} missing security types with 'Derivatives'")
print(f"4. Imputed {mask_esop_market.sum():,} missing security types with 'Equity Shares'")
print(f"5. Dropped {mask_drop.sum():,} rows with mostly missing data") 

print_section_header("6. Handling Missing Categories and Pledge Cases")

# 1. Handle missing categories for ESOP cases
esop_mask = (
    (df[category_col].isna() | (df[category_col] == '-')) & 
    ((df[mode_col].str.contains('ESOP', case=False, na=False)) |
     (df[regulation_col].str.contains('7\(2\)', case=False, na=False)))
)
print(f"\nFound {esop_mask.sum():,} ESOP-related cases with missing category")
df.loc[esop_mask, category_col] = "Employees"

# 2. Fix pledge cases where prior securities should match acquired/disposed
pledge_mask = df[transaction_type_col].str.contains('PLEDGE|REVOKE', case=False, na=False)
pledge_cases = df[pledge_mask]
print(f"\nFound {len(pledge_cases):,} pledge/revoke cases")

# Check for mismatches
mismatch_mask = (
    pledge_mask & 
    (df[prior_security_col] != df[acquired_disposed_col]) &
    (df[prior_security_col] == "Nil")
)
print(f"Found {mismatch_mask.sum():,} cases where prior securities don't match acquired/disposed for pledges")

if mismatch_mask.sum() > 0:
    print("\nFixing pledge cases where prior securities are Nil:")
    df.loc[mismatch_mask, prior_security_col] = df.loc[mismatch_mask, acquired_disposed_col]

# Print summary of changes
print("\nSummary of changes made:")
print(f"1. Set category as 'Employees' for {esop_mask.sum():,} ESOP-related cases")
print(f"2. Fixed {mismatch_mask.sum():,} pledge cases where prior securities were Nil")

# Print remaining missing categories
missing_categories = df[category_col].isna() | (df[category_col] == '-')
print(f"\nRemaining missing categories: {missing_categories.sum():,} ({missing_categories.mean()*100:.2f}%)")

# Print remaining Nil prior securities
nil_prior = (df[prior_security_col] == "Nil")
print(f"Remaining Nil prior securities: {nil_prior.sum():,} ({nil_prior.mean()*100:.2f}%)")

# Save the final cleaned dataset
print_section_header("7. Saving Final Cleaned Dataset")
df.to_csv('nse_cleaned.csv', index=False)
print("Final cleaned dataset saved as 'nse_cleaned.csv'")

# Print final statistics
print("\nFinal Dataset Statistics:")
print(f"Total rows: {len(df):,}")
print("\nMissing values in key columns:")
key_cols = [category_col, prior_security_col, exchange_col, security_prior_col]
for col in key_cols:
    missing = df[col].isna().sum()
    print(f"{col}: {missing:,} ({missing/len(df)*100:.2f}%)") 