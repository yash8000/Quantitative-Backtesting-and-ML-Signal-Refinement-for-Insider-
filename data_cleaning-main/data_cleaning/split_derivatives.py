import pandas as pd
import numpy as np

def print_section_header(section):
    print(f"\n{'='*80}")
    print(f"=== {section} ===")
    print(f"{'='*80}\n")

# Read the cleaned dataset
print_section_header("Loading Data")
df = pd.read_csv('nse_cleaned.csv')
print(f"Total records loaded: {len(df):,}")

# Get exact column names
security_prior_col = [col for col in df.columns if 'TYPE OF SECURITY (PRIOR)' in col][0]
deriv_type_col = [col for col in df.columns if 'DERIVATIVE TYPE SECURITY' in col][0]
deriv_contract_col = [col for col in df.columns if 'DERIVATIVE CONTRACT SPECIFICATION' in col][0]

# Identify derivative transactions
print_section_header("Identifying Derivative Transactions")

# A transaction is considered a derivative if:
# 1. TYPE OF SECURITY (PRIOR) is 'Derivatives' OR
# 2. DERIVATIVE TYPE SECURITY is not '-' OR
# 3. DERIVATIVE CONTRACT SPECIFICATION is not '-'
derivative_mask = (
    (df[security_prior_col] == 'Derivatives') |
    (df[deriv_type_col] != '-') |
    (df[deriv_contract_col] != '-')
)

# Split the dataframes
derivatives_df = df[derivative_mask]
non_derivatives_df = df[~derivative_mask]

print("\nSplit Statistics:")
print(f"Derivative transactions: {len(derivatives_df):,} ({len(derivatives_df)/len(df)*100:.2f}%)")
print(f"Non-derivative transactions: {len(non_derivatives_df):,} ({len(non_derivatives_df)/len(df)*100:.2f}%)")

# Analyze derivative types
print("\nTypes of derivatives:")
print(derivatives_df[deriv_type_col].value_counts())

print("\nDerivative contract specifications:")
print(derivatives_df[deriv_contract_col].value_counts().head())

# Save the split datasets
print_section_header("Saving Split Datasets")

derivatives_df.to_csv('nse_derivatives.csv', index=False)
non_derivatives_df.to_csv('nse_non_derivatives.csv', index=False)

print("Datasets saved:")
print("1. nse_derivatives.csv - Contains all derivative transactions")
print("2. nse_non_derivatives.csv - Contains all non-derivative transactions")

# Print sample statistics for both datasets
print_section_header("Dataset Statistics")

def print_dataset_stats(name, data):
    print(f"\n{name} Statistics:")
    print(f"Total records: {len(data):,}")
    
    # Transaction type distribution
    trans_type_col = [col for col in data.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]
    print("\nTransaction type distribution:")
    print(data[trans_type_col].value_counts().head())
    
    # Mode of acquisition distribution
    mode_col = [col for col in data.columns if 'MODE OF ACQUISITION' in col][0]
    print("\nMode of acquisition distribution:")
    print(data[mode_col].value_counts().head())
    
    # Category distribution
    category_col = [col for col in data.columns if 'CATEGORY OF PERSON' in col][0]
    print("\nCategory distribution:")
    print(data[category_col].value_counts().head())

print_dataset_stats("Derivatives Dataset", derivatives_df)
print_dataset_stats("Non-Derivatives Dataset", non_derivatives_df) 