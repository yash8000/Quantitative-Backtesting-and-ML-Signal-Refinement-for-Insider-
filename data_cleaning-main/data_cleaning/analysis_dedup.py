import pandas as pd
import numpy as np
from datetime import datetime

# Set display options to show all rows and columns without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read the CSV file
df = pd.read_csv('nse_non_derivatives_cleaned_final.csv')

def print_section_header(section):
    print(f"\n{'='*50}")
    print(f"=== {section} ===")
    print(f"{'='*50}\n")

# === 0-A. Basic frame description ===
print_section_header("0-A. Basic frame description")
print("Shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))
print("\nLast 3 rows:")
print(df.tail(3))

# === 0-B. dtypes, non-null counts ===
print_section_header("0-B. dtypes, non-null counts")
print("Dataframe Info:")
df.info(show_counts=True)

# === 0-C. Summary stats for all columns ===
print_section_header("0-C. Summary stats for all columns")
print(df.describe(include='all', percentiles=[.1, .25, .5, .75, .9]))

# === 0-D. Nulls & uniqueness ===
print_section_header("0-D. Nulls & uniqueness")
print("Null counts (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))
print("\nUnique counts per column:")
print(df.nunique(dropna=False))

# === 0-E. Duplicates Analysis ===
print_section_header("0-E. Duplicates Analysis")

# First check exact duplicates (all columns)
exact_duplicates = df[df.duplicated(keep=False)]
print("1. Exact Duplicates (considering all columns):")
print(f"Total number of exact duplicate rows: {len(exact_duplicates):,}")

# Now check exact duplicates excluding broadcast, XBRL, and exchange
exclude_cols = [col for col in df.columns if any(term in col.upper() for term in ['BROADCAST', 'XBRL', 'EXCHANGE'])]
cols_for_exact_dupes = [col for col in df.columns if col not in exclude_cols]

print("\n2. Exact Duplicates (excluding broadcast, XBRL, and exchange):")
print("Columns excluded from duplicate check:")
print(exclude_cols)

# Get broadcast date column name
broadcast_col = [col for col in df.columns if 'BROADCAST' in col.upper()][0]

# Convert broadcast date string to datetime for proper sorting
df['broadcast_datetime'] = pd.to_datetime(df[broadcast_col], format='%d-%b-%Y %H:%M')

# Sort by broadcast datetime and then remove duplicates keeping first (earliest) occurrence
df_sorted = df.sort_values('broadcast_datetime')
df_no_exact_dupes = df_sorted.drop_duplicates(subset=cols_for_exact_dupes, keep='first')

# Drop the temporary datetime column
df_no_exact_dupes = df_no_exact_dupes.drop('broadcast_datetime', axis=1)

print(f"\nOriginal number of rows: {len(df):,}")
print(f"Rows after removing exact duplicates: {len(df_no_exact_dupes):,}")
print(f"Number of exact duplicates removed: {len(df) - len(df_no_exact_dupes):,}")

# Calculate percentage of duplicates
dup_percentage = ((len(df) - len(df_no_exact_dupes)) / len(df)) * 100
print(f"Percentage of duplicate rows: {dup_percentage:.2f}%")

# Print example of how sorting worked for a duplicate group
print("\nExample of duplicate handling (before deduplication):")
print("For one duplicate group, showing broadcast times sorted:")
example_dup = df_sorted[df_sorted.duplicated(subset=cols_for_exact_dupes, keep=False)].head(3)
if not example_dup.empty:
    print(example_dup[[broadcast_col] + cols_for_exact_dupes[:2]])  # Show broadcast time and first 2 identifying columns
    print("\nIn the deduplicated dataset, the earliest broadcast time was kept.")

# Save deduplicated dataset
print("\nSaving deduplicated dataset...")
df_no_exact_dupes.to_csv('nse_cleaned_final.csv', index=False)
print("Deduplicated dataset saved as 'nse_deduplicated.csv'") 