import pandas as pd
import numpy as np

# Set display options to show all rows and columns without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read the CSV file
df = pd.read_csv('nse_non_derivatives_clean.csv')

print("\n=== 0-A. Basic frame description ===")
print("\nShape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))
print("\nLast 3 rows:")
print(df.tail(3))

print("\n=== 0-B. dtypes, non-null counts ===")
print("\nDataframe Info:")
df.info(show_counts=True)

print("\n=== 0-C. Summary stats for all columns ===")
print(df.describe(include='all', percentiles=[.1, .25, .5, .75, .9]))

print("\n=== 0-D. Nulls & uniqueness ===")
print("\nNull counts (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))
print("\nUnique counts per column:")
print(df.nunique(dropna=False))

print("\n=== 0-E. Duplicates ===")
print("\nExact duplicates (all columns):")
duplicates = df[df.duplicated(keep=False)]
print(f"Total number of exact duplicate rows: {len(duplicates)}")
if not duplicates.empty:
    print("\nExample of exact duplicates (first 5 sets):")
    print(duplicates.head())

print("\nLogical duplicates:")
# Get the exact column names from the DataFrame
symbol_col = [col for col in df.columns if 'SYMBOL' in col][0]
date_col = [col for col in df.columns if 'DATE OF INITMATION TO COMPANY' in col][0]
name_col = [col for col in df.columns if 'NAME OF THE ACQUIRER/DISPOSER' in col][0]
securities_col = [col for col in df.columns if 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)' in col][0]

logical_dupes = df[df.duplicated(subset=[symbol_col, date_col, name_col, securities_col], keep=False)]
print(f"Total number of logical duplicate rows: {len(logical_dupes)}")
if not logical_dupes.empty:
    print("\nExample of logical duplicates (first 5 sets):")
    print(logical_dupes.head())

# Add missing column definitions
transaction_type_col = [col for col in df.columns if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in col][0]
exchange_col = [col for col in df.columns if 'EXCHANGE' in col][0]

print("\n=== Duplicate Transaction Analysis ===")

# First get all duplicates
logical_dupes = df[df.duplicated(subset=[symbol_col, date_col, name_col, securities_col], keep=False)]
total_transactions = len(df)
total_duplicate_rows = len(logical_dupes)
unique_duplicate_groups = logical_dupes.groupby([symbol_col, date_col, name_col, securities_col]).ngroups

print(f"\nOverall Statistics:")
print(f"Total transactions in dataset: {total_transactions:,}")
print(f"Total rows involved in duplicates: {total_duplicate_rows:,}")
print(f"Number of unique duplicate groups: {unique_duplicate_groups:,}")

# Analyze transaction types within duplicates
interesting_cases = []
same_type_cases = []

for _, group in logical_dupes.groupby([symbol_col, date_col, name_col, securities_col]):
    unique_transaction_types = group[transaction_type_col].unique()
    if len(unique_transaction_types) > 1:
        interesting_cases.append(group)
    else:
        same_type_cases.append(group)

print(f"\nBreakdown of Duplicate Cases:")
print(f"Cases with different transaction types: {len(interesting_cases):,}")
print(f"Cases with same transaction type: {len(same_type_cases):,}")

# Show distribution of transaction types in same-type cases
print("\nDistribution of Transaction Types in Same-Type Cases:")
type_counts = pd.Series([group[transaction_type_col].iloc[0] for group in same_type_cases]).value_counts()
print(type_counts)

print("\n=== Sample Cases of Different Transaction Types ===")
if interesting_cases:
    num_examples = min(5, len(interesting_cases))
    print(f"\nShowing {num_examples} sample cases out of {len(interesting_cases)} total cases:")
    
    # Get a mix of different combination types for examples
    shown_combinations = set()
    examples_shown = 0
    
    for group in interesting_cases:
        types = " + ".join(sorted(group[transaction_type_col].unique()))
        if types not in shown_combinations and examples_shown < num_examples:
            shown_combinations.add(types)
            examples_shown += 1
            
            print("\n" + "="*80)
            print(f"Example {examples_shown} (Transaction Types: {types})")
            print(f"Symbol: {group[symbol_col].iloc[0]}")
            print(f"Date: {group[date_col].iloc[0]}")
            print(f"Person: {group[name_col].iloc[0]}")
            print(f"Number of Securities: {group[securities_col].iloc[0]}")
            print("\nTransactions:")
            print(group[[transaction_type_col, exchange_col, 'MODE OF ACQUISITION \n', 'VALUE OF SECURITY (ACQUIRED/DISPLOSED) \n']])
    
    print("\n" + "="*80)
    print(f"... and {len(interesting_cases) - num_examples} more cases")

print("\n=== Sample Cases of Same Transaction Type ===")
if same_type_cases:
    num_examples = min(5, len(same_type_cases))
    print(f"\nShowing {num_examples} sample cases out of {len(same_type_cases)} total cases:")
    
    # Get a mix of different transaction types for examples
    shown_types = set()
    examples_shown = 0
    
    for group in same_type_cases:
        trans_type = group[transaction_type_col].iloc[0]
        if trans_type not in shown_types and examples_shown < num_examples:
            shown_types.add(trans_type)
            examples_shown += 1
            
            print("\n" + "="*80)
            print(f"Example {examples_shown} (Transaction Type: {trans_type})")
            print(f"Symbol: {group[symbol_col].iloc[0]}")
            print(f"Date: {group[date_col].iloc[0]}")
            print(f"Person: {group[name_col].iloc[0]}")
            print(f"Number of Securities: {group[securities_col].iloc[0]}")
            print(f"Number of duplicate entries: {len(group)}")
            print("\nAll Reports:")
            print(group[[exchange_col, 'MODE OF ACQUISITION \n', 'VALUE OF SECURITY (ACQUIRED/DISPLOSED) \n']])
    
    print("\n" + "="*80)
    print(f"... and {len(same_type_cases) - num_examples} more cases")

print("\n=== Additional Analysis ===")
print("\nValue counts for specific columns:")
cols_of_interest = ['REGULATION', 'CATEGORY OF PERSON', 'ACQUISITION/DISPOSAL TRANSACTION TYPE',
                    'MODE OF ACQUISITION', 'DERIVATIVE TYPE SECURITY', 'EXCHANGE']
for col in cols_of_interest:
    matching_cols = [c for c in df.columns if col in c]
    if matching_cols:
        print(f"\n{matching_cols[0]} value counts:")
        print(df[matching_cols[0]].value_counts(dropna=False))

print("\nSample of 10 rows (transposed):")
print(df.sample(10, random_state=42).T)

print("\n=== Analysis After Removing Exact Duplicates (Excluding Broadcast & XBRL) ===")
# Get columns to exclude from exact duplicate check
exclude_cols = [col for col in df.columns if any(term in col.upper() for term in ['BROADCAST', 'XBRL', 'EXCHANGE'])]
cols_for_exact_dupes = [col for col in df.columns if col not in exclude_cols]

# Remove exact duplicates considering only the selected columns
df_no_exact_dupes = df.drop_duplicates(subset=cols_for_exact_dupes)
print(f"\nColumns excluded from exact duplicate check:")
print(exclude_cols)
print(f"\nOriginal number of rows: {len(df):,}")
print(f"Rows after removing exact duplicates: {len(df_no_exact_dupes):,}")
print(f"Number of exact duplicates removed: {len(df) - len(df_no_exact_dupes):,}")

# Now check for logical duplicates with same transaction type
logical_dupes_after = df_no_exact_dupes[df_no_exact_dupes.duplicated(subset=[symbol_col, date_col, name_col, securities_col], keep=False)]

# Analyze transaction types within these logical duplicates
interesting_cases = []
same_type_cases = []

for _, group in logical_dupes_after.groupby([symbol_col, date_col, name_col, securities_col]):
    unique_transaction_types = group[transaction_type_col].unique()
    if len(unique_transaction_types) > 1:
        interesting_cases.append(group)
    else:
        same_type_cases.append(group)

print(f"\nAnalysis of Logical Duplicates After Removing Exact Duplicates:")
print(f"Total rows involved in logical duplicates: {len(logical_dupes_after):,}")
print(f"Number of unique logical duplicate groups: {logical_dupes_after.groupby([symbol_col, date_col, name_col, securities_col]).ngroups:,}")
print(f"Groups with different transaction types: {len(interesting_cases):,}")
print(f"Groups with same transaction type: {len(same_type_cases):,}")

if same_type_cases:
    print("\nDistribution of Transaction Types in Same-Type Cases:")
    type_counts = pd.Series([group[transaction_type_col].iloc[0] for group in same_type_cases]).value_counts()
    print(type_counts)
    
    print("\nAnalyzing what makes same-type cases different:")
    for idx, group in enumerate(same_type_cases[:3]):  # Look at first 3 groups
        print(f"\nExample Group {idx + 1}:")
        print(f"Symbol: {group[symbol_col].iloc[0]}")
        print(f"Date: {group[date_col].iloc[0]}")
        print(f"Transaction Type: {group[transaction_type_col].iloc[0]}")
        print(f"Number of Securities: {group[securities_col].iloc[0]}")
        
        # Find columns that differ within this group
        varying_cols = []
        for col in group.columns:
            if group[col].nunique() > 1:
                varying_cols.append(col)
        
        print("Columns with different values:")
        print(varying_cols)
        if varying_cols:
            print("\nDetailed differences:")
            print(group[varying_cols])
