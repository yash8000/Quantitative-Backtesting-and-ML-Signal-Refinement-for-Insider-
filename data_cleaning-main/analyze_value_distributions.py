import pandas as pd
import numpy as np

def analyze_column_distribution(df, column, max_display=10):
    """Analyze the distribution of values in a column."""
    total_rows = len(df)
    
    # Get value counts and percentages
    value_counts = df[column].value_counts()
    value_percentages = (value_counts / total_rows * 100).round(2)
    
    # Get number of unique values
    unique_count = len(value_counts)
    
    print(f"\n=== {column} ===")
    print(f"Total unique values: {unique_count}")
    
    # If number of unique values is less than max_display, show distribution
    if unique_count <= max_display:
        print("\nValue distribution:")
        for value, count in value_counts.items():
            percentage = value_percentages[value]
            print(f"  {value}: {count:,} rows ({percentage}%)")
    else:
        print(f"More than {max_display} unique values - showing top {max_display}:")
        for value, count in value_counts.head(max_display).items():
            percentage = value_percentages[value]
            print(f"  {value}: {count:,} rows ({percentage}%)")

def analyze_dataset(file_path, dataset_name):
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"\nTotal rows: {len(df):,}")
        print(f"Total columns: {len(df.columns):,}")
        
        # Analyze each column
        for column in df.columns:
            analyze_column_distribution(df, column)
            
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {str(e)}")

def main():
    # Analyze both datasets
    analyze_dataset('nse_non_derivatives_clean.csv', 'Non-Derivatives Data')
    #analyze_dataset('nse_derivatives.csv', 'Derivatives Data')

if __name__ == "__main__":
    main() 