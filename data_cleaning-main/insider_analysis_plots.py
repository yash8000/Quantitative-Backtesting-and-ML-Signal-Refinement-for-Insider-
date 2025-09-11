import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np

# Constants
START_DATE = '2022-07-07'
END_DATE = '2025-07-07'

def remove_outliers(df, column, n_std=3):
    """Remove outliers beyond n standard deviations from the mean."""
    mean = df[column].mean()
    std = df[column].std()
    return df[np.abs(df[column] - mean) <= n_std * std]

def load_and_prepare_data():
    """Load and prepare the insider trading data."""
    # Read the data
    df = pd.read_csv('nse_non_derivatives_clean.csv')
    
    # Convert date columns to datetime
    df['date_intimation'] = pd.to_datetime(df['date_intimation'], format='%d-%b-%Y')
    df['broadcast_datetime'] = pd.to_datetime(df['broadcast_datetime'], format='%d-%b-%Y %H:%M')
    
    # Convert value_traded to numeric
    df['value_traded'] = pd.to_numeric(df['value_traded'], errors='coerce')
    
    # Filter for date range
    df = df[
        (df['date_intimation'] >= START_DATE) & 
        (df['date_intimation'] <= END_DATE)
    ]
    
    # Remove extreme outliers from value_traded
    df = remove_outliers(df, 'value_traded', n_std=3)
    
    return df

def plot_category_transaction_distribution(df):
    """Plot distribution of trades by person category and transaction type."""
    plt.figure(figsize=(15, 8))
    
    # Filter for main transaction types and categories for better visibility
    main_types = ['Buy', 'Sell']
    main_categories = [
        'Promoters', 'Promoter Group', 'Director', 
        'Key Managerial Personnel', 'Immediate relative'
    ]
    
    plot_df = df[
        (df['transaction_type'].isin(main_types)) & 
        (df['person_category'].isin(main_categories))
    ]
    
    # Create the plot
    sns.countplot(data=plot_df, x='person_category', hue='transaction_type')
    plt.xticks(rotation=45, ha='right')
    plt.title("Insider Transactions by Person Category\n(Jul 2022 - Jul 2025)")
    plt.xlabel("Person Category")
    plt.ylabel("Number of Transactions")
    plt.legend(title="Transaction Type")
    plt.tight_layout()
    plt.savefig('plots/category_transaction_distribution.png')
    plt.close()

def plot_promoter_buy_activity(df):
    """Plot time series of promoter buy activity."""
    plt.figure(figsize=(15, 6))
    
    # Filter for promoter buys
    df_filtered = df[
        (df['transaction_type'] == 'Buy') & 
        (df['person_category'].isin(['Promoters', 'Promoter Group']))
    ]
    
    # Group by date and calculate 7-day rolling sum
    buy_counts = df_filtered.groupby('date_intimation').size()
    buy_counts = buy_counts.sort_index()  # Ensure chronological order
    rolling_counts = buy_counts.rolling(7, min_periods=1).sum()
    
    # Plot
    ax = rolling_counts.plot(title="Promoter Buy Activity (7-day rolling sum)\n(Jul 2022 - Jul 2025)")
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    
    plt.ylabel("Number of Buy Transactions")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/promoter_buy_activity.png')
    plt.close()

def plot_buy_sell_activity(df):
    """Plot daily buy vs sell activity weighted by value."""
    plt.figure(figsize=(15, 6))
    
    # Filter for buy/sell transactions
    df_filtered = df[df['transaction_type'].isin(['Buy', 'Sell'])]
    
    # Group by date and transaction type, sum the value traded
    daily_activity = df_filtered.groupby(['date_intimation', 'transaction_type'])['value_traded'].sum().unstack().fillna(0)
    
    # Calculate 7-day moving average
    rolling_activity = daily_activity.rolling(7, min_periods=1).mean()
    
    # Plot
    ax = rolling_activity.plot(title="7-Day Avg of Buy vs Sell Values\n(Jul 2022 - Jul 2025)")
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    
    plt.ylabel("Value Traded (₹)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Transaction Type")
    
    # Format y-axis to show values in crores
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'₹{x/10000000:.1f}Cr' for x in current_values])
    
    plt.tight_layout()
    plt.savefig('plots/buy_sell_activity.png')
    plt.close()

def plot_value_distribution(df):
    """Plot distribution of transaction values by category."""
    plt.figure(figsize=(15, 6))
    
    # Filter for main categories and buy/sell transactions
    main_categories = [
        'Promoters', 'Promoter Group', 'Director', 
        'Key Managerial Personnel', 'Immediate relative'
    ]
    plot_df = df[
        (df['transaction_type'].isin(['Buy', 'Sell'])) & 
        (df['person_category'].isin(main_categories))
    ]
    
    # Create box plot
    sns.boxplot(data=plot_df, x='person_category', y='value_traded', hue='transaction_type', showfliers=False)  # Hide outlier points
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribution of Transaction Values by Category\n(Jul 2022 - Jul 2025)")
    plt.xlabel("Person Category")
    plt.ylabel("Value Traded (₹)")
    
    # Format y-axis to show values in lakhs
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'₹{x/100000:.1f}L' for x in current_values])
    
    plt.tight_layout()
    plt.savefig('plots/value_distribution.png')
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics for the data."""
    print("\nSummary Statistics (after removing outliers):")
    print("==========================================")
    
    # Date range
    print(f"Date Range: {df['date_intimation'].min().strftime('%d %b %Y')} to {df['date_intimation'].max().strftime('%d %b %Y')}")
    
    # Transaction counts
    buy_sell_counts = df[df['transaction_type'].isin(['Buy', 'Sell'])]['transaction_type'].value_counts()
    print("\nTransaction Counts:")
    for tx_type, count in buy_sell_counts.items():
        print(f"{tx_type}: {count:,} transactions")
    
    # Value traded statistics
    value_stats = df[df['transaction_type'].isin(['Buy', 'Sell'])]['value_traded']
    print(f"\nValue Traded Statistics:")
    print(f"Total Value: ₹{value_stats.sum()/10000000:.2f} Cr")
    print(f"Mean Value: ₹{value_stats.mean()/100000:.2f} L")
    print(f"Median Value: ₹{value_stats.median()/100000:.2f} L")
    print(f"Max Value: ₹{value_stats.max()/10000000:.2f} Cr")
    
    # Category-wise statistics
    print("\nCategory-wise Transaction Counts:")
    category_counts = df[
        (df['transaction_type'].isin(['Buy', 'Sell'])) &
        (df['person_category'].isin([
            'Promoters', 'Promoter Group', 'Director',
            'Key Managerial Personnel', 'Immediate relative'
        ]))
    ]['person_category'].value_counts()
    
    for category, count in category_counts.items():
        print(f"{category}: {count:,} transactions")

def main():
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data()
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("1. Plotting category-transaction distribution...")
    plot_category_transaction_distribution(df)
    
    print("2. Plotting promoter buy activity...")
    plot_promoter_buy_activity(df)
    
    print("3. Plotting buy-sell activity...")
    plot_buy_sell_activity(df)
    
    print("4. Plotting value distribution...")
    plot_value_distribution(df)
    
    print("\nPlots have been saved to the 'plots' directory:")
    print("1. category_transaction_distribution.png")
    print("2. promoter_buy_activity.png")
    print("3. buy_sell_activity.png")
    print("4. value_distribution.png")

if __name__ == "__main__":
    main() 