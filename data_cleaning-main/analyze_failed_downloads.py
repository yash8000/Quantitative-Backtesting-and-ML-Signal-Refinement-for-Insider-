import pandas as pd

def get_failed_symbols():
    """Read failed symbols from the failed_downloads.txt file."""
    failed_symbols = []
    with open('failed_downloads.txt', 'r') as f:
        for line in f:
            if ':' in line:  # Lines with symbol info contain ':'
                symbol = line.split(':')[0].strip()
                failed_symbols.append(symbol)
    return failed_symbols

def analyze_and_clean_trades():
    # Get failed symbols
    failed_symbols = get_failed_symbols()
    print(f"Found {len(failed_symbols)} failed symbols")
    
    try:
        # Read non-derivatives dataset
        print("\nReading non-derivatives dataset...")
        non_derivatives = pd.read_csv('nse_non_derivatives.csv')
        total_rows = len(non_derivatives)
        
        # Find trades with failed symbols
        failed_trades = non_derivatives[non_derivatives['symbol'].isin(failed_symbols)]
        print(f"\nFound {len(failed_trades):,} trades with failed symbols ({len(failed_trades)/total_rows*100:.2f}% of total)")
        
        # Group by symbol and show counts
        symbol_counts = failed_trades['symbol'].value_counts()
        print("\nBreakdown of trades to be removed:")
        print("-" * 60)
        for symbol, count in symbol_counts.items():
            print(f"{symbol:<15} {count:>10,d} trades")
        
        # Remove trades with failed symbols
        clean_data = non_derivatives[~non_derivatives['symbol'].isin(failed_symbols)]
        rows_removed = total_rows - len(clean_data)
        print(f"\nRemoving {rows_removed:,} rows...")
        
        # Save cleaned dataset
        output_file = 'nse_non_derivatives_clean.csv'
        clean_data.to_csv(output_file, index=False)
        print(f"\nSaved cleaned dataset to {output_file}")
        print(f"Original rows: {total_rows:,}")
        print(f"Remaining rows: {len(clean_data):,}")
        print(f"Removed rows: {rows_removed:,}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    analyze_and_clean_trades() 