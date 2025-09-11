import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import time

def get_unique_symbols():
    """Get unique symbols from both derivatives and non-derivatives datasets."""
    try:
        # Read both datasets
        non_derivatives = pd.read_csv('nse_non_derivatives.csv')
        derivatives = pd.read_csv('nse_derivatives.csv')
        
        # Combine symbols and get unique values
        all_symbols = pd.concat([non_derivatives['symbol'], derivatives['symbol']]).unique()
        return sorted(all_symbols)
    except Exception as e:
        print(f"Error reading datasets: {str(e)}")
        return []

def format_nse_symbol(symbol):
    """Format symbol for NSE (append .NS if not already present)."""
    if not symbol.endswith('.NS'):
        return f"{symbol}.NS"
    return symbol

def download_stock_data(symbol, start_date="2022-07-07", end_date="2025-07-07"):
    """Download stock data for a given symbol with error handling."""
    try:
        # Format symbol for NSE
        nse_symbol = format_nse_symbol(symbol)
        
        # Download data
        data = yf.download(nse_symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for {nse_symbol}")
            return None, "No data available"
            
        # Clean and format the data
        data = data.round(2)  # Round to 2 decimal places
        data.index = data.index.strftime('%Y-%m-%d')  # Format dates
        
        return data, None
    except Exception as e:
        error_msg = str(e)
        print(f"Error downloading {symbol}: {error_msg}")
        return None, error_msg

def save_stock_data(symbol, data):
    """Save stock data to CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs('stock_data', exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join('stock_data', f"{symbol}_stock_data.csv")
        data.to_csv(output_file)
        print(f"Saved data for {symbol} to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving data for {symbol}: {str(e)}")
        return False

def main():
    # Get unique symbols
    symbols = get_unique_symbols()
    print(f"Found {len(symbols)} unique symbols")
    
    # Create directory for stock data
    os.makedirs('stock_data', exist_ok=True)
    
    # Track failures
    failed_downloads = {}
    successful_count = 0
    
    # Download data for each symbol
    for i, symbol in enumerate(symbols, 1):
        print(f"\nProcessing {i}/{len(symbols)}: {symbol}")
        
        # Download data
        data, error = download_stock_data(symbol)
        
        if data is not None:
            # Save data
            if save_stock_data(symbol, data):
                successful_count += 1
            else:
                failed_downloads[symbol] = "Failed to save data"
        else:
            failed_downloads[symbol] = error
        
        # Add delay to avoid hitting rate limits
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"Total symbols processed: {len(symbols)}")
    print(f"Successfully downloaded: {successful_count}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print("\nFailed Downloads Details:")
        print("-"*80)
        for symbol, error in failed_downloads.items():
            print(f"{symbol}: {error}")
    
    # Save failed downloads to file
    if failed_downloads:
        failed_file = "failed_downloads.txt"
        with open(failed_file, 'w') as f:
            f.write("Failed Downloads Report\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total failures: {len(failed_downloads)}\n\n")
            for symbol, error in failed_downloads.items():
                f.write(f"{symbol}: {error}\n")
        print(f"\nDetailed failure report saved to {failed_file}")

if __name__ == "__main__":
    main()
