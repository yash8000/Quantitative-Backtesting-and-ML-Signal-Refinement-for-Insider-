import yfinance as yf
import pandas as pd
import os
from pathlib import Path
import time

def get_stock_symbols():
    """Get list of stock symbols from stock_data directory."""
    stock_data_dir = Path('stock_data')
    symbols = []
    
    for file in stock_data_dir.glob('*_stock_data.csv'):
        # Remove '_stock_data.csv' from filename to get symbol
        symbol = file.name.replace('_stock_data.csv', '')
        symbols.append(symbol)
    
    return symbols

def get_outstanding_shares(symbol):
    """Get outstanding shares for a symbol using yfinance."""
    try:
        # Add .NS suffix for NSE stocks
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        shares = info.get('sharesOutstanding', None)
        
        # Additional info that might be useful
        market_cap = info.get('marketCap', None)
        float_shares = info.get('floatShares', None)
        
        return {
            'symbol': symbol,
            'shares_outstanding': shares,
            'market_cap': market_cap,
            'float_shares': float_shares
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'shares_outstanding': None,
            'market_cap': None,
            'float_shares': None
        }

def main():
    # Get list of symbols
    symbols = get_stock_symbols()
    print(f"Found {len(symbols)} symbols")
    
    # Create list to store results
    results = []
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        print(f"Processing {symbol} ({i}/{len(symbols)})")
        result = get_outstanding_shares(symbol)
        results.append(result)
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('outstanding_shares.csv', index=False)
    print("\nData saved to outstanding_shares.csv")
    
    # Print summary
    print("\nSummary:")
    print(f"Total symbols processed: {len(df)}")
    print(f"Symbols with data: {df['shares_outstanding'].notna().sum()}")
    print(f"Symbols missing data: {df['shares_outstanding'].isna().sum()}")

if __name__ == "__main__":
    main() 