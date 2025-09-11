import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Create required directories if they don't exist
os.makedirs('plots', exist_ok=True)

# Constants
BASE_POSITION_SIZE = 100  # Base position size of ₹100
MIN_VALUE_TRADED_PCT = 0.001  # 1% of market cap minimum trade value

# Define position sizing tiers
POSITION_SIZING = {
    'Immediate relative': 1.0,  # 100% of base size (₹100)
    'Director': 0.75,           # 100% of base size (₹100)
    'Key Managerial Personnel': 0.75  # 50% of base size (₹50)
}

# Load outstanding shares data
def load_outstanding_shares():
    """Load outstanding shares data from CSV."""
    try:
        shares_df = pd.read_csv('outstanding_shares.csv')
        # Create a dictionary for faster lookups
        shares_dict = shares_df.set_index('symbol')['shares_outstanding'].to_dict()
        return shares_dict
    except Exception as e:
        print(f"Error loading outstanding shares data: {str(e)}")
        return {}

# Define insider categories we're interested in
INSIDER_CATEGORIES = list(POSITION_SIZING.keys())

def get_position_size(category):
    """Get the position size multiplier for a given insider category."""
    return POSITION_SIZING.get(category, 0.0) * BASE_POSITION_SIZE

def calculate_market_cap(shares_outstanding, price):
    """Calculate market cap given shares outstanding and price."""
    if pd.isna(shares_outstanding) or pd.isna(price):
        return None
    return shares_outstanding * price

def load_stock_data(symbol):
    """Load stock data for a given symbol."""
    file_path = f"stock_data/{symbol}_stock_data.csv"
    if os.path.exists(file_path):
        try:
            # Skip the ticker info row and empty date row
            df = pd.read_csv(file_path, skiprows=[1,2])
            
            # The first column should be dates
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]  # Remove the date column since it's now the index
            
            # Sort by date to ensure chronological order
            df.sort_index(inplace=True)
            
            # Verify we have valid data
            if len(df) == 0:
                print(f"Warning: No data found for {symbol}")
                return None
                
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
            return None
    else:
        print(f"No stock data file found for {symbol}")
        return None

def check_stock_listing(broadcast_date, stock_data, symbol):
    """Check if the stock was listed before the broadcast date."""
    earliest_date = stock_data.index.min()
    if broadcast_date < earliest_date:
        days_diff = (earliest_date - broadcast_date).days
        print(f"Warning: For {symbol} - Stock data starts {days_diff} days after broadcast date. "
              f"Broadcast: {broadcast_date.date()}, Earliest data: {earliest_date.date()}")
        return False
    return True

def get_next_trading_day(date, stock_data, symbol):
    """Get the next available trading day from stock data.
    If the exact next day is not available (due to holidays/weekends), 
    find the closest next available trading day. Skip if gap is too large."""
    try:
        # Convert to datetime if string
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Get all dates after the broadcast date
        future_dates = stock_data.index[stock_data.index > date]
        
        if len(future_dates) > 0:
            # Return the first available date after the broadcast
            next_date = future_dates[0]
            days_diff = (next_date - date).days
            if days_diff > 10:  # Warn if gap is more than 10 days
                print(f"Warning: For {symbol} - Large gap ({days_diff} days) between {date.date()} and next trading day {next_date.date()}")
                return None
            return next_date
        else:
            print(f"Warning: For {symbol} - No future trading dates found after {date.date()}")
    except Exception as e:
        print(f"Error finding next trading day for {symbol} after {date}: {str(e)}")
    return None

def get_trading_day_after_n_days(start_date, n_days, stock_data, symbol):
    """Get the trading day that's n trading days after start_date.
    Accounts for holidays and weekends by counting actual trading days."""
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Get all trading days after start_date
        future_dates = stock_data.index[stock_data.index > start_date]
        
        if len(future_dates) >= n_days:
            target_date = future_dates[n_days-1]  # -1 because index is 0-based
            days_diff = (target_date - start_date).days
            if days_diff > n_days * 3:  # Warn if average gap is more than 3x expected
                print(f"Warning: For {symbol} - Large gap ({days_diff} days) for {n_days} trading days from {start_date.date()}")
            return target_date
    except Exception as e:
        print(f"Error finding trading day after {n_days} days from {start_date} for {symbol}: {str(e)}")
    return None

def plot_strategy_results(results_df):
    """Create and save visualizations of strategy performance."""
    # Sort results by date for proper cumulative calculation
    results_df_sorted = results_df.sort_values('entry_date')
    
    # Create main cumulative returns plot
    plt.figure(figsize=(15, 8))
    
    # Calculate weighted cumulative returns
    weighted_returns = (results_df_sorted['pnl_percentage'] * results_df_sorted['position_size']).cumsum() / results_df_sorted['position_size'].cumsum()
    
    plt.plot(results_df_sorted['entry_date'], weighted_returns, 
             color='blue', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Weighted Cumulative Return (%)')
    plt.title('Weighted Cumulative Percentage Returns Over Time')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the main plot
    plt.savefig('plots/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 2x2 subplot figure for additional analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Strategy Analysis', fontsize=16, y=0.95)
    
    # 1. Monthly Returns Distribution
    monthly_returns = results_df_sorted.groupby(
        results_df_sorted['entry_date'].dt.strftime('%Y-%m')
    )['pnl_percentage'].sum()
    ax1.hist(monthly_returns, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Monthly Returns Distribution')
    ax1.set_xlabel('Monthly Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling 20-Trade Win Rate
    rolling_wins = (results_df_sorted['pnl_percentage'] > 0).rolling(20).mean() * 100
    ax2.plot(range(len(rolling_wins)), rolling_wins, color='green')
    ax2.set_title('Rolling 20-Trade Win Rate')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Win Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average Returns by Holding Period
    holding_returns = results_df_sorted.groupby('holding_days')['pnl_percentage'].mean()
    ax3.bar(holding_returns.index, holding_returns, alpha=0.7, color='blue')
    ax3.set_title('Average Returns by Holding Period')
    ax3.set_xlabel('Holding Days')
    ax3.set_ylabel('Average Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Returns by Exit Type
    exit_returns = results_df_sorted.groupby('exit_reason')['pnl_percentage'].agg(['mean', 'count'])
    colors = {'profit_target': 'green', 'stop_loss': 'red', 'time_exit': 'blue'}
    for i, (exit_type, stats) in enumerate(exit_returns.iterrows()):
        ax4.bar(i, stats['mean'], color=colors.get(exit_type, 'gray'), 
                alpha=0.7, label=f"{exit_type} (n={int(stats['count'])})")
    ax4.set_title('Average Returns by Exit Type')
    ax4.set_xticks(range(len(exit_returns)))
    ax4.set_xticklabels(exit_returns.index, rotation=45)
    ax4.set_ylabel('Average Return (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('plots/strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create category performance plot
    plt.figure(figsize=(12, 6))
    
    # Calculate category-wise performance metrics
    category_stats = results_df_sorted.groupby('person_category').agg({
        'pnl_percentage': ['mean', 'count', lambda x: (x > 0).mean() * 100]
    })
    category_stats.columns = ['avg_return', 'trade_count', 'win_rate']
    
    # Plot average returns by category
    bars = plt.bar(range(len(category_stats)), category_stats['avg_return'], alpha=0.7, color='blue')
    
    # Add trade count and win rate annotations
    for i, (category, stats) in enumerate(category_stats.iterrows()):
        plt.text(i, stats['avg_return'], 
                f"n={int(stats['trade_count'])}\nWR={stats['win_rate']:.1f}%",
                ha='center', va='bottom')
    
    plt.title('Performance by Insider Category')
    plt.xlabel('Insider Category')
    plt.ylabel('Average Return (%)')
    plt.xticks(range(len(category_stats)), category_stats.index, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save category performance plot
    plt.savefig('plots/category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nStrategy performance plots saved to plots/")
    print("- plots/cumulative_returns.png: Main cumulative returns plot")
    print("- plots/strategy_analysis.png: Additional analysis plots")
    print("- plots/category_performance.png: Category-wise performance analysis")
    
    print("\nCategory-wise Performance:")
    print("=========================")
    print(category_stats.round(2))

def implement_strategy():
    # Load outstanding shares data
    shares_outstanding_dict = load_outstanding_shares()
    print(f"Loaded outstanding shares data for {len(shares_outstanding_dict)} symbols")
    
    # Create required directories
    os.makedirs('plots', exist_ok=True)
    
    # Read insider trading data
    print("Loading insider trading data...")
    trades_df = pd.read_csv('nse_non_derivatives_clean.csv')
    
    # Convert date columns to datetime
    trades_df['broadcast_datetime'] = pd.to_datetime(trades_df['broadcast_datetime'], format='%d-%b-%Y %H:%M')
    
    # Convert value_traded to numeric
    trades_df['value_traded'] = pd.to_numeric(trades_df['value_traded'], errors='coerce')
    
    # Filter for insider buys with market purchases
    insider_buys = trades_df[
        (trades_df['person_category'].isin(INSIDER_CATEGORIES)) &
        (trades_df['transaction_type'] == 'Buy') &
        (trades_df['transaction_mode'] == 'Market Purchase')
    ].copy()
    
    print("\nTrade Distribution by Category:")
    print("==============================")
    category_counts = insider_buys['person_category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(insider_buys)) * 100
        print(f"{category:25} : {count:4} trades ({percentage:5.1f}%)")
    
    print(f"\nTotal initial trades: {len(insider_buys)}")
    
    # Get the absolute last date in our dataset for backtest end
    last_backtest_date = None
    for symbol in insider_buys['symbol'].unique():
        stock_data = load_stock_data(symbol)
        if stock_data is not None:
            if last_backtest_date is None or stock_data.index[-1] > last_backtest_date:
                last_backtest_date = stock_data.index[-1]
    
    if last_backtest_date is None:
        print("Error: Could not determine last backtest date!")
        return
        
    print(f"\nBacktest period ends on: {last_backtest_date.date()}")
    
    # Initialize results tracking
    trades_log = []
    skipped_trades = []
    
    # Process each insider buy
    for idx, trade in insider_buys.iterrows():
        symbol = trade['symbol']
        broadcast_date = trade['broadcast_datetime']
        person_category = trade['person_category']
        value_traded = trade['value_traded']
        
        # Skip if no outstanding shares data
        shares_outstanding = shares_outstanding_dict.get(symbol)
        if shares_outstanding is None:
            skipped_trades.append({
                'symbol': symbol,
                'date': broadcast_date,
                'reason': 'no_shares_outstanding_data',
                'category': person_category
            })
            continue
        
        # Get position size for this category
        position_size = get_position_size(person_category)
        
        # Load stock data
        stock_data = load_stock_data(symbol)
        if stock_data is None:
            skipped_trades.append({
                'symbol': symbol, 
                'date': broadcast_date, 
                'reason': 'no_stock_data',
                'category': person_category
            })
            continue
            
        # Get entry point (next trading day after broadcast)
        entry_date = get_next_trading_day(broadcast_date, stock_data, symbol)
        if entry_date is None:
            skipped_trades.append({
                'symbol': symbol, 
                'date': broadcast_date, 
                'reason': 'large_gap_or_no_next_day', 
                'category': person_category,
                'details': 'Gap > 10 days or no next trading day found'
            })
            continue
            
        # Get entry price and calculate market cap
        entry_price = stock_data.loc[entry_date, 'Open']
        market_cap = calculate_market_cap(shares_outstanding, entry_price)
        
        if market_cap is None:
            skipped_trades.append({
                'symbol': symbol,
                'date': broadcast_date,
                'reason': 'cannot_calculate_market_cap',
                'category': person_category
            })
            continue
        
        # Check if trade value is significant enough (> 1% of market cap)
        min_value_traded = market_cap * MIN_VALUE_TRADED_PCT
        if value_traded < min_value_traded:
            skipped_trades.append({
                'symbol': symbol,
                'date': broadcast_date,
                'reason': 'insufficient_value_traded',
                'category': person_category,
                'details': f'Value traded: ₹{value_traded:,.0f}, Required: ₹{min_value_traded:,.0f}'
            })
            continue
            
        shares = position_size / entry_price  # Now uses category-specific position size
        
        # Get the last available date for this stock
        last_available_date = stock_data.index[-1]
        
        # Calculate days we can look ahead
        available_future_days = len(stock_data.loc[entry_date:last_available_date])
        
        # Skip if we have very little future data
        if available_future_days < 5:  # Require at least 5 days of data
            skipped_trades.append({
                'symbol': symbol, 
                'date': broadcast_date, 
                'reason': 'insufficient_future_data', 
                'category': person_category
            })
            continue
        
        # Initialize variables
        exit_triggered = False
        first_target_hit = False
        second_target_hit = False
        remaining_shares = shares
        total_pnl = 0
        exit_price = None
        exit_date = None
        exit_reason = None
        first_exit_price = None
        first_exit_date = None
        second_exit_price = None
        second_exit_date = None
        stop_loss_level = entry_price * 0.80  # Initial stop loss at -15%
        
        # Get max holding period date (120 days or last available date, whichever is earlier)
        max_exit_date = get_trading_day_after_n_days(entry_date, 200, stock_data, symbol)
        if max_exit_date is None or max_exit_date > last_available_date:
            max_exit_date = last_available_date
        
        holding_period_data = stock_data.loc[entry_date:max_exit_date]
        
        # Check each day's price action
        for date, row in holding_period_data.iterrows():
            if date == entry_date:
                continue
            
            # First target check (30%) if not yet hit
            if not first_target_hit and row['High'] >= entry_price * 1.30:
                first_target_hit = True
                first_exit_date = date
                first_exit_price = entry_price * 1.30
                # Sell one-third position
                sold_shares = remaining_shares / 3
                remaining_shares = remaining_shares * 2/3
                # Calculate PnL for first third
                first_exit_pnl = (first_exit_price - entry_price) * sold_shares
                total_pnl += first_exit_pnl
                # Raise stop loss to -15% from current price
                stop_loss_level = first_exit_price * 0.85
                continue
            
            # After first target hit, check for second target (60%) or trailing stop
            if first_target_hit and not second_target_hit and row['High'] >= entry_price * 1.60:
                second_target_hit = True
                second_exit_date = date
                second_exit_price = entry_price * 1.60
                # Sell another one-third of original position
                sold_shares = shares / 3
                remaining_shares = shares / 3  # Only 1/3 remains
                # Calculate PnL for second third
                second_exit_pnl = (second_exit_price - entry_price) * sold_shares
                total_pnl += second_exit_pnl
                # Raise stop loss to -15% from current price
                stop_loss_level = second_exit_price * 0.85
                continue
            
            # After second target hit, check for third target (100%) or trailing stop
            if second_target_hit:
                # Check if third target hit (100%)
                if row['High'] >= entry_price * 2.00:
                    exit_price = entry_price * 2.00
                    exit_date = date
                    exit_reason = 'third_target'
                    # Calculate PnL for remaining position
                    total_pnl += (exit_price - entry_price) * remaining_shares
                    exit_triggered = True
                    break
                # Check if trailing stop loss hit
                elif row['Low'] <= stop_loss_level:
                    exit_price = stop_loss_level
                    exit_date = date
                    exit_reason = 'trailing_stop'
                    # Calculate PnL for remaining position
                    total_pnl += (exit_price - entry_price) * remaining_shares
                    exit_triggered = True
                    break
            # After first target hit but before second target
            elif first_target_hit:
                # Check if trailing stop loss hit
                if row['Low'] <= stop_loss_level:
                    exit_price = stop_loss_level
                    exit_date = date
                    exit_reason = 'trailing_stop'
                    # Calculate PnL for remaining position
                    total_pnl += (exit_price - entry_price) * remaining_shares
                    exit_triggered = True
                    break
            else:
                # Check initial stop loss if first target not hit
                if row['Low'] <= entry_price * 0.80:
                    exit_price = entry_price * 0.80
                    exit_date = date
                    exit_reason = 'stop_loss'
                    total_pnl += (exit_price - entry_price) * shares
                    exit_triggered = True
                    break
        
        # If no exit was triggered
        if not exit_triggered:
            exit_date = max_exit_date
            exit_price = holding_period_data.loc[max_exit_date, 'Close']
            
            # Calculate final PnL based on whether first target was hit
            if first_target_hit:
                # Add PnL for remaining half position
                total_pnl += (exit_price - entry_price) * remaining_shares
            else:
                # Add PnL for full position
                total_pnl += (exit_price - entry_price) * shares
            
            # Determine exit reason based on whether we hit max holding period or end of backtest
            if len(holding_period_data) >= 200:
                exit_reason = 'time_exit'
            else:
                exit_reason = 'backtest_end'
        
        # Calculate final P&L percentage based on initial position size
        pnl_percentage = (total_pnl / (position_size)) * 100
        actual_profit = total_pnl
        holding_days = len(stock_data.loc[entry_date:exit_date])
        
        trade_info = {
            'symbol': symbol,
            'broadcast_date': broadcast_date,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'market_cap': market_cap,
            'value_traded': value_traded,
            'value_traded_pct': (value_traded / market_cap) * 100,
            'first_target_hit': first_target_hit,
            'first_exit_date': first_exit_date if first_target_hit else None,
            'first_exit_price': first_exit_price if first_target_hit else None,
            'second_target_hit': second_target_hit,
            'second_exit_date': second_exit_date if second_target_hit else None,
            'second_exit_price': second_exit_price if second_target_hit else None,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'initial_shares': shares,
            'remaining_shares': remaining_shares if second_target_hit else shares,
            'position_size': position_size,
            'holding_days': holding_days,
            'pnl_percentage': pnl_percentage,
            'actual_profit': actual_profit,
            'exit_reason': exit_reason,
            'person_category': person_category
        }
        trades_log.append(trade_info)
    
    if not trades_log:
        print("No trades were executed!")
        return
    
    # Convert trades log to DataFrame
    results_df = pd.DataFrame(trades_log)
    
    # Calculate weighted statistics
    total_trades = len(results_df)
    total_investment = results_df['position_size'].sum()
    total_profit = results_df['actual_profit'].sum()
    
    # Calculate weighted win rate
    weighted_wins = (results_df[results_df['pnl_percentage'] > 0]['position_size'].sum() / total_investment) * 100
    
    # Calculate category-wise statistics
    category_stats = results_df.groupby('person_category').agg({
        'pnl_percentage': ['mean', 'count'],
        'position_size': 'sum',
        'actual_profit': 'sum'
    }).round(2)
    
    category_stats.columns = ['Avg Return %', 'Trade Count', 'Total Investment', 'Total P&L']
    category_stats['ROI %'] = (category_stats['Total P&L'] / category_stats['Total Investment'] * 100).round(2)
    
    print("\nOverall Strategy Results:")
    print("=======================")
    print(f"Total Trades: {total_trades}")
    print(f"Total Investment: ₹{total_investment:,.2f}")
    print(f"Portfolio End Value: ₹{total_investment + total_profit:,.2f}")
    print(f"Total P&L: ₹{total_profit:,.2f}")
    print(f"Total Portfolio Return: {(total_profit / total_investment * 100):.2f}%")
    print(f"Win Rate (Position-Weighted): {weighted_wins:.2f}%")
    print(f"Average Trade Return: {results_df['pnl_percentage'].mean():.2f}%")
    print(f"Median Trade Return: {results_df['pnl_percentage'].median():.2f}%")
    
    
    print("\nRisk Metrics:")
    print("============")
    print(f"Max Drawdown: {results_df['pnl_percentage'].min():.2f}%")
    print(f"Volatility: {results_df['pnl_percentage'].std():.2f}%")
    print(f"Average Holding Period: {results_df['holding_days'].mean():.1f} days")
    
    print("\nPosition Sizing Summary:")
    print("=====================")
    for category in INSIDER_CATEGORIES:
        size = get_position_size(category)
        print(f"{category:25}: ₹{size:,.2f}")
    
    print("\nCategory-wise Performance:")
    print("=====================")
    print(category_stats)
    
    print("\nExit Types (Weighted by Position Size):")
    print("=================================")
    exit_types = {
        'second_target': 'Second Target (30%)',
        'trailing_stop': 'Trailing Stop',
        'stop_loss': 'Initial Stop Loss',
        'time_exit': 'Time Exit',
        'backtest_end': 'Backtest End'
    }
    
    for exit_type, description in exit_types.items():
        exits_df = results_df[results_df['exit_reason'] == exit_type]
        if len(exits_df) > 0:
            weighted_pct = (exits_df['position_size'].sum() / total_investment) * 100
            avg_return = (exits_df['actual_profit'].sum() / exits_df['position_size'].sum()) * 100
            print(f"{description:20}: {len(exits_df):4} trades ({weighted_pct:5.1f}% of capital) | Avg: {avg_return:6.2f}%")
    
    # Print first target statistics
    first_target_hits = results_df[results_df['first_target_hit'] == True]
    if len(first_target_hits) > 0:
        target_hit_pct = (len(first_target_hits) / len(results_df)) * 100
        print(f"\nFirst Target (20%) Statistics:")
        print(f"Hit Rate: {target_hit_pct:.1f}% ({len(first_target_hits)} trades)")
        avg_days_to_target = (first_target_hits['first_exit_date'] - first_target_hits['entry_date']).dt.days.mean()
        print(f"Average Days to First Target: {avg_days_to_target:.1f} days")
    
    # Calculate monthly statistics
    results_df['month'] = results_df['entry_date'].dt.strftime('%Y-%m')
    monthly_stats = results_df.groupby('month').agg({
        'actual_profit': ['sum', 'mean', 'count'],
        'pnl_percentage': ['mean']
    }).round(2)
    
    print("\nMonthly Performance Summary:")
    print("=========================")
    print(monthly_stats)
    
    # Save detailed trade log
    results_df.to_csv('promoter_strategy_trades.csv', index=False)
    print("\nDetailed trade log saved to promoter_strategy_trades.csv")
    
    # Create and save performance plots
    plot_strategy_results(results_df)

    # After processing all trades, print summary of skipped trades
    if skipped_trades:
        print("\nSkipped Trades Summary:")
        skipped_df = pd.DataFrame(skipped_trades)
        for reason in skipped_df['reason'].unique():
            count = len(skipped_df[skipped_df['reason'] == reason])
            print(f"\n{reason}: {count} trades")
            if count > 0:
                print("Examples:")
                examples = skipped_df[skipped_df['reason'] == reason].head(3)
                for _, example in examples.iterrows():
                    if reason == 'stock_not_listed':
                        print(f"  {example['symbol']} - Broadcast: {example['date']}, Listed: {example['listing_date']}")
                    else:
                        print(f"  {example['symbol']} on {example['date']}")

if __name__ == "__main__":
    implement_strategy() 