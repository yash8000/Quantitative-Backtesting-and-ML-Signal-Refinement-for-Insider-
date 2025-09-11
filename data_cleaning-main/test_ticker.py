import yfinance as yf

def ticker_exists(symbol):
    try:
        info = yf.Ticker(symbol).info
        print(info)
    except Exception as e:
        print(f"Error for {symbol}: {e}")
        return False

print(ticker_exists("5PAISA.NS"))   # True
