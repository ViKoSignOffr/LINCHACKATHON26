import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load daily stock price data from a CSV file in wide format.
    Columns represent tickers, and rows represent dates.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'date'})
    df = df.set_index('date')
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the stock price data."""
    # Handle missing values
    df = df.dropna(how='any')  # Drop rows with any missing values
    
    # Ensure sufficient history for analysis
    df = df[df.index >= df.index.min() + pd.DateOffset(days=252)]  # At least 252 days of data
    
    return df

def get_tickers(df: pd.DataFrame) -> list:
    """Get unique tickers from the DataFrame."""
    return df.columns.tolist()