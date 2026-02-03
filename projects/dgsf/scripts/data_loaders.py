"""
Data Loading Module for DGSF Feature Engineering Pipeline

This module implements the 5 data loaders defined in Step 1 of the execution plan.
Each loader reads raw data, applies date filtering, and returns a validated DataFrame.

Module: data_loaders.py
Version: 0.1.0
Date: 2026-02-03
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass


def _validate_date_range(start_date: str, end_date: str) -> tuple:
    """
    Validate and parse date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        ValueError: If dates are invalid or start > end
    """
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start_dt >= end_dt:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    
    return start_dt, end_dt


def _align_to_month_end(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Align dates to month-end for consistency with PanelTree rebalancing.
    
    Args:
        df: Input DataFrame with date column
        date_column: Name of the date column
        
    Returns:
        DataFrame with dates aligned to month-end
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] = df[date_column] + pd.offsets.MonthEnd(0)
    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: list, source_name: str) -> None:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        source_name: Name of data source (for error messages)
        
    Raises:
        DataValidationError: If required columns are missing
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"{source_name}: Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )


def load_price_data(start_date: str, end_date: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load price data (adjusted closing prices).
    
    Data Source: price_data from config
    Returns: DataFrame with columns [date, firm_id, price]
    Frequency: Daily/Monthly (aligned to month-end)
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame with columns: [date, firm_id, price]
        - date: Month-end aligned datetime
        - firm_id: String identifier
        - price: Float (adjusted closing price)
        
    Raises:
        FileNotFoundError: If data file not found
        DataValidationError: If data validation fails
    """
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Get data source configuration
    price_config = config['data_sources']['price_data']
    data_path = Path(price_config['path'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Price data file not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path)
    
    # Map columns
    col_mapping = price_config['columns']
    df = df.rename(columns={
        col_mapping['date']: 'date',
        col_mapping['firm_id']: 'firm_id',
        col_mapping['price']: 'price'
    })
    
    # Validate required columns
    _validate_required_columns(df, ['date', 'firm_id', 'price'], 'price_data')
    
    # Convert date and filter
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    
    # Align to month-end
    df = _align_to_month_end(df, 'date')
    
    # Validate data quality
    if df['price'].isna().any():
        na_count = df['price'].isna().sum()
        total = len(df)
        print(f"Warning: price_data contains {na_count}/{total} missing values ({na_count/total*100:.1f}%)")
    
    # Remove negative/zero prices
    invalid_prices = (df['price'] <= 0).sum()
    if invalid_prices > 0:
        print(f"Warning: Removing {invalid_prices} rows with invalid prices (<=0)")
        df = df[df['price'] > 0]
    
    return df[['date', 'firm_id', 'price']].reset_index(drop=True)


def load_shares_outstanding(start_date: str, end_date: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load shares outstanding data.
    
    Data Source: shares_outstanding from config
    Returns: DataFrame with columns [date, firm_id, shares]
    Frequency: Quarterly/Monthly
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame with columns: [date, firm_id, shares]
        - date: Month-end aligned datetime
        - firm_id: String identifier
        - shares: Float (shares outstanding in millions)
        
    Raises:
        FileNotFoundError: If data file not found
        DataValidationError: If data validation fails
    """
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Get data source configuration
    shares_config = config['data_sources']['shares_outstanding']
    data_path = Path(shares_config['path'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Shares outstanding file not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path)
    
    # Map columns
    col_mapping = shares_config['columns']
    df = df.rename(columns={
        col_mapping['date']: 'date',
        col_mapping['firm_id']: 'firm_id',
        col_mapping['shares']: 'shares'
    })
    
    # Validate required columns
    _validate_required_columns(df, ['date', 'firm_id', 'shares'], 'shares_outstanding')
    
    # Convert date and filter
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    
    # Align to month-end
    df = _align_to_month_end(df, 'date')
    
    # Validate data quality
    if df['shares'].isna().any():
        na_count = df['shares'].isna().sum()
        total = len(df)
        print(f"Warning: shares_outstanding contains {na_count}/{total} missing values ({na_count/total*100:.1f}%)")
    
    # Remove negative/zero shares
    invalid_shares = (df['shares'] <= 0).sum()
    if invalid_shares > 0:
        print(f"Warning: Removing {invalid_shares} rows with invalid shares (<=0)")
        df = df[df['shares'] > 0]
    
    return df[['date', 'firm_id', 'shares']].reset_index(drop=True)


def load_financial_statements(start_date: str, end_date: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load financial statements (balance sheet + income statement).
    
    Data Source: financial_statements from config
    Returns: DataFrame with columns [date, firm_id, total_assets, total_liabilities, 
                                      stockholders_equity, operating_income, net_income]
    Frequency: Quarterly
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame with columns: [date, firm_id, total_assets, total_liabilities,
                                     stockholders_equity, operating_income, net_income]
        - date: Month-end aligned datetime (report date)
        - firm_id: String identifier
        - total_assets: Float (in millions)
        - total_liabilities: Float (in millions)
        - stockholders_equity: Float (book equity in millions)
        - operating_income: Float (in millions)
        - net_income: Float (in millions)
        
    Raises:
        FileNotFoundError: If data file not found
        DataValidationError: If data validation fails
    """
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Get data source configuration
    fin_config = config['data_sources']['financial_statements']
    data_path = Path(fin_config['path'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Financial statements file not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path)
    
    # Map columns
    col_mapping = fin_config['columns']
    df = df.rename(columns={
        col_mapping['date']: 'date',
        col_mapping['firm_id']: 'firm_id',
        col_mapping['total_assets']: 'total_assets',
        col_mapping['total_liabilities']: 'total_liabilities',
        col_mapping['stockholders_equity']: 'stockholders_equity',
        col_mapping['operating_income']: 'operating_income',
        col_mapping['net_income']: 'net_income'
    })
    
    # Validate required columns
    required_cols = ['date', 'firm_id', 'total_assets', 'total_liabilities', 
                     'stockholders_equity', 'operating_income', 'net_income']
    _validate_required_columns(df, required_cols, 'financial_statements')
    
    # Convert date and filter (extend range by 90 days for reporting lag)
    df['date'] = pd.to_datetime(df['date'])
    extended_start = start_dt - pd.DateOffset(days=90)
    df = df[(df['date'] >= extended_start) & (df['date'] <= end_dt)]
    
    # Align to month-end
    df = _align_to_month_end(df, 'date')
    
    # Validate data quality
    for col in ['total_assets', 'stockholders_equity']:
        if df[col].isna().any():
            na_count = df[col].isna().sum()
            total = len(df)
            print(f"Warning: financial_statements.{col} contains {na_count}/{total} missing values ({na_count/total*100:.1f}%)")
    
    return df[required_cols].reset_index(drop=True)


def load_monthly_returns(start_date: str, end_date: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load monthly returns data.
    
    Data Source: monthly_returns from config
    Returns: DataFrame with columns [date, firm_id, return]
    Frequency: Monthly
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame with columns: [date, firm_id, return]
        - date: Month-end aligned datetime
        - firm_id: String identifier
        - return: Float (monthly return, e.g., 0.05 = 5%)
        
    Raises:
        FileNotFoundError: If data file not found
        DataValidationError: If data validation fails
    """
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Get data source configuration
    returns_config = config['data_sources']['monthly_returns']
    data_path = Path(returns_config['path'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Monthly returns file not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path)
    
    # Map columns
    col_mapping = returns_config['columns']
    df = df.rename(columns={
        col_mapping['date']: 'date',
        col_mapping['firm_id']: 'firm_id',
        col_mapping['return']: 'return'
    })
    
    # Validate required columns
    _validate_required_columns(df, ['date', 'firm_id', 'return'], 'monthly_returns')
    
    # Convert date and filter (extend range by 12 months for momentum computation)
    df['date'] = pd.to_datetime(df['date'])
    extended_start = start_dt - pd.DateOffset(months=12)
    df = df[(df['date'] >= extended_start) & (df['date'] <= end_dt)]
    
    # Align to month-end
    df = _align_to_month_end(df, 'date')
    
    # Validate data quality
    if df['return'].isna().any():
        na_count = df['return'].isna().sum()
        total = len(df)
        print(f"Warning: monthly_returns contains {na_count}/{total} missing values ({na_count/total*100:.1f}%)")
    
    # Flag extreme returns (potential data errors)
    extreme_returns = ((df['return'] < -0.9) | (df['return'] > 10)).sum()
    if extreme_returns > 0:
        print(f"Warning: Found {extreme_returns} extreme returns (< -90% or > 1000%)")
    
    return df[['date', 'firm_id', 'return']].reset_index(drop=True)


def load_risk_free_rate(start_date: str, end_date: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load risk-free rate data (e.g., Treasury bill rates).
    
    Data Source: risk_free_rate from config
    Returns: DataFrame with columns [date, risk_free_rate]
    Frequency: Monthly
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame with columns: [date, risk_free_rate]
        - date: Month-end aligned datetime
        - risk_free_rate: Float (monthly rate, e.g., 0.002 = 0.2%)
        
    Raises:
        FileNotFoundError: If data file not found
        DataValidationError: If data validation fails
    """
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Get data source configuration
    rf_config = config['data_sources']['risk_free_rate']
    data_path = Path(rf_config['path'])
    
    if not data_path.exists():
        raise FileNotFoundError(f"Risk-free rate file not found: {data_path}")
    
    # Read CSV
    df = pd.read_csv(data_path)
    
    # Map columns
    col_mapping = rf_config['columns']
    df = df.rename(columns={
        col_mapping['date']: 'date',
        col_mapping['rate']: 'risk_free_rate'
    })
    
    # Validate required columns
    _validate_required_columns(df, ['date', 'risk_free_rate'], 'risk_free_rate')
    
    # Convert date and filter (extend range by 12 months for lagged computations)
    df['date'] = pd.to_datetime(df['date'])
    extended_start = start_dt - pd.DateOffset(months=12)
    df = df[(df['date'] >= extended_start) & (df['date'] <= end_dt)]
    
    # Align to month-end
    df = _align_to_month_end(df, 'date')
    
    # Validate data quality
    if df['risk_free_rate'].isna().any():
        na_count = df['risk_free_rate'].isna().sum()
        total = len(df)
        print(f"Warning: risk_free_rate contains {na_count}/{total} missing values ({na_count/total*100:.1f}%)")
    
    # Flag negative rates (unusual but possible in some periods)
    negative_rates = (df['risk_free_rate'] < -0.01).sum()
    if negative_rates > 0:
        print(f"Warning: Found {negative_rates} negative risk-free rates (< -1%)")
    
    return df[['date', 'risk_free_rate']].reset_index(drop=True)


# Summary of data loaders
LOADERS = {
    'price_data': load_price_data,
    'shares_outstanding': load_shares_outstanding,
    'financial_statements': load_financial_statements,
    'monthly_returns': load_monthly_returns,
    'risk_free_rate': load_risk_free_rate
}


def load_all_data(start_date: str, end_date: str, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load all 5 data sources in parallel (conceptually).
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping source names to DataFrames:
        {
            'price_data': pd.DataFrame,
            'shares_outstanding': pd.DataFrame,
            'financial_statements': pd.DataFrame,
            'monthly_returns': pd.DataFrame,
            'risk_free_rate': pd.DataFrame
        }
        
    Raises:
        FileNotFoundError: If any data file not found
        DataValidationError: If any validation fails
    """
    data = {}
    for source_name, loader_func in LOADERS.items():
        print(f"Loading {source_name}...")
        data[source_name] = loader_func(start_date, end_date, config)
        print(f"  ✓ Loaded {len(data[source_name])} rows")
    
    return data
