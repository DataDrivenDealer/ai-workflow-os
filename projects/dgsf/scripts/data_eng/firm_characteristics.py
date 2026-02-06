"""
Firm Characteristics Computation Module

This module computes 5 firm-level characteristics for SDF feature engineering:
1. size: Log market capitalization
2. momentum: 12-month cumulative return (excluding last month)
3. profitability: Operating income / book equity
4. volatility: Realized volatility over 12 months
5. book_to_market: Book equity / market capitalization

All characteristics are winsorized to handle outliers and aligned to monthly frequency.

References:
- SDF_FEATURE_DEFINITIONS.md (Feature 1-5)
- SDF_SPEC v3.1 Section 2.1 (X_state inputs)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


def winsorize(
    df: pd.DataFrame,
    columns: list,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize specified columns to handle outliers.
    
    Args:
        df: DataFrame with firm characteristics
        columns: List of column names to winsorize
        lower: Lower quantile (default: 1%)
        upper: Upper quantile (default: 99%)
        
    Returns:
        DataFrame with winsorized values
        
    Note:
        - Different characteristics use different percentiles:
          * size, momentum, volatility: [1%, 99%]
          * book_to_market, profitability: [0.5%, 99.5%]
    """
    df_winsorized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame, skipping winsorization")
            continue
            
        # Compute cross-sectional quantiles at each date
        lower_bound = df.groupby('date')[col].transform(lambda x: x.quantile(lower))
        upper_bound = df.groupby('date')[col].transform(lambda x: x.quantile(upper))
        
        # Clip values
        df_winsorized[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Count winsorized observations
        n_winsorized = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if n_winsorized > 0:
            pct_winsorized = 100 * n_winsorized / len(df)
            warnings.warn(
                f"Winsorized {n_winsorized} observations ({pct_winsorized:.2f}%) "
                f"for '{col}' at [{lower*100}%, {upper*100}%] percentiles"
            )
    
    return df_winsorized


def compute_size(
    price_data: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    min_market_cap: float = 5e6,
) -> pd.DataFrame:
    """
    Compute size (log market capitalization).
    
    Formula:
        size[firm, t] = log(market_cap[firm, t])
        where market_cap = price * shares_outstanding
    
    Args:
        price_data: DataFrame[date, firm_id, price]
        shares_outstanding: DataFrame[date, firm_id, shares]
        min_market_cap: Minimum market cap threshold (default: $5M, microcap filter)
        
    Returns:
        DataFrame[date, firm_id, size] with log market cap
        
    Data Quality:
        - Firms with market_cap < min_market_cap excluded
        - Missing values forward-filled (max 3 months)
        - Winsorized at [1%, 99%]
    """
    # Merge price and shares on [date, firm_id]
    merged = price_data.merge(
        shares_outstanding,
        on=['date', 'firm_id'],
        how='inner',
        suffixes=('_price', '_shares')
    )
    
    # Compute market cap
    merged['market_cap'] = merged['price'] * merged['shares']
    
    # Apply microcap filter
    n_before = len(merged)
    merged = merged[merged['market_cap'] >= min_market_cap]
    n_after = len(merged)
    if n_before > n_after:
        warnings.warn(
            f"Excluded {n_before - n_after} observations ({100*(n_before-n_after)/n_before:.1f}%) "
            f"with market_cap < ${min_market_cap/1e6:.1f}M"
        )
    
    # Compute log size
    merged['size'] = np.log(merged['market_cap'])
    
    # Forward-fill missing values (max 3 months)
    merged = merged.sort_values(['firm_id', 'date'])
    merged['size'] = merged.groupby('firm_id')['size'].transform(
        lambda x: x.ffill(limit=3)
    )
    
    # Winsorize
    result = merged[['date', 'firm_id', 'size']].copy()
    result = winsorize(result, ['size'], lower=0.01, upper=0.99)
    
    return result


def compute_momentum(
    returns: pd.DataFrame,
    lookback_months: int = 12,
    skip_last_month: bool = True,
    min_obs: int = 8,
) -> pd.DataFrame:
    """
    Compute momentum (cumulative return over past 12 months, excluding last month).
    
    Formula:
        momentum[firm, t] = Π_{i=2}^{12} (1 + R[firm, t-i]) - 1
        (Excludes t-1 to avoid microstructure noise)
    
    Args:
        returns: DataFrame[date, firm_id, return]
        lookback_months: Number of months to look back (default: 12)
        skip_last_month: Whether to skip the most recent month (default: True)
        min_obs: Minimum observations required (default: 8)
        
    Returns:
        DataFrame[date, firm_id, momentum] with cumulative returns
        
    Data Quality:
        - Firms with < min_obs months excluded
        - Winsorized at [1%, 99%]
    """
    # Sort by firm and date
    returns = returns.sort_values(['firm_id', 'date']).copy()
    
    # Determine lag start (2 if skip_last_month, else 1)
    lag_start = 2 if skip_last_month else 1
    lag_end = lookback_months + (1 if skip_last_month else 0)
    
    # Compute cumulative return using rolling window
    def compute_cumulative_return(group):
        """Compute cumulative return for each date in group"""
        result_list = []
        
        for idx in range(len(group)):
            # Get return window [t-lag_end, t-lag_start]
            start_idx = max(0, idx - lag_end + 1)
            end_idx = max(0, idx - lag_start + 1)
            
            if end_idx - start_idx < min_obs:
                result_list.append(np.nan)
                continue
            
            # Compute cumulative return: Π(1+r) - 1
            window_returns = group.iloc[start_idx:end_idx]['return'].values
            if len(window_returns) < min_obs:
                result_list.append(np.nan)
            else:
                cum_return = np.prod(1 + window_returns) - 1
                result_list.append(cum_return)
        
        return pd.Series(result_list, index=group.index)
    
    # Apply momentum computation
    returns['momentum'] = returns.groupby('firm_id', group_keys=False)['return'].apply(
        lambda group_series: compute_cumulative_return(
            returns.loc[group_series.index]
        )
    )
    
    # Exclude firms with insufficient history
    n_before = len(returns)
    returns = returns.dropna(subset=['momentum'])
    n_after = len(returns)
    if n_before > n_after:
        warnings.warn(
            f"Excluded {n_before - n_after} observations ({100*(n_before-n_after)/n_before:.1f}%) "
            f"with < {min_obs} months of return history"
        )
    
    # Winsorize
    result = returns[['date', 'firm_id', 'momentum']].copy()
    result = winsorize(result, ['momentum'], lower=0.01, upper=0.99)
    
    return result


def compute_profitability(
    financial_statements: pd.DataFrame,
    min_book_equity: float = 0.0,
) -> pd.DataFrame:
    """
    Compute profitability (operating income / book equity).
    
    Formula:
        profitability[firm, t] = operating_income[firm, t] / book_equity[firm, t]
        where book_equity = total_assets - total_liabilities - preferred_stock
    
    Args:
        financial_statements: DataFrame[date, firm_id, operating_income, 
                                        total_assets, total_liabilities, 
                                        stockholders_equity, ...]
        min_book_equity: Minimum book equity threshold (default: 0, exclude negative)
        
    Returns:
        DataFrame[date, firm_id, profitability] with operating ROE
        
    Data Quality:
        - Firms with book_equity <= min_book_equity excluded
        - Forward-filled quarterly data to monthly (max 3 months)
        - Winsorized at [0.5%, 99.5%]
    """
    # Compute book equity
    # Preferred stock may not be available, default to 0
    if 'preferred_stock' in financial_statements.columns:
        book_equity = (
            financial_statements['total_assets']
            - financial_statements['total_liabilities']
            - financial_statements['preferred_stock']
        )
    else:
        # Fallback: use stockholders_equity directly
        if 'stockholders_equity' in financial_statements.columns:
            book_equity = financial_statements['stockholders_equity']
        else:
            book_equity = (
                financial_statements['total_assets']
                - financial_statements['total_liabilities']
            )
    
    financial_statements = financial_statements.copy()
    financial_statements['book_equity'] = book_equity
    
    # Exclude negative book equity
    n_before = len(financial_statements)
    financial_statements = financial_statements[
        financial_statements['book_equity'] > min_book_equity
    ]
    n_after = len(financial_statements)
    if n_before > n_after:
        warnings.warn(
            f"Excluded {n_before - n_after} observations ({100*(n_before-n_after)/n_before:.1f}%) "
            f"with book_equity <= ${min_book_equity/1e6:.1f}M"
        )
    
    # Compute profitability
    financial_statements['profitability'] = (
        financial_statements['operating_income'] / financial_statements['book_equity']
    )
    
    # Forward-fill quarterly data to monthly (max 3 months)
    financial_statements = financial_statements.sort_values(['firm_id', 'date'])
    financial_statements['profitability'] = financial_statements.groupby('firm_id')['profitability'].transform(
        lambda x: x.ffill(limit=3)
    )
    
    # Winsorize (more conservative: 0.5%, 99.5%)
    result = financial_statements[['date', 'firm_id', 'profitability']].copy()
    result = winsorize(result, ['profitability'], lower=0.005, upper=0.995)
    
    return result


def compute_volatility(
    returns: pd.DataFrame,
    lookback_months: int = 12,
    min_obs: int = 6,
) -> pd.DataFrame:
    """
    Compute volatility (realized volatility over past 12 months).
    
    Formula:
        volatility[firm, t] = sqrt( (1/n) * Σ_{i=1}^{n} (R[firm, t-i] - μ[firm, t])^2 )
        where n = lookback_months, μ = mean monthly return
    
    Args:
        returns: DataFrame[date, firm_id, return]
        lookback_months: Number of months to compute volatility (default: 12)
        min_obs: Minimum observations required (default: 6)
        
    Returns:
        DataFrame[date, firm_id, volatility] with realized volatility
        
    Data Quality:
        - Firms with < min_obs months excluded
        - Winsorized at [1%, 99%]
        
    Note:
        This is TOTAL volatility, not idiosyncratic (no factor model residuals)
    """
    # Sort by firm and date
    returns = returns.sort_values(['firm_id', 'date']).copy()
    
    # Compute rolling volatility
    returns['volatility'] = returns.groupby('firm_id')['return'].transform(
        lambda x: x.rolling(window=lookback_months, min_periods=min_obs).std()
    )
    
    # Exclude firms with insufficient history
    n_before = len(returns)
    result = returns.dropna(subset=['volatility'])
    n_after = len(result)
    if n_before > n_after:
        warnings.warn(
            f"Excluded {n_before - n_after} observations ({100*(n_before-n_after)/n_before:.1f}%) "
            f"with < {min_obs} months of return history"
        )
    
    # Winsorize
    result = result[['date', 'firm_id', 'volatility']].copy()
    result = winsorize(result, ['volatility'], lower=0.01, upper=0.99)
    
    return result


def compute_book_to_market(
    financial_statements: pd.DataFrame,
    size_data: pd.DataFrame,
    min_book_equity: float = 0.0,
) -> pd.DataFrame:
    """
    Compute book_to_market (book equity / market capitalization).
    
    Formula:
        book_to_market[firm, t] = book_equity[firm, t] / market_cap[firm, t]
        where market_cap = exp(size[firm, t]) (from compute_size output)
    
    Args:
        financial_statements: DataFrame[date, firm_id, total_assets, total_liabilities, ...]
        size_data: DataFrame[date, firm_id, size] (log market cap from compute_size)
        min_book_equity: Minimum book equity threshold (default: 0, exclude negative)
        
    Returns:
        DataFrame[date, firm_id, book_to_market]
        
    Data Quality:
        - Firms with book_equity <= min_book_equity excluded
        - Forward-filled quarterly financial data to monthly (max 3 months)
        - Winsorized at [0.5%, 99.5%]
        
    Note:
        This depends on size, so must be computed AFTER compute_size
    """
    # Compute book equity (same as profitability)
    if 'preferred_stock' in financial_statements.columns:
        book_equity = (
            financial_statements['total_assets']
            - financial_statements['total_liabilities']
            - financial_statements['preferred_stock']
        )
    else:
        if 'stockholders_equity' in financial_statements.columns:
            book_equity = financial_statements['stockholders_equity']
        else:
            book_equity = (
                financial_statements['total_assets']
                - financial_statements['total_liabilities']
            )
    
    financial_statements = financial_statements.copy()
    financial_statements['book_equity'] = book_equity
    
    # Exclude negative book equity
    n_before = len(financial_statements)
    financial_statements = financial_statements[
        financial_statements['book_equity'] > min_book_equity
    ]
    n_after = len(financial_statements)
    if n_before > n_after:
        warnings.warn(
            f"Excluded {n_before - n_after} observations ({100*(n_before-n_after)/n_before:.1f}%) "
            f"with book_equity <= ${min_book_equity/1e6:.1f}M (book_to_market computation)"
        )
    
    # Merge with size data to get market cap
    merged = financial_statements.merge(
        size_data,
        on=['date', 'firm_id'],
        how='inner'
    )
    
    # Convert log size back to market cap
    merged['market_cap'] = np.exp(merged['size'])
    
    # Compute book_to_market
    merged['book_to_market'] = merged['book_equity'] / merged['market_cap']
    
    # Forward-fill quarterly data to monthly (max 3 months)
    merged = merged.sort_values(['firm_id', 'date'])
    merged['book_to_market'] = merged.groupby('firm_id')['book_to_market'].transform(
        lambda x: x.ffill(limit=3)
    )
    
    # Winsorize (more conservative: 0.5%, 99.5%)
    result = merged[['date', 'firm_id', 'book_to_market']].copy()
    result = winsorize(result, ['book_to_market'], lower=0.005, upper=0.995)
    
    return result


def compute_all_characteristics(
    price_data: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    financial_statements: pd.DataFrame,
    returns: pd.DataFrame,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute all 5 firm characteristics in dependency order.
    
    Execution Order:
        Step 2 (Independent, parallel):
            1. size = compute_size(price, shares)
            2. momentum = compute_momentum(returns)
            3. profitability = compute_profitability(financials)
            4. volatility = compute_volatility(returns)
        
        Step 3 (Dependent):
            5. book_to_market = compute_book_to_market(financials, size)
    
    Args:
        price_data: DataFrame[date, firm_id, price]
        shares_outstanding: DataFrame[date, firm_id, shares]
        financial_statements: DataFrame[date, firm_id, operating_income, 
                                        total_assets, total_liabilities, ...]
        returns: DataFrame[date, firm_id, return]
        **kwargs: Additional arguments passed to individual functions
        
    Returns:
        Tuple of (size, momentum, profitability, volatility, book_to_market)
        Each is DataFrame[date, firm_id, characteristic_value]
    """
    print("Computing firm characteristics...")
    
    # Step 2: Independent characteristics (can be parallelized)
    print("  [1/5] Computing size...")
    size = compute_size(price_data, shares_outstanding, **kwargs.get('size_kwargs', {}))
    
    print("  [2/5] Computing momentum...")
    momentum = compute_momentum(returns, **kwargs.get('momentum_kwargs', {}))
    
    print("  [3/5] Computing profitability...")
    profitability = compute_profitability(financial_statements, **kwargs.get('profitability_kwargs', {}))
    
    print("  [4/5] Computing volatility...")
    volatility = compute_volatility(returns, **kwargs.get('volatility_kwargs', {}))
    
    # Step 3: Dependent characteristic (requires size)
    print("  [5/5] Computing book_to_market...")
    book_to_market = compute_book_to_market(
        financial_statements,
        size,
        **kwargs.get('book_to_market_kwargs', {})
    )
    
    print(f"✓ All characteristics computed successfully")
    print(f"  size: {len(size)} observations")
    print(f"  momentum: {len(momentum)} observations")
    print(f"  profitability: {len(profitability)} observations")
    print(f"  volatility: {len(volatility)} observations")
    print(f"  book_to_market: {len(book_to_market)} observations")
    
    return size, momentum, profitability, volatility, book_to_market
