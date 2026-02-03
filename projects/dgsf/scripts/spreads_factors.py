"""
Cross-Sectional Spreads and Factor Computation Module

This module computes:
1. Cross-sectional spreads (style_spreads) from 5 firm characteristics
2. 5 factor portfolios: market_factor, SMB, HML, momentum_factor, reversal
3. X_state assembly for SDF encoder input

Execution Order (from Dependency Graph):
- Step 4: Compute cross-sectional spreads (requires characteristics from Step 2-3)
- Step 5: Compute factors (3-way parallel: market, SMB+HML, momentum, reversal)
- Step 6: Assemble X_state (requires spreads + characteristics)

References:
- SDF_FEATURE_DEFINITIONS.md (Factor 1-5, Execution Order)
- SDF_SPEC v3.1 Section 2.1 (X_state inputs)
- Fama-French 5-factor model (SMB, HML construction methodology)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import warnings


def compute_market_factor(
    returns: pd.DataFrame,
    risk_free_rate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute market factor (excess market return).
    
    Formula:
        market_factor[t] = mean(returns[t]) - risk_free[t]
        (Value-weighted or equal-weighted market return minus risk-free rate)
    
    Args:
        returns: DataFrame[date, firm_id, return]
        risk_free_rate: DataFrame[date, rf_rate]
        
    Returns:
        DataFrame[date, market_factor]
        
    References:
        - Factor 1 in SDF_FEATURE_DEFINITIONS.md
        - CAPM market premium
    """
    # Compute equal-weighted market return for each date
    market_return = returns.groupby('date')['return'].mean().reset_index()
    market_return.columns = ['date', 'market_return']
    
    # Merge with risk-free rate
    merged = market_return.merge(risk_free_rate, on='date', how='left')
    
    # Handle missing risk-free rates (forward-fill, then default to 0)
    if 'rf_rate' not in merged.columns:
        raise ValueError("risk_free_rate DataFrame must contain 'rf_rate' column")
    
    merged['rf_rate'] = merged['rf_rate'].ffill().fillna(0)
    
    # Compute market factor (excess return)
    merged['market_factor'] = merged['market_return'] - merged['rf_rate']
    
    # Return [date, market_factor]
    result = merged[['date', 'market_factor']]
    
    return result


def compute_smb_hml(
    size_data: pd.DataFrame,
    book_to_market_data: pd.DataFrame,
    returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute SMB (Small Minus Big) and HML (High Minus Low) factors using 2×3 sorts.
    
    Methodology (Fama-French):
        1. Sort firms into 2 size groups (median split)
        2. Within each size group, sort into 3 B/M groups (30%, 40%, 30%)
        3. Form 6 portfolios (2 × 3 = 6)
        4. Compute SMB = mean(Small portfolios) - mean(Big portfolios)
        5. Compute HML = mean(High B/M portfolios) - mean(Low B/M portfolios)
    
    Args:
        size_data: DataFrame[date, firm_id, size]
        book_to_market_data: DataFrame[date, firm_id, book_to_market]
        returns: DataFrame[date, firm_id, return]
        
    Returns:
        Tuple of (SMB, HML)
        SMB: DataFrame[date, smb_factor]
        HML: DataFrame[date, hml_factor]
        
    References:
        - Factor 2-3 in SDF_FEATURE_DEFINITIONS.md
        - Fama-French (1993) methodology
        
    Note:
        SMB and HML share the same 2×3 sorts (optimization)
    """
    # Merge size, book_to_market, and returns
    merged = size_data.merge(
        book_to_market_data, 
        on=['date', 'firm_id'], 
        how='inner'
    ).merge(
        returns, 
        on=['date', 'firm_id'], 
        how='inner'
    )
    
    # Drop missing values
    merged = merged.dropna(subset=['size', 'book_to_market', 'return'])
    
    # For each date, perform 2×3 sorts
    smb_list = []
    hml_list = []
    
    for date, group in merged.groupby('date'):
        # Step 1: Size breakpoint (median)
        size_median = group['size'].median()
        group['size_group'] = group['size'].apply(lambda x: 'Small' if x <= size_median else 'Big')
        
        # Step 2: B/M breakpoints (30%, 40%, 30% - tertiles)
        bm_30 = group['book_to_market'].quantile(0.30)
        bm_70 = group['book_to_market'].quantile(0.70)
        
        def bm_group(bm):
            if bm <= bm_30:
                return 'Low'
            elif bm <= bm_70:
                return 'Medium'
            else:
                return 'High'
        
        group['bm_group'] = group['book_to_market'].apply(bm_group)
        
        # Step 3: Form 6 portfolios (equal-weighted returns)
        portfolios = group.groupby(['size_group', 'bm_group'])['return'].mean()
        
        # Step 4: Compute SMB
        # SMB = (S/L + S/M + S/H)/3 - (B/L + B/M + B/H)/3
        small_portfolios = [
            portfolios.get(('Small', 'Low'), np.nan),
            portfolios.get(('Small', 'Medium'), np.nan),
            portfolios.get(('Small', 'High'), np.nan),
        ]
        big_portfolios = [
            portfolios.get(('Big', 'Low'), np.nan),
            portfolios.get(('Big', 'Medium'), np.nan),
            portfolios.get(('Big', 'High'), np.nan),
        ]
        
        smb = np.nanmean(small_portfolios) - np.nanmean(big_portfolios)
        
        # Step 5: Compute HML
        # HML = (S/H + B/H)/2 - (S/L + B/L)/2
        high_portfolios = [
            portfolios.get(('Small', 'High'), np.nan),
            portfolios.get(('Big', 'High'), np.nan),
        ]
        low_portfolios = [
            portfolios.get(('Small', 'Low'), np.nan),
            portfolios.get(('Big', 'Low'), np.nan),
        ]
        
        hml = np.nanmean(high_portfolios) - np.nanmean(low_portfolios)
        
        smb_list.append({'date': date, 'smb_factor': smb})
        hml_list.append({'date': date, 'hml_factor': hml})
    
    # Convert to DataFrames
    smb_df = pd.DataFrame(smb_list)
    hml_df = pd.DataFrame(hml_list)
    
    return smb_df, hml_df


def compute_momentum_factor(
    momentum_data: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute momentum factor (WML - Winners Minus Losers).
    
    Formula:
        1. Sort firms into 3 momentum groups (tertiles: 30%, 40%, 30%)
        2. WML[t] = mean(Winner returns[t]) - mean(Loser returns[t])
    
    Args:
        momentum_data: DataFrame[date, firm_id, momentum]
        returns: DataFrame[date, firm_id, return]
        
    Returns:
        DataFrame[date, momentum_factor]
        
    References:
        - Factor 4 in SDF_FEATURE_DEFINITIONS.md
        - Jegadeesh-Titman (1993) momentum strategy
    """
    # Merge momentum and returns
    merged = momentum_data.merge(returns, on=['date', 'firm_id'], how='inner')
    merged = merged.dropna(subset=['momentum', 'return'])
    
    # For each date, perform tertile sorts
    wml_list = []
    
    for date, group in merged.groupby('date'):
        # Tertile breakpoints (30%, 70%)
        mom_30 = group['momentum'].quantile(0.30)
        mom_70 = group['momentum'].quantile(0.70)
        
        # Classify firms
        losers = group[group['momentum'] <= mom_30]['return']
        winners = group[group['momentum'] >= mom_70]['return']
        
        # WML = Winners - Losers
        wml = winners.mean() - losers.mean() if len(winners) > 0 and len(losers) > 0 else np.nan
        
        wml_list.append({'date': date, 'momentum_factor': wml})
    
    return pd.DataFrame(wml_list)


def compute_reversal(
    returns: pd.DataFrame,
    lookback_months: int = 1,
) -> pd.DataFrame:
    """
    Compute short-term reversal factor.
    
    Formula:
        1. Sort firms by last month's return (t-1)
        2. Reversal[t] = mean(Loser returns[t]) - mean(Winner returns[t])
        (Note: Opposite sign to momentum - losers outperform winners)
    
    Args:
        returns: DataFrame[date, firm_id, return]
        lookback_months: Number of months to look back (default: 1)
        
    Returns:
        DataFrame[date, reversal_factor]
        
    References:
        - Factor 5 in SDF_FEATURE_DEFINITIONS.md
        - Jegadeesh (1990) short-term reversal
    """
    # Sort by date and firm
    returns_sorted = returns.sort_values(['firm_id', 'date']).copy()
    
    # Compute lagged returns (previous month)
    returns_sorted['lagged_return'] = returns_sorted.groupby('firm_id')['return'].shift(lookback_months)
    
    # Drop rows with missing lagged returns
    returns_sorted = returns_sorted.dropna(subset=['lagged_return', 'return'])
    
    # For each date, sort by lagged return and compute reversal
    reversal_list = []
    
    for date, group in returns_sorted.groupby('date'):
        # Tertile breakpoints based on lagged return
        lag_30 = group['lagged_return'].quantile(0.30)
        lag_70 = group['lagged_return'].quantile(0.70)
        
        # Classify: past losers (low lagged return), past winners (high lagged return)
        past_losers = group[group['lagged_return'] <= lag_30]['return']
        past_winners = group[group['lagged_return'] >= lag_70]['return']
        
        # Reversal = Losers - Winners (opposite sign to momentum)
        reversal = past_losers.mean() - past_winners.mean() if len(past_losers) > 0 and len(past_winners) > 0 else np.nan
        
        reversal_list.append({'date': date, 'reversal_factor': reversal})
    
    return pd.DataFrame(reversal_list)


def compute_style_spreads(
    size: pd.DataFrame,
    book_to_market: pd.DataFrame,
    momentum: pd.DataFrame,
    profitability: pd.DataFrame,
    volatility: pd.DataFrame,
    market_cap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional spreads (long-short portfolios) for 5 characteristics.
    
    Methodology:
        For each characteristic and each date t:
        1. Sort firms into tertiles (30%, 40%, 30%)
        2. Compute market-cap weighted average for top and bottom tertiles
        3. Spread = top_tertile_avg - bottom_tertile_avg
        4. Output: 5D vector [size_spread, bm_spread, mom_spread, prof_spread, vol_spread]
    
    Args:
        size: DataFrame[date, firm_id, size]
        book_to_market: DataFrame[date, firm_id, book_to_market]
        momentum: DataFrame[date, firm_id, momentum]
        profitability: DataFrame[date, firm_id, profitability]
        volatility: DataFrame[date, firm_id, volatility]
        market_cap: DataFrame[date, firm_id, market_cap] (optional, for weighting)
        
    Returns:
        DataFrame[date, size_spread, bm_spread, momentum_spread, 
                  profitability_spread, volatility_spread]
        
    References:
        - SDF_SPEC v3.1 Section 2.1 (style_spreads component of X_state)
        - Feature 1-5 in SDF_FEATURE_DEFINITIONS.md
        
    Notes:
        - If market_cap not provided, use equal weighting
        - Winsorization already applied in firm_characteristics.py
    """
    # Merge all characteristics
    characteristics = {
        'size': size,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'profitability': profitability,
        'volatility': volatility,
    }
    
    # Start with size as base
    merged = size.copy()
    for name, df in characteristics.items():
        if name != 'size':
            merged = merged.merge(df, on=['date', 'firm_id'], how='outer', suffixes=('', f'_{name}'))
    
    # Add market_cap if provided (for weighting)
    use_market_cap = market_cap is not None
    if use_market_cap:
        merged = merged.merge(market_cap, on=['date', 'firm_id'], how='left')
    
    # Compute spreads for each date
    spreads_list = []
    
    for date, group in merged.groupby('date'):
        spreads = {'date': date}
        
        for char_name in ['size', 'book_to_market', 'momentum', 'profitability', 'volatility']:
            # Drop NaN for this characteristic
            char_data = group[['firm_id', char_name]].dropna()
            
            if len(char_data) < 3:
                # Not enough data for tertiles
                spreads[f'{char_name}_spread'] = np.nan
                continue
            
            # Tertile breakpoints (30%, 70%)
            q30 = char_data[char_name].quantile(0.30)
            q70 = char_data[char_name].quantile(0.70)
            
            # Classify into tertiles
            bottom_tertile = char_data[char_data[char_name] <= q30]
            top_tertile = char_data[char_data[char_name] >= q70]
            
            # Compute weighted or equal-weighted averages
            if use_market_cap and 'market_cap' in group.columns:
                # Market-cap weighted
                bottom_firms = group[group['firm_id'].isin(bottom_tertile['firm_id'])]
                top_firms = group[group['firm_id'].isin(top_tertile['firm_id'])]
                
                bottom_weights = bottom_firms['market_cap'] / bottom_firms['market_cap'].sum()
                top_weights = top_firms['market_cap'] / top_firms['market_cap'].sum()
                
                bottom_avg = (bottom_firms[char_name] * bottom_weights).sum()
                top_avg = (top_firms[char_name] * top_weights).sum()
            else:
                # Equal-weighted
                bottom_avg = bottom_tertile[char_name].mean()
                top_avg = top_tertile[char_name].mean()
            
            # Spread = top - bottom (long high, short low)
            spreads[f'{char_name}_spread'] = top_avg - bottom_avg
        
        spreads_list.append(spreads)
    
    return pd.DataFrame(spreads_list)


def assemble_X_state(
    characteristics: Dict[str, pd.DataFrame],
    spreads: pd.DataFrame,
    factors: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Assemble X_state input for SDF encoder.
    
    Formula:
        X_state[t, d] = concat([
            characteristics[t],  # 5D: size, book_to_market, momentum, profitability, volatility
            spreads[t],          # 5D: cross-sectional spreads
            factors[t]           # 5D (optional): market, SMB, HML, momentum_factor, reversal
        ])
        where d = 10 (characteristics + spreads) or 15 (if factors included)
    
    Args:
        characteristics: Dict of 5 DataFrames (size, book_to_market, momentum, 
                         profitability, volatility), each [date, firm_id, value]
        spreads: DataFrame[date, size_spread, bm_spread, momentum_spread, 
                 profitability_spread, volatility_spread]
        factors: Dict of 5 DataFrames (market_factor, smb, hml, momentum_factor, 
                 reversal), each [date, factor_value] (optional)
        
    Returns:
        DataFrame[date, X_state_dim_0, X_state_dim_1, ..., X_state_dim_d]
        where d = 10 or 15
        
    References:
        - SDF_SPEC v3.1 Section 2.1 (X_state = macro + style_spreads + ...)
        - Section 2.2 (Encoder h_θ: X_state → z_t)
        
    Data Quality:
        - All inputs must be monthly frequency, aligned to same date index
        - Missing values forward-filled (max 1 month)
        - Verify no data leakage: X_state[t] uses only data up to t-1
    """
    # Compute cross-sectional means for characteristics (aggregate firm-level to market-level)
    char_aggregates = []
    
    for char_name, char_df in characteristics.items():
        # Compute equal-weighted cross-sectional mean for each date
        char_mean = char_df.groupby('date')[char_name].mean().reset_index()
        char_mean.columns = ['date', f'{char_name}_mean']
        char_aggregates.append(char_mean)
    
    # Merge all characteristic means
    X_state = char_aggregates[0]
    for char_agg in char_aggregates[1:]:
        X_state = X_state.merge(char_agg, on='date', how='outer')
    
    # Merge spreads
    X_state = X_state.merge(spreads, on='date', how='outer')
    
    # Merge factors if provided
    if factors is not None:
        for factor_name, factor_df in factors.items():
            X_state = X_state.merge(factor_df, on='date', how='outer')
    
    # Sort by date
    X_state = X_state.sort_values('date').reset_index(drop=True)
    
    # Forward-fill missing values (max 1 month)
    for col in X_state.columns:
        if col != 'date':
            X_state[col] = X_state[col].ffill(limit=1)
    
    # Rename columns to X_state_dim_i format
    feature_cols = [col for col in X_state.columns if col != 'date']
    for i, col in enumerate(feature_cols):
        X_state = X_state.rename(columns={col: f'X_state_dim_{i}'})
    
    return X_state
