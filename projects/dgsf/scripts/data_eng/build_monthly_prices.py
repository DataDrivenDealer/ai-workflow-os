"""
Build DE1 monthly canonical prices from daily prices.

DE1_MONTHLY_001: Generate monthly aggregated prices from raw daily data.

Usage:
    python projects/dgsf/scripts/build_monthly_prices.py
"""

import pandas as pd
from pathlib import Path


def build_monthly_prices(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly prices from daily prices.
    
    Aggregates daily data to monthly frequency:
    - date: YYYYMM format
    - trade_date: Last trading day of month (YYYYMMDD)
    - close: Close price on last trading day
    - open: Open price on first trading day
    - high: Highest high during month
    - low: Lowest low during month
    - vol: Total volume during month
    - amount: Total amount during month
    """
    if daily_df.empty:
        return pd.DataFrame()
    
    df = daily_df.copy()
    if not pd.api.types.is_integer_dtype(df['trade_date']):
        df['trade_date'] = df['trade_date'].astype('int32')
    
    # Extract year-month for grouping
    df['year_month'] = df['trade_date'] // 100
    
    # Group by stock and month
    monthly_data = []
    for (ts_code, ym), group in df.groupby(['ts_code', 'year_month']):
        group = group.sort_values('trade_date')
        monthly_data.append({
            'ts_code': ts_code,
            'date': ym,  # YYYYMM format
            'trade_date': group['trade_date'].iloc[-1],  # Last trading day
            'close': group['close'].iloc[-1],
            'open': group['open'].iloc[0],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'vol': group['vol'].sum(),
            'amount': group['amount'].sum(),
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df['date'] = monthly_df['date'].astype('int32')
    monthly_df['trade_date'] = monthly_df['trade_date'].astype('int32')
    monthly_df = monthly_df.sort_values(['ts_code', 'date']).reset_index(drop=True)
    
    return monthly_df


def main():
    # Paths
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    full_dir = Path(__file__).parent.parent / 'data' / 'full'
    full_dir.mkdir(exist_ok=True)
    
    # Load daily prices
    daily_path = raw_dir / 'daily_prices.parquet'
    print(f"Loading {daily_path}...")
    daily_df = pd.read_parquet(daily_path)
    print(f"  Loaded: {len(daily_df):,} rows, {daily_df['ts_code'].nunique():,} stocks")
    print(f"  Date range: {daily_df['trade_date'].min()} to {daily_df['trade_date'].max()}")
    
    # Build monthly prices
    print("\nBuilding monthly prices...")
    monthly_df = build_monthly_prices(daily_df)
    print(f"  Built: {len(monthly_df):,} rows, {monthly_df['ts_code'].nunique():,} stocks")
    print(f"  Date range: {monthly_df['date'].min()} to {monthly_df['date'].max()}")
    
    # Save
    output_path = full_dir / 'de1_canonical_monthly.parquet'
    monthly_df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Also save to raw for compatibility
    raw_output = raw_dir / 'monthly_prices.parquet'
    monthly_df.to_parquet(raw_output, index=False)
    print(f"  Also saved to: {raw_output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
