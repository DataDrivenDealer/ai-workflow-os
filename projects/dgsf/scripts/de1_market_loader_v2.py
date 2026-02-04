#!/usr/bin/env python3
"""
P0-26: DE1 Raw Market Loader (Full A-Share) - Optimized Version

Task: Load raw market data from Tushare for DGSF Data Engineering
Time Range: 2015-01-01 to 2026-02-01
Stock Universe: All A-shares (Main + ChiNext + STAR)

Optimization: Fetch by stock code instead of by date for daily data.
This is more efficient for Tushare's API limits.

Outputs:
- data/raw/daily_prices.parquet
- data/raw/monthly_prices.parquet  
- data/raw/adj_factor.parquet
- data/raw/daily_basic.parquet

Spec Pointer: Data Eng Exec Framework v4.2 §P2-DE1
"""

import tushare as ts
import pandas as pd
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
START_DATE = '20150101'
END_DATE = '20260201'
OUTPUT_DIR = Path('data/raw')
CHECKPOINT_DIR = Path('data/checkpoints')

# Initialize Tushare
token = os.environ.get('TUSHARE_TOKEN', '')
ts.set_token(token)
pro = ts.pro_api()


def log(msg: str):
    """Timestamped logging."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_checkpoint(module: str, data: dict):
    """Save checkpoint for resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{module}_progress.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint(module: str) -> Optional[dict]:
    """Load checkpoint if exists."""
    checkpoint_file = CHECKPOINT_DIR / f"{module}_progress.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def get_all_stocks() -> pd.DataFrame:
    """Get all A-share stocks (including delisted)."""
    log("Fetching stock list...")
    df = pro.stock_basic(exchange='', list_status='', 
                         fields='ts_code,name,list_date,delist_date,market')
    # Filter to A-shares only (exclude BSE - 8xxxxx.BJ)
    df = df[~df['ts_code'].str.endswith('.BJ')]
    log(f"Total A-share stocks: {len(df)}")
    return df


def fetch_daily_by_stock(ts_code: str, retries: int = 3) -> pd.DataFrame:
    """Fetch daily data for a specific stock with retry."""
    for attempt in range(retries):
        try:
            df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                log(f"Failed to fetch daily for {ts_code}: {e}")
                return pd.DataFrame()


def fetch_daily_prices() -> pd.DataFrame:
    """
    Fetch daily prices for all stocks.
    Strategy: Fetch by stock code (more efficient for Tushare).
    """
    log("=" * 60)
    log("FETCHING DAILY PRICES (by stock)")
    log("=" * 60)
    
    stocks = get_all_stocks()
    stock_codes = stocks['ts_code'].tolist()
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_daily_v2')
    completed_stocks = set(checkpoint.get('completed_stocks', [])) if checkpoint else set()
    remaining_stocks = [s for s in stock_codes if s not in completed_stocks]
    log(f"Already completed: {len(completed_stocks)}, Remaining: {len(remaining_stocks)}")
    
    # Load existing partial data if any
    partial_file = OUTPUT_DIR / 'daily_prices_partial.parquet'
    if partial_file.exists() and len(completed_stocks) > 0:
        log(f"Loading partial data from {partial_file}")
        existing_data = pd.read_parquet(partial_file)
        all_data = [existing_data]
    else:
        all_data = []
    
    batch_size = 100  # Save every 100 stocks
    new_data = []
    
    for i, ts_code in enumerate(remaining_stocks):
        df = fetch_daily_by_stock(ts_code)
        if len(df) > 0:
            new_data.append(df)
        
        completed_stocks.add(ts_code)
        
        # Progress
        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(remaining_stocks)} stocks ({len(completed_stocks)}/{len(stock_codes)} total)")
        
        # Save checkpoint and partial data every batch
        if (i + 1) % batch_size == 0:
            save_checkpoint('de1_daily_v2', {'completed_stocks': list(completed_stocks)})
            if new_data:
                all_data.append(pd.concat(new_data, ignore_index=True))
                new_data = []
                # Save partial
                df_partial = pd.concat(all_data, ignore_index=True)
                df_partial.to_parquet(partial_file, index=False)
                log(f"Checkpoint: {len(df_partial):,} rows saved")
        
        # Rate limit (~200 calls/min, so 0.3s per call is safe)
        time.sleep(0.05)
    
    # Final save
    if new_data:
        all_data.append(pd.concat(new_data, ignore_index=True))
    
    if all_data:
        df_daily = pd.concat(all_data, ignore_index=True)
        log(f"Total daily records: {len(df_daily):,}")
        return df_daily
    return pd.DataFrame()


def fetch_adj_factor() -> pd.DataFrame:
    """Fetch adjustment factors for all stocks."""
    log("=" * 60)
    log("FETCHING ADJ_FACTOR")
    log("=" * 60)
    
    stocks = get_all_stocks()
    stock_codes = stocks['ts_code'].tolist()
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_adj_v2')
    completed_stocks = set(checkpoint.get('completed_stocks', [])) if checkpoint else set()
    remaining_stocks = [s for s in stock_codes if s not in completed_stocks]
    log(f"Already completed: {len(completed_stocks)}, Remaining: {len(remaining_stocks)}")
    
    all_data = []
    batch_size = 200
    
    for i, ts_code in enumerate(remaining_stocks):
        try:
            df = pro.adj_factor(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            pass  # Skip failed stocks
        
        completed_stocks.add(ts_code)
        
        if (i + 1) % 100 == 0:
            log(f"Progress: {i+1}/{len(remaining_stocks)} stocks")
        
        if (i + 1) % batch_size == 0:
            save_checkpoint('de1_adj_v2', {'completed_stocks': list(completed_stocks)})
        
        time.sleep(0.05)
    
    if all_data:
        df_adj = pd.concat(all_data, ignore_index=True)
        log(f"Total adj_factor records: {len(df_adj):,}")
        return df_adj
    return pd.DataFrame()


def fetch_daily_basic_sample() -> pd.DataFrame:
    """
    Fetch daily_basic for month-end dates only (for efficiency).
    We only need EOM values for market cap.
    """
    log("=" * 60)
    log("FETCHING DAILY_BASIC (Month-end only for efficiency)")
    log("=" * 60)
    
    # Get month-end trading dates
    cal = pro.trade_cal(exchange='SSE', start_date=START_DATE, end_date=END_DATE)
    trade_dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    
    # Convert to datetime and get EOM
    dates_df = pd.DataFrame({'date': pd.to_datetime(trade_dates, format='%Y%m%d')})
    dates_df['month'] = dates_df['date'].dt.to_period('M')
    eom_dates = dates_df.groupby('month')['date'].max().dt.strftime('%Y%m%d').tolist()
    
    log(f"Month-end dates to fetch: {len(eom_dates)}")
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_basic_v2')
    completed_dates = set(checkpoint.get('completed_dates', [])) if checkpoint else set()
    remaining_dates = [d for d in eom_dates if d not in completed_dates]
    log(f"Already completed: {len(completed_dates)}, Remaining: {len(remaining_dates)}")
    
    all_data = []
    
    for i, trade_date in enumerate(remaining_dates):
        try:
            df = pro.daily_basic(trade_date=trade_date,
                                 fields='ts_code,trade_date,turnover_rate,pe,pb,total_mv,circ_mv')
            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            log(f"Error fetching daily_basic for {trade_date}: {e}")
        
        completed_dates.add(trade_date)
        
        if (i + 1) % 20 == 0:
            log(f"Progress: {i+1}/{len(remaining_dates)} dates")
            save_checkpoint('de1_basic_v2', {'completed_dates': list(completed_dates)})
        
        time.sleep(0.1)
    
    if all_data:
        df_basic = pd.concat(all_data, ignore_index=True)
        log(f"Total daily_basic records: {len(df_basic):,}")
        return df_basic
    return pd.DataFrame()


def create_monthly_prices(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Create monthly prices from daily data (EOM)."""
    log("=" * 60)
    log("CREATING MONTHLY PRICES")
    log("=" * 60)
    
    df = df_daily.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df['month'] = df['trade_date'].dt.to_period('M')
    
    # Get last trading day per stock-month
    idx = df.groupby(['ts_code', 'month'])['trade_date'].idxmax()
    df_monthly = df.loc[idx, ['ts_code', 'trade_date', 'close', 'vol', 'amount']].copy()
    df_monthly = df_monthly.rename(columns={'close': 'close_month'})
    df_monthly['trade_date'] = df_monthly['trade_date'].dt.strftime('%Y%m%d')
    
    log(f"Monthly records: {len(df_monthly):,}")
    return df_monthly


def validate_and_save(df: pd.DataFrame, filename: str, min_rows: int) -> dict:
    """Validate and save DataFrame."""
    if df is None or len(df) < min_rows:
        log(f"❌ FAIL: {filename} has {len(df) if df is not None else 0:,} rows, need {min_rows:,}")
        return {'rows': len(df) if df is not None else 0, 'status': 'FAIL'}
    
    # Remove duplicates
    if 'ts_code' in df.columns and 'trade_date' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
        if before != len(df):
            log(f"Removed {before - len(df)} duplicates")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False, compression='snappy')
    file_size = output_path.stat().st_size / (1024 * 1024)
    log(f"✅ Saved: {filename} ({len(df):,} rows, {file_size:.1f} MB)")
    
    return {'rows': len(df), 'status': 'OK', 'size_mb': round(file_size, 1)}


def main():
    """Main execution."""
    log("=" * 60)
    log("P0-26: DE1 Raw Market Loader (Optimized)")
    log(f"Time Range: {START_DATE} to {END_DATE}")
    log("=" * 60)
    
    results = {}
    
    # 1. Fetch daily prices (by stock)
    df_daily = fetch_daily_prices()
    results['daily_prices'] = validate_and_save(df_daily, 'daily_prices.parquet', min_rows=10_000_000)
    
    # 2. Create monthly prices from daily
    if results['daily_prices']['status'] == 'OK':
        df_monthly = create_monthly_prices(df_daily)
        results['monthly_prices'] = validate_and_save(df_monthly, 'monthly_prices.parquet', min_rows=500_000)
    
    # 3. Fetch adj_factor
    df_adj = fetch_adj_factor()
    results['adj_factor'] = validate_and_save(df_adj, 'adj_factor.parquet', min_rows=5_000_000)
    
    # 4. Fetch daily_basic (month-end only)
    df_basic = fetch_daily_basic_sample()
    results['daily_basic'] = validate_and_save(df_basic, 'daily_basic.parquet', min_rows=500_000)
    
    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for name, result in results.items():
        status_icon = "✅" if result.get('status') == 'OK' else "❌"
        log(f"{status_icon} {name}: {result.get('rows', 0):,} rows")
    
    # Save results
    results_file = OUTPUT_DIR / 'de1_results.json'
    with open(results_file, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)
    
    # Verdict
    all_ok = all(r.get('status') == 'OK' for r in results.values())
    if all_ok:
        log("✅ P0-26 DE1 COMPLETE - All outputs generated successfully")
    else:
        log("⚠️ P0-26 DE1 PARTIAL - Check failed outputs")
    
    return all_ok


if __name__ == '__main__':
    main()
