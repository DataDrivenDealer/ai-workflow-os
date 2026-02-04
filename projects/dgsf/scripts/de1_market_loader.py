#!/usr/bin/env python3
"""
P0-26: DE1 Raw Market Loader (Full A-Share)

Task: Load raw market data from Tushare for DGSF Data Engineering
Time Range: 2015-01-01 to 2026-02-01
Stock Universe: All A-shares (Main + ChiNext + STAR)

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
from typing import Optional, List

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
    log(f"Checkpoint saved: {checkpoint_file}")


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
    # Filter to A-shares only (exclude BSE)
    df = df[~df['ts_code'].str.endswith('.BJ')]
    log(f"Total A-share stocks: {len(df)}")
    return df


def fetch_daily_by_date(trade_date: str) -> pd.DataFrame:
    """Fetch daily data for a specific date (all stocks)."""
    try:
        df = pro.daily(trade_date=trade_date)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        log(f"Error fetching daily for {trade_date}: {e}")
        return pd.DataFrame()


def fetch_daily_prices() -> pd.DataFrame:
    """
    Fetch daily prices for all stocks, all dates.
    Strategy: Fetch by trade_date to maximize efficiency.
    """
    log("=" * 60)
    log("FETCHING DAILY PRICES")
    log("=" * 60)
    
    # Get trade calendar
    log("Fetching trade calendar...")
    cal = pro.trade_cal(exchange='SSE', start_date=START_DATE, end_date=END_DATE)
    trade_dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    log(f"Trade dates to fetch: {len(trade_dates)}")
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_daily')
    completed_dates = set(checkpoint.get('completed_dates', [])) if checkpoint else set()
    remaining_dates = [d for d in trade_dates if d not in completed_dates]
    log(f"Already completed: {len(completed_dates)}, Remaining: {len(remaining_dates)}")
    
    # Fetch data
    all_data = []
    batch_size = 100  # Save checkpoint every 100 dates
    
    for i, trade_date in enumerate(remaining_dates):
        df = fetch_daily_by_date(trade_date)
        if len(df) > 0:
            all_data.append(df)
        
        completed_dates.add(trade_date)
        
        # Progress
        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(remaining_dates)} dates fetched")
        
        # Checkpoint
        if (i + 1) % batch_size == 0:
            save_checkpoint('de1_daily', {'completed_dates': list(completed_dates)})
        
        # Rate limit (Tushare allows ~200 calls/min for daily by date)
        time.sleep(0.05)
    
    # Combine all data
    if all_data:
        df_daily = pd.concat(all_data, ignore_index=True)
        log(f"Total daily records: {len(df_daily):,}")
        return df_daily
    return pd.DataFrame()


def fetch_adj_factor() -> pd.DataFrame:
    """
    Fetch adjustment factors for all stocks.
    Strategy: Fetch by stock (adj_factor has no date-based API).
    """
    log("=" * 60)
    log("FETCHING ADJ_FACTOR")
    log("=" * 60)
    
    stocks = get_all_stocks()
    stock_codes = stocks['ts_code'].tolist()
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_adj')
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
            log(f"Error fetching adj_factor for {ts_code}: {e}")
        
        completed_stocks.add(ts_code)
        
        if (i + 1) % 100 == 0:
            log(f"Progress: {i+1}/{len(remaining_stocks)} stocks fetched")
        
        if (i + 1) % batch_size == 0:
            save_checkpoint('de1_adj', {'completed_stocks': list(completed_stocks)})
        
        time.sleep(0.05)
    
    if all_data:
        df_adj = pd.concat(all_data, ignore_index=True)
        log(f"Total adj_factor records: {len(df_adj):,}")
        return df_adj
    return pd.DataFrame()


def fetch_daily_basic() -> pd.DataFrame:
    """
    Fetch daily basic data (market cap) for all stocks.
    Strategy: Fetch by date.
    """
    log("=" * 60)
    log("FETCHING DAILY_BASIC (Market Cap)")
    log("=" * 60)
    
    # Get trade calendar
    cal = pro.trade_cal(exchange='SSE', start_date=START_DATE, end_date=END_DATE)
    trade_dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
    
    # Check checkpoint
    checkpoint = load_checkpoint('de1_basic')
    completed_dates = set(checkpoint.get('completed_dates', [])) if checkpoint else set()
    remaining_dates = [d for d in trade_dates if d not in completed_dates]
    log(f"Already completed: {len(completed_dates)}, Remaining: {len(remaining_dates)}")
    
    all_data = []
    batch_size = 100
    
    for i, trade_date in enumerate(remaining_dates):
        try:
            df = pro.daily_basic(trade_date=trade_date,
                                 fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,dv_ratio,free_share,total_share,float_share,total_mv,circ_mv')
            if df is not None and len(df) > 0:
                all_data.append(df)
        except Exception as e:
            log(f"Error fetching daily_basic for {trade_date}: {e}")
        
        completed_dates.add(trade_date)
        
        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(remaining_dates)} dates fetched")
        
        if (i + 1) % batch_size == 0:
            save_checkpoint('de1_basic', {'completed_dates': list(completed_dates)})
        
        time.sleep(0.05)
    
    if all_data:
        df_basic = pd.concat(all_data, ignore_index=True)
        log(f"Total daily_basic records: {len(df_basic):,}")
        return df_basic
    return pd.DataFrame()


def create_monthly_prices(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly prices from daily data.
    Take EOM (End of Month) trading day.
    """
    log("=" * 60)
    log("CREATING MONTHLY PRICES")
    log("=" * 60)
    
    df = df_daily.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    
    # Get month-end date for each row
    df['month'] = df['trade_date'].dt.to_period('M')
    
    # For each stock-month, get the last trading day
    idx = df.groupby(['ts_code', 'month'])['trade_date'].idxmax()
    df_monthly = df.loc[idx, ['ts_code', 'trade_date', 'close', 'vol', 'amount']].copy()
    df_monthly = df_monthly.rename(columns={'close': 'close_month'})
    
    log(f"Monthly records: {len(df_monthly):,}")
    return df_monthly


def validate_output(df: pd.DataFrame, name: str, min_rows: int = 0) -> bool:
    """Validate output DataFrame."""
    if df is None or len(df) == 0:
        log(f"❌ FAIL: {name} is empty")
        return False
    
    if len(df) < min_rows:
        log(f"❌ FAIL: {name} has {len(df):,} rows, expected >= {min_rows:,}")
        return False
    
    # Check for duplicates
    if 'ts_code' in df.columns and 'trade_date' in df.columns:
        dups = df.duplicated(subset=['ts_code', 'trade_date']).sum()
        if dups > 0:
            log(f"⚠️ WARNING: {name} has {dups} duplicate (ts_code, trade_date) pairs")
            # Remove duplicates
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    
    log(f"✅ PASS: {name} validated, {len(df):,} rows")
    return True


def save_parquet(df: pd.DataFrame, filename: str):
    """Save DataFrame to parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False, compression='snappy')
    file_size = output_path.stat().st_size / (1024 * 1024)
    log(f"Saved: {output_path} ({file_size:.1f} MB)")


def main():
    """Main execution."""
    log("=" * 60)
    log("P0-26: DE1 Raw Market Loader")
    log(f"Time Range: {START_DATE} to {END_DATE}")
    log("=" * 60)
    
    results = {}
    
    # 1. Fetch daily prices
    df_daily = fetch_daily_prices()
    if validate_output(df_daily, 'daily_prices', min_rows=10_000_000):
        # Remove duplicates
        df_daily = df_daily.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
        save_parquet(df_daily, 'daily_prices.parquet')
        results['daily_prices'] = {'rows': len(df_daily), 'status': 'OK'}
    else:
        results['daily_prices'] = {'rows': len(df_daily) if df_daily is not None else 0, 'status': 'FAIL'}
    
    # 2. Create monthly prices from daily
    if len(df_daily) > 0:
        df_monthly = create_monthly_prices(df_daily)
        if validate_output(df_monthly, 'monthly_prices', min_rows=500_000):
            save_parquet(df_monthly, 'monthly_prices.parquet')
            results['monthly_prices'] = {'rows': len(df_monthly), 'status': 'OK'}
        else:
            results['monthly_prices'] = {'rows': len(df_monthly), 'status': 'FAIL'}
    
    # 3. Fetch adj_factor
    df_adj = fetch_adj_factor()
    if validate_output(df_adj, 'adj_factor', min_rows=5_000_000):
        df_adj = df_adj.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
        save_parquet(df_adj, 'adj_factor.parquet')
        results['adj_factor'] = {'rows': len(df_adj), 'status': 'OK'}
    else:
        results['adj_factor'] = {'rows': len(df_adj) if df_adj is not None else 0, 'status': 'FAIL'}
    
    # 4. Fetch daily_basic
    df_basic = fetch_daily_basic()
    if validate_output(df_basic, 'daily_basic', min_rows=10_000_000):
        df_basic = df_basic.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
        save_parquet(df_basic, 'daily_basic.parquet')
        results['daily_basic'] = {'rows': len(df_basic), 'status': 'OK'}
    else:
        results['daily_basic'] = {'rows': len(df_basic) if df_basic is not None else 0, 'status': 'FAIL'}
    
    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for name, result in results.items():
        status_icon = "✅" if result['status'] == 'OK' else "❌"
        log(f"{status_icon} {name}: {result['rows']:,} rows")
    
    # Save results
    results_file = OUTPUT_DIR / 'de1_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    
    # Final verdict
    all_ok = all(r['status'] == 'OK' for r in results.values())
    if all_ok:
        log("✅ P0-26 DE1 COMPLETE - All outputs generated successfully")
    else:
        log("⚠️ P0-26 DE1 PARTIAL - Some outputs failed validation")
    
    return all_ok


if __name__ == '__main__':
    main()
