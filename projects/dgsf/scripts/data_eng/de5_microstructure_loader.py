#!/usr/bin/env python3
"""
DE5/DE6 Microstructure & Universe Mask Loader
==============================================
P0-29: Microstructure (DE5) & Universe Mask (DE6)

Data Sources (Tushare Pro):
- daily_basic: Daily turnover, market cap, volume ratio
- Computed: Universe mask (ST, delisted, suspension, liquidity filters)

Note: daily_basic is fetched by DATE (not by stock) for efficiency.
Each date returns ~5000+ stocks.

Spec: Data Eng Exec Framework v4.2 §P2-DE5, §P2-DE6
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import tushare as ts
import pandas as pd
import numpy as np

# === CONFIG ===
START_DATE = "20150101"
END_DATE = "20260201"
API_DELAY = 0.12

# Fields to extract from daily_basic
BASIC_FIELDS = [
    'ts_code', 'trade_date',
    'turnover_rate', 'turnover_rate_f',  # Free-float turnover
    'volume_ratio',
    'total_mv', 'circ_mv',  # Market cap
    'pe', 'pe_ttm', 'pb',  # Valuation
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FULL = PROJECT_ROOT / "data" / "full"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "de5"


def init_api():
    """Initialize Tushare API"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("ERROR: TUSHARE_TOKEN not set")
        sys.exit(1)
    ts.set_token(token)
    return ts.pro_api()


def get_trading_dates(pro, start_date, end_date):
    """Get list of trading dates"""
    df = pro.trade_cal(
        exchange='SSE', 
        start_date=start_date, 
        end_date=end_date,
        is_open=1
    )
    return sorted(df.cal_date.tolist())


def fetch_daily_basic_for_date(pro, trade_date):
    """Fetch daily_basic for a single trading date"""
    try:
        fields_str = ','.join(BASIC_FIELDS)
        df = pro.daily_basic(trade_date=trade_date, fields=fields_str)
        time.sleep(API_DELAY)
        return df
    except Exception as e:
        print(f"  WARN: daily_basic({trade_date}) failed: {e}")
        return pd.DataFrame()


def get_completed_dates():
    """Get list of completed dates from checkpoints"""
    if not CHECKPOINT_DIR.exists():
        return set()
    files = CHECKPOINT_DIR.glob("basic_*.parquet")
    return {f.stem.split('_')[1] for f in files}


def process_date_batch(pro, dates, batch_start, batch_size):
    """Process a batch of dates"""
    basic_list = []
    
    for i, trade_date in enumerate(dates[batch_start:batch_start+batch_size]):
        idx = batch_start + i
        print(f"  [{idx+1}/{len(dates)}] {trade_date}...", end="", flush=True)
        
        df = fetch_daily_basic_for_date(pro, trade_date)
        if not df.empty:
            basic_list.append(df)
        
        print(f" rows:{len(df)}")
    
    return basic_list


def save_checkpoint(basic_list, checkpoint_id):
    """Save checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if basic_list:
        df = pd.concat(basic_list, ignore_index=True)
        df.to_parquet(CHECKPOINT_DIR / f"basic_{checkpoint_id}.parquet", compression='snappy')
        print(f"  Saved checkpoint {checkpoint_id}: {len(df)} rows")


def merge_checkpoints():
    """Merge all checkpoints"""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_FULL.mkdir(parents=True, exist_ok=True)
    
    files = sorted(CHECKPOINT_DIR.glob("basic_*.parquet"))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        
        # Remove BSE stocks
        df = df[~df.ts_code.str.endswith('.BJ')]
        
        # Sort
        df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        # Save
        df.to_parquet(DATA_RAW / "daily_basic.parquet", compression='snappy')
        df.to_parquet(DATA_FULL / "de5_daily_basic.parquet", compression='snappy')
        print(f"Merged daily_basic.parquet: {len(df):,} rows")
        return df
    return pd.DataFrame()


def build_universe_mask(basic_df, pro):
    """Build universe mask based on filters"""
    print("\nBuilding universe mask...")
    
    # Get ST stock list
    # For simplicity, we'll mark stocks with extreme valuations or low liquidity
    # In production, you'd use a proper ST stock list
    
    mask = basic_df[['ts_code', 'trade_date', 'turnover_rate', 'total_mv', 'circ_mv']].copy()
    
    # Initialize mask to True (tradeable)
    mask['is_tradeable'] = True
    
    # Filter 1: No market cap (suspended/delisted)
    mask.loc[mask['total_mv'].isna() | (mask['total_mv'] <= 0), 'is_tradeable'] = False
    
    # Filter 2: Extremely low turnover (< 0.1%)
    mask.loc[mask['turnover_rate'] < 0.1, 'is_tradeable'] = False
    
    # Filter 3: Micro-cap threshold (< 500M CNY circ_mv)
    mask.loc[mask['circ_mv'] < 5, 'is_tradeable'] = False  # circ_mv in 亿元
    
    # Summary
    total = len(mask)
    tradeable = mask['is_tradeable'].sum()
    print(f"  Total rows: {total:,}")
    print(f"  Tradeable: {tradeable:,} ({tradeable/total*100:.1f}%)")
    print(f"  Filtered: {total-tradeable:,} ({(total-tradeable)/total*100:.1f}%)")
    
    # Save
    mask_out = mask[['ts_code', 'trade_date', 'is_tradeable']]
    mask_out.to_parquet(DATA_FULL / "de6_universe_mask.parquet", compression='snappy')
    print(f"  Saved de6_universe_mask.parquet: {len(mask_out):,} rows")
    
    return mask_out


def run_sample_validation(pro, n_days=5):
    """Run quick validation with small sample"""
    print("\n" + "=" * 60)
    print("SAMPLE VALIDATION")
    print("=" * 60)
    
    # Get recent trading dates
    dates = get_trading_dates(pro, '20250101', '20250131')[:n_days]
    print(f"Testing {len(dates)} dates: {dates[0]} - {dates[-1]}")
    
    basic_list = []
    for trade_date in dates:
        print(f"  Fetching {trade_date}...")
        df = fetch_daily_basic_for_date(pro, trade_date)
        if not df.empty:
            basic_list.append(df)
    
    if basic_list:
        df = pd.concat(basic_list, ignore_index=True)
        df = df[~df.ts_code.str.endswith('.BJ')]
        
        # Save sample
        df.to_parquet(DATA_RAW / "daily_basic_sample.parquet", compression='snappy')
        
        print(f"\n  Sample saved: {len(df):,} rows")
        print(f"  Dates: {df.trade_date.nunique()}")
        print(f"  Stocks per date: ~{len(df) // df.trade_date.nunique()}")
        print(f"  Columns: {list(df.columns)}")
        
        return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DE5/DE6 Microstructure Loader")
    parser.add_argument("--sample", type=int, help="Run sample validation with N days")
    parser.add_argument("--merge-only", action="store_true", help="Just merge checkpoints")
    parser.add_argument("--batch-size", type=int, default=50, help="Dates per batch")
    parser.add_argument("--start-date", default=START_DATE, help="Start date")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DE5/DE6 MICROSTRUCTURE LOADER - DGSF Data Engineering P0-29")
    print("=" * 60)
    
    pro = init_api()
    
    if args.sample:
        run_sample_validation(pro, args.sample)
        return
    
    if args.merge_only:
        df = merge_checkpoints()
        if not df.empty:
            build_universe_mask(df, pro)
        return
    
    # Get trading dates
    print("\nFetching trading calendar...")
    dates = get_trading_dates(pro, args.start_date, END_DATE)
    print(f"Total trading dates: {len(dates)}")
    
    # Process in batches
    completed = get_completed_dates()
    print(f"Previously completed checkpoints: {len(completed)}")
    
    start_time = datetime.now()
    batch_num = 0
    
    for i in range(0, len(dates), args.batch_size):
        batch_dates = dates[i:i+args.batch_size]
        checkpoint_id = f"{batch_dates[0]}_{batch_dates[-1]}"
        
        if checkpoint_id in completed:
            print(f"Skipping batch {checkpoint_id} (already completed)")
            batch_num += 1
            continue
        
        print(f"\n=== BATCH {batch_num} ({checkpoint_id}) ===")
        basic_list = process_date_batch(pro, dates, i, args.batch_size)
        save_checkpoint(basic_list, checkpoint_id)
        
        batch_num += 1
        elapsed = (datetime.now() - start_time).total_seconds()
        remaining_batches = (len(dates) - i - args.batch_size) // args.batch_size
        eta = (elapsed / batch_num) * remaining_batches if batch_num > 0 else 0
        print(f"  ETA: {eta/60:.1f} minutes")
    
    # Final merge
    print("\n" + "=" * 60)
    print("MERGING CHECKPOINTS")
    print("=" * 60)
    df = merge_checkpoints()
    
    if not df.empty:
        build_universe_mask(df, pro)
    
    print("\n" + "=" * 60)
    print("DE5/DE6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
