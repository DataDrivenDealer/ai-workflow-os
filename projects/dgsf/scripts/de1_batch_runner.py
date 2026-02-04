#!/usr/bin/env python3
"""
DE1 Batch Runner - Robust Market Data Loader with Batched Execution
====================================================================
P0-26: Raw Market Loader (DE1) - DGSF Data Engineering Stage 6

Design:
- Batch stocks in groups of 100 to avoid timeout
- Save checkpoints after each batch
- Resume from last checkpoint automatically

Usage:
  python scripts/de1_batch_runner.py [--batch N] [--start-batch N]

Spec: Data Eng Exec Framework v4.2 §P2-DE1
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import tushare as ts
import pandas as pd

# === CONFIG ===
START_DATE = "20150101"
END_DATE = "20260201"
BATCH_SIZE = 100  # Process 100 stocks per batch
API_DELAY = 0.12  # 120ms between API calls (Tushare limit)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "de1"

def init_api():
    """Initialize Tushare API"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("ERROR: TUSHARE_TOKEN not set")
        sys.exit(1)
    ts.set_token(token)
    return ts.pro_api()

def get_stock_universe(pro):
    """Get full A-share stock list (excluding BSE)"""
    df = pro.stock_basic(exchange='', list_status='', 
                         fields='ts_code,symbol,name,area,industry,list_date,delist_date,market')
    # Exclude Beijing Stock Exchange
    df = df[~df.ts_code.str.endswith('.BJ')]
    return df.sort_values('ts_code').reset_index(drop=True)

def fetch_daily_for_stock(pro, ts_code):
    """Fetch daily OHLCV for a single stock"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
        time.sleep(API_DELAY)
        return df
    except Exception as e:
        print(f"  WARN: daily({ts_code}) failed: {e}")
        return pd.DataFrame()

def fetch_adj_factor_for_stock(pro, ts_code):
    """Fetch adj_factor for a single stock"""
    try:
        df = pro.adj_factor(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
        time.sleep(API_DELAY)
        return df
    except Exception as e:
        print(f"  WARN: adj_factor({ts_code}) failed: {e}")
        return pd.DataFrame()

def process_batch(pro, stocks_batch, batch_num, total_batches):
    """Process a single batch of stocks"""
    daily_list = []
    adj_list = []
    
    print(f"\n=== BATCH {batch_num}/{total_batches} ({len(stocks_batch)} stocks) ===")
    
    for i, ts_code in enumerate(stocks_batch):
        print(f"  [{i+1}/{len(stocks_batch)}] {ts_code}...", end="", flush=True)
        
        # Fetch daily
        df_daily = fetch_daily_for_stock(pro, ts_code)
        if not df_daily.empty:
            daily_list.append(df_daily)
        
        # Fetch adj_factor
        df_adj = fetch_adj_factor_for_stock(pro, ts_code)
        if not df_adj.empty:
            adj_list.append(df_adj)
        
        print(f" daily:{len(df_daily)}, adj:{len(df_adj)}")
    
    return daily_list, adj_list

def save_batch_checkpoint(daily_list, adj_list, batch_num):
    """Save batch results to checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if daily_list:
        df = pd.concat(daily_list, ignore_index=True)
        df.to_parquet(CHECKPOINT_DIR / f"daily_batch_{batch_num:04d}.parquet", compression='snappy')
        print(f"  Saved daily checkpoint: {len(df)} rows")
    
    if adj_list:
        df = pd.concat(adj_list, ignore_index=True)
        df.to_parquet(CHECKPOINT_DIR / f"adj_batch_{batch_num:04d}.parquet", compression='snappy')
        print(f"  Saved adj checkpoint: {len(df)} rows")

def merge_checkpoints():
    """Merge all batch checkpoints into final files"""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # Merge daily
    daily_files = sorted(CHECKPOINT_DIR.glob("daily_batch_*.parquet"))
    if daily_files:
        df = pd.concat([pd.read_parquet(f) for f in daily_files], ignore_index=True)
        df.to_parquet(DATA_RAW / "daily_prices.parquet", compression='snappy')
        print(f"Merged daily_prices.parquet: {len(df):,} rows")
    
    # Merge adj_factor
    adj_files = sorted(CHECKPOINT_DIR.glob("adj_batch_*.parquet"))
    if adj_files:
        df = pd.concat([pd.read_parquet(f) for f in adj_files], ignore_index=True)
        df.to_parquet(DATA_RAW / "adj_factor.parquet", compression='snappy')
        print(f"Merged adj_factor.parquet: {len(df):,} rows")

def get_completed_batches():
    """Get list of completed batch numbers"""
    if not CHECKPOINT_DIR.exists():
        return set()
    files = CHECKPOINT_DIR.glob("daily_batch_*.parquet")
    return {int(f.stem.split('_')[-1]) for f in files}

def main():
    parser = argparse.ArgumentParser(description="DE1 Batch Market Loader")
    parser.add_argument("--batch", type=int, help="Run specific batch only")
    parser.add_argument("--start-batch", type=int, default=0, help="Start from batch N")
    parser.add_argument("--merge-only", action="store_true", help="Just merge existing checkpoints")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DE1 BATCH RUNNER - DGSF Data Engineering P0-26")
    print("=" * 60)
    
    if args.merge_only:
        print("\nMerging checkpoints only...")
        merge_checkpoints()
        return
    
    # Initialize API
    pro = init_api()
    
    # Get stock universe
    print("\nFetching stock universe...")
    stocks = get_stock_universe(pro)
    print(f"Total stocks: {len(stocks)}")
    
    # Split into batches
    stock_codes = stocks.ts_code.tolist()
    batches = [stock_codes[i:i+BATCH_SIZE] for i in range(0, len(stock_codes), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Total batches: {total_batches} (@ {BATCH_SIZE} stocks each)")
    
    # Check completed batches
    completed = get_completed_batches()
    print(f"Previously completed batches: {len(completed)}")
    
    # Process batches
    start_time = datetime.now()
    
    if args.batch is not None:
        # Run single batch
        batch_num = args.batch
        if batch_num < total_batches:
            daily_list, adj_list = process_batch(pro, batches[batch_num], batch_num, total_batches)
            save_batch_checkpoint(daily_list, adj_list, batch_num)
        else:
            print(f"ERROR: Batch {batch_num} out of range (max: {total_batches-1})")
    else:
        # Run all batches from start point
        for batch_num in range(args.start_batch, total_batches):
            if batch_num in completed:
                print(f"Skipping batch {batch_num} (already completed)")
                continue
            
            daily_list, adj_list = process_batch(pro, batches[batch_num], batch_num, total_batches)
            save_batch_checkpoint(daily_list, adj_list, batch_num)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            batches_done = batch_num - args.start_batch + 1
            eta_seconds = (elapsed / batches_done) * (total_batches - batch_num - 1)
            print(f"  ETA: {eta_seconds/60:.1f} minutes remaining")
    
    # Final merge
    print("\n" + "=" * 60)
    print("MERGING CHECKPOINTS")
    print("=" * 60)
    merge_checkpoints()
    
    print("\n" + "=" * 60)
    print("DE1 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
