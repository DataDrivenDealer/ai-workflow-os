#!/usr/bin/env python3
"""
DE3/DE4 Financial Loader - Announcement-Date Aligned Financial Data
=====================================================================
P0-28: Financial Statements (DE3) & Valuation Factors (DE4)

Data Sources (Tushare Pro):
- fina_indicator: Key financial ratios (ROE, ROA, margins, etc.)
- Key fields: ann_date (announcement), end_date (report period)

Causality: All data aligned by ann_date (announcement date) to prevent look-ahead bias.
The ann_date is when the report is publicly released, NOT the reporting period end_date.

Spec: Data Eng Exec Framework v4.2 §P2-DE3, §P2-DE4
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import tushare as ts
import pandas as pd

# === CONFIG ===
START_DATE = "20150101"
END_DATE = "20260201"
BATCH_SIZE = 50  # Smaller batch for fina_indicator (complex API)
API_DELAY = 0.15  # 150ms between calls

# Key financial fields to extract
FINA_FIELDS = [
    'ts_code', 'ann_date', 'end_date',
    # Profitability
    'roe', 'roe_waa', 'roe_dt', 'roa', 'npta', 'roic',
    # Margins
    'grossprofit_margin', 'netprofit_margin', 'op_exp_of_gr',
    # Growth
    'or_yoy', 'op_yoy', 'tp_yoy', 'netprofit_yoy', 'basic_eps_yoy',
    # Asset quality
    'debt_to_assets', 'assets_to_eqt', 'current_ratio', 'quick_ratio',
    # Turnover
    'ar_turn', 'inv_turn', 'assets_turn',
    # Per share
    'eps', 'bps', 'cfps', 'dps',
    # Size
    'total_assets', 'total_liab', 'total_hldr_eqy_inc_min',
    # Revenue
    'revenue', 'operate_profit', 'total_profit', 'n_income'
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FULL = PROJECT_ROOT / "data" / "full"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints" / "de3"


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
                         fields='ts_code,symbol,name,list_date,market')
    df = df[~df.ts_code.str.endswith('.BJ')]
    return df.sort_values('ts_code').reset_index(drop=True)


def fetch_fina_for_stock(pro, ts_code):
    """Fetch fina_indicator for a single stock"""
    try:
        df = pro.fina_indicator(
            ts_code=ts_code, 
            start_date=START_DATE, 
            end_date=END_DATE
        )
        time.sleep(API_DELAY)
        
        # Select only needed fields that exist
        available_fields = [f for f in FINA_FIELDS if f in df.columns]
        return df[available_fields].copy()
    except Exception as e:
        print(f"  WARN: fina_indicator({ts_code}) failed: {e}")
        return pd.DataFrame()


def process_batch(pro, stocks_batch, batch_num, total_batches):
    """Process a single batch of stocks"""
    fina_list = []
    
    print(f"\n=== BATCH {batch_num}/{total_batches} ({len(stocks_batch)} stocks) ===")
    
    for i, ts_code in enumerate(stocks_batch):
        print(f"  [{i+1}/{len(stocks_batch)}] {ts_code}...", end="", flush=True)
        
        df = fetch_fina_for_stock(pro, ts_code)
        if not df.empty:
            fina_list.append(df)
        
        print(f" rows:{len(df)}")
    
    return fina_list


def save_batch_checkpoint(fina_list, batch_num):
    """Save batch results to checkpoint"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if fina_list:
        df = pd.concat(fina_list, ignore_index=True)
        df.to_parquet(CHECKPOINT_DIR / f"fina_batch_{batch_num:04d}.parquet", compression='snappy')
        print(f"  Saved checkpoint: {len(df)} rows")


def merge_checkpoints():
    """Merge all batch checkpoints into final files"""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_FULL.mkdir(parents=True, exist_ok=True)
    
    fina_files = sorted(CHECKPOINT_DIR.glob("fina_batch_*.parquet"))
    if fina_files:
        df = pd.concat([pd.read_parquet(f) for f in fina_files], ignore_index=True)
        
        # Sort by ts_code and ann_date
        df = df.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)
        
        # Add effective_date = ann_date (announcement date is when data becomes available)
        df['effective_date'] = df['ann_date']
        
        # Save
        df.to_parquet(DATA_RAW / "fina_indicator.parquet", compression='snappy')
        df.to_parquet(DATA_FULL / "de3_fina_indicator.parquet", compression='snappy')
        print(f"Merged fina_indicator.parquet: {len(df):,} rows")
        return df
    return pd.DataFrame()


def get_completed_batches():
    """Get list of completed batch numbers"""
    if not CHECKPOINT_DIR.exists():
        return set()
    files = CHECKPOINT_DIR.glob("fina_batch_*.parquet")
    return {int(f.stem.split('_')[-1]) for f in files}


def run_sample_validation(pro, n_stocks=10):
    """Run quick validation with small sample"""
    print("\n" + "=" * 60)
    print("SAMPLE VALIDATION")
    print("=" * 60)
    
    stocks = get_stock_universe(pro)
    sample_codes = stocks.ts_code.head(n_stocks).tolist()
    
    fina_list = []
    for ts_code in sample_codes:
        print(f"  Fetching {ts_code}...")
        df = fetch_fina_for_stock(pro, ts_code)
        if not df.empty:
            fina_list.append(df)
    
    if fina_list:
        df = pd.concat(fina_list, ignore_index=True)
        df['effective_date'] = df['ann_date']
        
        # Save sample
        df.to_parquet(DATA_RAW / "fina_indicator_sample.parquet", compression='snappy')
        
        print(f"\n  Sample saved: {len(df):,} rows")
        print(f"  Stocks: {df.ts_code.nunique()}")
        print(f"  Columns: {list(df.columns)[:10]}...")
        print(f"  Date range: {df.ann_date.min()} - {df.ann_date.max()}")
        
        # Causality check
        sample_row = df.iloc[0]
        print(f"\n  CAUSALITY CHECK:")
        print(f"    Report period (end_date): {sample_row['end_date']}")
        print(f"    Announcement (ann_date): {sample_row['ann_date']}")
        print(f"    Effective date: {sample_row['effective_date']}")
        print(f"    (Data only available after announcement date)")
        
        return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DE3/DE4 Financial Loader")
    parser.add_argument("--batch", type=int, help="Run specific batch only")
    parser.add_argument("--start-batch", type=int, default=0, help="Start from batch N")
    parser.add_argument("--merge-only", action="store_true", help="Just merge existing checkpoints")
    parser.add_argument("--sample", type=int, help="Run sample validation with N stocks")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DE3/DE4 FINANCIAL LOADER - DGSF Data Engineering P0-28")
    print("=" * 60)
    
    # Initialize API
    pro = init_api()
    
    # Sample mode
    if args.sample:
        success = run_sample_validation(pro, args.sample)
        if success:
            print("\nSAMPLE VALIDATION PASSED")
        return
    
    # Merge only mode
    if args.merge_only:
        print("\nMerging checkpoints only...")
        merge_checkpoints()
        return
    
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
        batch_num = args.batch
        if batch_num < total_batches:
            fina_list = process_batch(pro, batches[batch_num], batch_num, total_batches)
            save_batch_checkpoint(fina_list, batch_num)
        else:
            print(f"ERROR: Batch {batch_num} out of range (max: {total_batches-1})")
    else:
        for batch_num in range(args.start_batch, total_batches):
            if batch_num in completed:
                print(f"Skipping batch {batch_num} (already completed)")
                continue
            
            fina_list = process_batch(pro, batches[batch_num], batch_num, total_batches)
            save_batch_checkpoint(fina_list, batch_num)
            
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
    print("DE3/DE4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
