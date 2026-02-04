#!/usr/bin/env python3
"""
P0-26: DE1 Raw Market Loader - Validation Run

This is a VALIDATION script that tests the data pipeline with a small sample.
Full data loading requires extended runtime (~30+ minutes).

Sample: 100 stocks, full time range
Purpose: Validate API access, schema, and pipeline logic.
"""

import tushare as ts
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

START_DATE = '20150101'
END_DATE = '20260201'
OUTPUT_DIR = Path('data/raw')
SAMPLE_SIZE = 100  # Small sample for validation

token = os.environ.get('TUSHARE_TOKEN', '')
ts.set_token(token)
pro = ts.pro_api()

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    log("=" * 60)
    log("P0-26 VALIDATION RUN (Sample)")
    log(f"Sample: {SAMPLE_SIZE} stocks")
    log("=" * 60)
    
    # Get sample stocks (diverse: SH Main, SZ Main, ChiNext, STAR)
    df_stocks = pro.stock_basic(exchange='', list_status='L', 
                                fields='ts_code,name,list_date,market')
    df_stocks = df_stocks[~df_stocks['ts_code'].str.endswith('.BJ')]
    
    # Sample: 25 from each board type
    sh_main = df_stocks[df_stocks['ts_code'].str.startswith('60')].head(25)['ts_code'].tolist()
    sz_main = df_stocks[df_stocks['ts_code'].str.startswith('00')].head(25)['ts_code'].tolist()
    chinext = df_stocks[df_stocks['ts_code'].str.startswith('30')].head(25)['ts_code'].tolist()
    star = df_stocks[df_stocks['ts_code'].str.startswith('68')].head(25)['ts_code'].tolist()
    
    sample_stocks = sh_main + sz_main + chinext + star
    log(f"Sample stocks: {len(sample_stocks)} (SH:{len(sh_main)}, SZ:{len(sz_main)}, ChiNext:{len(chinext)}, STAR:{len(star)})")
    
    # 1. Fetch daily prices for sample
    log("\n--- DAILY PRICES ---")
    daily_data = []
    for i, ts_code in enumerate(sample_stocks):
        try:
            df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df) > 0:
                daily_data.append(df)
        except Exception as e:
            log(f"Error {ts_code}: {e}")
        if (i+1) % 20 == 0:
            log(f"Progress: {i+1}/{len(sample_stocks)}")
    
    df_daily = pd.concat(daily_data, ignore_index=True) if daily_data else pd.DataFrame()
    log(f"Daily rows: {len(df_daily):,}")
    
    # Check schema
    expected_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']
    missing_cols = [c for c in expected_cols if c not in df_daily.columns]
    log(f"Schema check: {'✅ PASS' if not missing_cols else f'❌ MISSING: {missing_cols}'}")
    
    # Check date range
    if len(df_daily) > 0:
        min_date = df_daily['trade_date'].min()
        max_date = df_daily['trade_date'].max()
        log(f"Date range: {min_date} to {max_date}")
        date_ok = min_date <= '20150131' and max_date >= '20260101'
        log(f"Date coverage: {'✅ PASS' if date_ok else '❌ FAIL'}")
    
    # Check duplicates
    if len(df_daily) > 0:
        dups = df_daily.duplicated(subset=['ts_code', 'trade_date']).sum()
        log(f"Duplicates: {dups} {'✅' if dups == 0 else '⚠️'}")
    
    # 2. Fetch adj_factor for sample
    log("\n--- ADJ_FACTOR ---")
    adj_data = []
    for ts_code in sample_stocks[:20]:  # Smaller sample
        try:
            df = pro.adj_factor(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df) > 0:
                adj_data.append(df)
        except:
            pass
    
    df_adj = pd.concat(adj_data, ignore_index=True) if adj_data else pd.DataFrame()
    log(f"Adj_factor rows: {len(df_adj):,}")
    log(f"Schema: {'✅ PASS' if 'adj_factor' in df_adj.columns else '❌ FAIL'}")
    
    # 3. Fetch daily_basic for 1 month-end date
    log("\n--- DAILY_BASIC ---")
    try:
        df_basic = pro.daily_basic(trade_date='20260117',
                                   fields='ts_code,trade_date,turnover_rate,pe,pb,total_mv,circ_mv')
        log(f"Daily_basic rows: {len(df_basic):,}")
        log(f"Schema: {'✅ PASS' if 'circ_mv' in df_basic.columns else '❌ FAIL'}")
    except Exception as e:
        log(f"Daily_basic error: {e}")
        df_basic = pd.DataFrame()
    
    # 4. Create monthly from daily
    log("\n--- MONTHLY PRICES ---")
    if len(df_daily) > 0:
        df = df_daily.copy()
        df['trade_date_dt'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df['month'] = df['trade_date_dt'].dt.to_period('M')
        idx = df.groupby(['ts_code', 'month'])['trade_date_dt'].idxmax()
        df_monthly = df.loc[idx, ['ts_code', 'trade_date', 'close']].copy()
        log(f"Monthly rows: {len(df_monthly):,}")
    
    # Summary
    log("\n" + "=" * 60)
    log("VALIDATION SUMMARY")
    log("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(sample_stocks),
        'daily_prices': {
            'rows': len(df_daily),
            'schema_ok': not missing_cols,
            'date_range_ok': date_ok if len(df_daily) > 0 else False,
        },
        'adj_factor': {
            'rows': len(df_adj),
            'schema_ok': 'adj_factor' in df_adj.columns if len(df_adj) > 0 else False,
        },
        'daily_basic': {
            'rows': len(df_basic),
            'schema_ok': 'circ_mv' in df_basic.columns if len(df_basic) > 0 else False,
        },
    }
    
    # Save sample data for inspection
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(df_daily) > 0:
        df_daily.to_parquet(OUTPUT_DIR / 'daily_prices_sample.parquet', index=False)
    if len(df_adj) > 0:
        df_adj.to_parquet(OUTPUT_DIR / 'adj_factor_sample.parquet', index=False)
    
    # Save results
    with open(OUTPUT_DIR / 'de1_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Verdict
    all_ok = (
        results['daily_prices']['rows'] > 100000 and
        results['daily_prices']['schema_ok'] and
        results['adj_factor']['rows'] > 10000
    )
    
    if all_ok:
        log("✅ VALIDATION PASSED")
        log("Pipeline logic verified. Full data load requires ~30+ min runtime.")
    else:
        log("❌ VALIDATION FAILED - Check results above")
    
    # Extrapolation
    if len(df_daily) > 0:
        avg_rows_per_stock = len(df_daily) / len(sample_stocks)
        estimated_total = avg_rows_per_stock * 5186
        log(f"\nEstimated full daily_prices: {estimated_total:,.0f} rows")
        log(f"Estimated runtime: {5186 * 0.1 / 60:.1f} minutes")


if __name__ == '__main__':
    main()
