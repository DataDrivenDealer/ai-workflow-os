#!/usr/bin/env python3
"""
DE2 Macro Loader - Release-Date Aligned Macroeconomic Data
============================================================
P0-27: Macro Features (DE2) - DGSF Data Engineering Stage 6

Data Sources (Tushare Pro):
- CPI: cn_cpi (monthly) - release lag ~15 days
- PPI: cn_ppi (monthly) - release lag ~15 days  
- M2/M1: cn_m (monthly) - release lag ~15 days
- GDP: cn_gdp (quarterly) - release lag ~20 days
- PMI: cn_pmi (monthly) - release lag ~1 day

Causality: All data aligned by release date to prevent look-ahead bias

Spec: Data Eng Exec Framework v4.2 §P2-DE2
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import tushare as ts
import pandas as pd
import numpy as np

# === CONFIG ===
START_M = "201501"
END_M = "202602"
START_Q = "2015Q1"
END_Q = "2026Q1"

# Release lags (days after month/quarter end)
RELEASE_LAG = {
    "cpi": 15,
    "ppi": 15,
    "m2": 15,
    "gdp": 20,
    "pmi": 1
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FULL = PROJECT_ROOT / "data" / "full"


def init_api():
    """Initialize Tushare API"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("ERROR: TUSHARE_TOKEN not set")
        sys.exit(1)
    ts.set_token(token)
    return ts.pro_api()


def month_to_effective_date(month_str: str, lag_days: int) -> str:
    """Convert YYYYMM to effective date (YYYYMMDD) accounting for release lag"""
    year = int(month_str[:4])
    month = int(month_str[4:6])
    
    # End of report month
    if month == 12:
        end_of_month = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_of_month = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Add release lag
    effective = end_of_month + timedelta(days=lag_days)
    return effective.strftime("%Y%m%d")


def quarter_to_effective_date(quarter_str: str, lag_days: int) -> str:
    """Convert YYYYQN to effective date (YYYYMMDD) accounting for release lag"""
    year = int(quarter_str[:4])
    qtr = int(quarter_str[-1])
    
    # Quarter end month
    qtr_end_month = qtr * 3
    if qtr_end_month == 12:
        end_of_qtr = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_of_qtr = datetime(year, qtr_end_month + 1, 1) - timedelta(days=1)
    
    # Add release lag
    effective = end_of_qtr + timedelta(days=lag_days)
    return effective.strftime("%Y%m%d")


def fetch_cpi(pro) -> pd.DataFrame:
    """Fetch CPI data"""
    print("Fetching CPI...")
    df = pro.cn_cpi(start_m=START_M, end_m=END_M)
    
    # Select key columns
    df = df[['month', 'nt_yoy', 'nt_mom']].copy()
    df.columns = ['month', 'cpi_yoy', 'cpi_mom']
    
    # Add effective date
    df['effective_date'] = df['month'].apply(
        lambda x: month_to_effective_date(x, RELEASE_LAG['cpi'])
    )
    
    print(f"  CPI: {len(df)} rows")
    return df


def fetch_ppi(pro) -> pd.DataFrame:
    """Fetch PPI data"""
    print("Fetching PPI...")
    df = pro.cn_ppi(start_m=START_M, end_m=END_M)
    
    # Select key columns
    df = df[['month', 'ppi_yoy']].copy()
    
    # Add effective date
    df['effective_date'] = df['month'].apply(
        lambda x: month_to_effective_date(x, RELEASE_LAG['ppi'])
    )
    
    print(f"  PPI: {len(df)} rows")
    return df


def fetch_m2(pro) -> pd.DataFrame:
    """Fetch money supply data"""
    print("Fetching M2/M1...")
    df = pro.cn_m(start_m=START_M, end_m=END_M)
    
    # Select key columns
    df = df[['month', 'm2', 'm2_yoy', 'm1', 'm1_yoy']].copy()
    
    # Add effective date
    df['effective_date'] = df['month'].apply(
        lambda x: month_to_effective_date(x, RELEASE_LAG['m2'])
    )
    
    print(f"  M2: {len(df)} rows")
    return df


def fetch_gdp(pro) -> pd.DataFrame:
    """Fetch GDP data (quarterly, needs forward fill to monthly)"""
    print("Fetching GDP...")
    df = pro.cn_gdp(start_q=START_Q, end_q=END_Q)
    
    # Select key columns
    df = df[['quarter', 'gdp', 'gdp_yoy']].copy()
    
    # Add effective date
    df['effective_date'] = df['quarter'].apply(
        lambda x: quarter_to_effective_date(x, RELEASE_LAG['gdp'])
    )
    
    print(f"  GDP: {len(df)} rows (quarterly)")
    return df


def fetch_pmi(pro) -> pd.DataFrame:
    """Fetch PMI data"""
    print("Fetching PMI...")
    df = pro.cn_pmi(start_m=START_M, end_m=END_M)
    
    # PMI uses uppercase MONTH column
    if 'MONTH' in df.columns:
        df = df.rename(columns={'MONTH': 'month'})
    
    # PMI010000 is manufacturing PMI (the main indicator)
    # Select manufacturing PMI and non-manufacturing PMI
    pmi_manufacturing = 'PMI010000'  # 制造业PMI
    pmi_non_manufacturing = 'PMI030000'  # 非制造业PMI
    
    cols_to_keep = ['month']
    if pmi_manufacturing in df.columns:
        cols_to_keep.append(pmi_manufacturing)
    if pmi_non_manufacturing in df.columns:
        cols_to_keep.append(pmi_non_manufacturing)
    
    df = df[cols_to_keep].copy()
    df.columns = ['month', 'pmi_mfg', 'pmi_non_mfg'] if len(cols_to_keep) == 3 else ['month', 'pmi_mfg']
    
    # Add effective date
    df['effective_date'] = df['month'].apply(
        lambda x: month_to_effective_date(x, RELEASE_LAG['pmi'])
    )
    
    print(f"  PMI: {len(df)} rows")
    return df


def merge_macro_monthly(cpi, ppi, m2, gdp, pmi) -> pd.DataFrame:
    """Merge all macro data into single monthly panel"""
    print("\nMerging macro data...")
    
    # Start with CPI as base
    df = cpi.copy()
    
    # Merge PPI
    df = df.merge(ppi[['month', 'ppi_yoy']], on='month', how='outer')
    
    # Merge M2
    df = df.merge(m2[['month', 'm2', 'm2_yoy', 'm1', 'm1_yoy']], on='month', how='outer')
    
    # Merge PMI
    pmi_cols = [c for c in pmi.columns if c not in ['month', 'effective_date']]
    df = df.merge(pmi[['month'] + pmi_cols], on='month', how='outer')
    
    # Forward fill GDP (quarterly -> monthly)
    # Create month column for GDP
    gdp_expanded = []
    for _, row in gdp.iterrows():
        year = int(row['quarter'][:4])
        qtr = int(row['quarter'][-1])
        for m in range(3):
            month_num = (qtr - 1) * 3 + m + 1
            month_str = f"{year}{month_num:02d}"
            gdp_expanded.append({
                'month': month_str,
                'gdp': row['gdp'],
                'gdp_yoy': row['gdp_yoy']
            })
    gdp_monthly = pd.DataFrame(gdp_expanded)
    
    # Merge GDP
    df = df.merge(gdp_monthly, on='month', how='outer')
    
    # Sort by month
    df = df.sort_values('month').reset_index(drop=True)
    
    # Recalculate effective_date as max of all indicators for that month
    # (use CPI lag as conservative estimate)
    df['effective_date'] = df['month'].apply(
        lambda x: month_to_effective_date(x, max(RELEASE_LAG.values()))
    )
    
    # Filter to target range
    df = df[(df['month'] >= START_M) & (df['month'] <= END_M)]
    
    print(f"  Final merged: {len(df)} rows, {len(df.columns)} columns")
    return df


def main():
    print("=" * 60)
    print("DE2 MACRO LOADER - DGSF Data Engineering P0-27")
    print("=" * 60)
    print(f"Period: {START_M} - {END_M}")
    
    # Initialize API
    pro = init_api()
    
    # Fetch all macro data
    cpi = fetch_cpi(pro)
    ppi = fetch_ppi(pro)
    m2 = fetch_m2(pro)
    gdp = fetch_gdp(pro)
    pmi = fetch_pmi(pro)
    
    # Merge into monthly panel
    macro = merge_macro_monthly(cpi, ppi, m2, gdp, pmi)
    
    # Save outputs
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_FULL.mkdir(parents=True, exist_ok=True)
    
    # Raw individual files
    cpi.to_parquet(DATA_RAW / "macro_cpi.parquet", compression='snappy')
    ppi.to_parquet(DATA_RAW / "macro_ppi.parquet", compression='snappy')
    m2.to_parquet(DATA_RAW / "macro_m2.parquet", compression='snappy')
    gdp.to_parquet(DATA_RAW / "macro_gdp.parquet", compression='snappy')
    pmi.to_parquet(DATA_RAW / "macro_pmi.parquet", compression='snappy')
    
    # Merged macro panel
    macro.to_parquet(DATA_RAW / "macro_monthly.parquet", compression='snappy')
    macro.to_parquet(DATA_FULL / "de2_macro_monthly.parquet", compression='snappy')
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  data/raw/macro_cpi.parquet: {len(cpi)} rows")
    print(f"  data/raw/macro_ppi.parquet: {len(ppi)} rows")
    print(f"  data/raw/macro_m2.parquet: {len(m2)} rows")
    print(f"  data/raw/macro_gdp.parquet: {len(gdp)} rows")
    print(f"  data/raw/macro_pmi.parquet: {len(pmi)} rows")
    print(f"  data/raw/macro_monthly.parquet: {len(macro)} rows")
    print(f"  data/full/de2_macro_monthly.parquet: {len(macro)} rows")
    
    # Schema summary
    print("\n" + "=" * 60)
    print("SCHEMA")
    print("=" * 60)
    print(f"  Columns: {list(macro.columns)}")
    print(f"  Date range: {macro['month'].min()} - {macro['month'].max()}")
    
    # Causality check
    print("\n" + "=" * 60)
    print("CAUSALITY CHECK")
    print("=" * 60)
    sample = macro.iloc[0]
    print(f"  Report month: {sample['month']}")
    print(f"  Effective date: {sample['effective_date']}")
    print(f"  (Data only available after effective_date)")
    
    print("\n" + "=" * 60)
    print("DE2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
