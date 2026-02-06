#!/usr/bin/env python3
"""
DGSF Data Fetcher - Fetch OHLCV data from exchanges

Task: DATA_2_DGSF_001
Created by: Data Engineer (王数据)
Date: 2026-02-01

Usage:
    python fetch_data.py --symbol BTC/USDT --start 2020-01-01 --end 2025-12-31
    python fetch_data.py --all  # Fetch all configured symbols
"""

import argparse
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

try:
    import ccxt
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install ccxt pandas pyarrow")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ohlcv"
CHECKSUM_FILE = DATA_DIR / "checksums.yaml"


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """Initialize exchange connection."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        exchange_id: Exchange identifier
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    logger.info(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}")
    
    exchange = get_exchange(exchange_id)
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_ohlcv = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000  # Most exchanges limit to 1000 candles per request
            )
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1  # Next millisecond after last candle
            
            logger.debug(f"Fetched {len(ohlcv)} candles, total: {len(all_ohlcv)}")
            
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit exceeded, waiting...")
            import time
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['symbol'] = symbol
    
    # Reorder columns
    df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp', 'symbol'])
    
    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h"
) -> Path:
    """
    Save data to parquet file and update checksums.
    
    Args:
        df: DataFrame to save
        symbol: Symbol name for filename
        timeframe: Timeframe for filename
        
    Returns:
        Path to saved file
    """
    # Ensure directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    symbol_clean = symbol.replace("/", "_").lower()
    filename = f"{symbol_clean}_{timeframe}.parquet"
    filepath = RAW_DIR / filename
    
    # Save to parquet
    df.to_parquet(filepath, index=False, engine='pyarrow')
    logger.info(f"Saved data to {filepath}")
    
    # Update checksums
    update_checksums(filepath, df)
    
    return filepath


def update_checksums(filepath: Path, df: pd.DataFrame) -> None:
    """Update checksums.yaml with new file entry."""
    # Load existing checksums
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE, 'r') as f:
            checksums = yaml.safe_load(f) or {}
    else:
        checksums = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'generated_by': 'Data Engineer',
            'checksums': {}
        }
    
    # Calculate checksum
    checksum = calculate_checksum(filepath)
    
    # Get relative path
    rel_path = str(filepath.relative_to(DATA_DIR))
    
    # Get date range from data
    date_min = df['timestamp'].min().strftime('%Y-%m-%d')
    date_max = df['timestamp'].max().strftime('%Y-%m-%d')
    
    # Update entry
    checksums['checksums'][rel_path] = {
        'sha256': checksum,
        'size_bytes': filepath.stat().st_size,
        'created_at': datetime.now().isoformat(),
        'rows': len(df),
        'date_range': [date_min, date_max]
    }
    checksums['generated_at'] = datetime.now().isoformat()
    
    # Save checksums
    with open(CHECKSUM_FILE, 'w') as f:
        yaml.dump(checksums, f, default_flow_style=False)
    
    logger.info(f"Updated checksums for {rel_path}")


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame schema against expected schema.
    
    G1 Gate Check: Schema Validation
    """
    expected_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    expected_types = {
        'timestamp': 'datetime64[ns, UTC]',
        'symbol': 'object',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64'
    }
    
    # Check columns
    if list(df.columns) != expected_columns:
        logger.error(f"Schema mismatch: expected {expected_columns}, got {list(df.columns)}")
        return False
    
    # Check types
    for col, expected_type in expected_types.items():
        actual_type = str(df[col].dtype)
        if actual_type != expected_type:
            logger.warning(f"Type mismatch for {col}: expected {expected_type}, got {actual_type}")
    
    logger.info("Schema validation passed")
    return True


def check_missing_rate(df: pd.DataFrame, threshold: float = 0.05) -> bool:
    """
    Check missing data rate.
    
    G1 Gate Check: Missing Rate < 5%
    """
    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
    
    if missing_rate > threshold:
        logger.error(f"Missing rate {missing_rate:.2%} exceeds threshold {threshold:.2%}")
        return False
    
    logger.info(f"Missing rate check passed: {missing_rate:.2%}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for DGSF project")
    parser.add_argument("--symbol", type=str, help="Trading pair (e.g., BTC/USDT)")
    parser.add_argument("--all", action="store_true", help="Fetch all configured symbols")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID")
    parser.add_argument("--dry-run", action="store_true", help="Don't save data")
    
    args = parser.parse_args()
    
    # Default symbols from RESEARCH_1 requirements
    symbols = ["BTC/USDT", "ETH/USDT"] if args.all else [args.symbol]
    
    if not args.all and not args.symbol:
        parser.error("Either --symbol or --all is required")
    
    results = []
    for symbol in symbols:
        try:
            df = fetch_ohlcv(
                symbol=symbol,
                timeframe=args.timeframe,
                start_date=args.start,
                end_date=args.end,
                exchange_id=args.exchange
            )
            
            # Validate
            schema_ok = validate_schema(df)
            missing_ok = check_missing_rate(df)
            
            if not args.dry_run:
                filepath = save_data(df, symbol, args.timeframe)
                results.append({
                    'symbol': symbol,
                    'rows': len(df),
                    'filepath': str(filepath),
                    'schema_valid': schema_ok,
                    'missing_ok': missing_ok
                })
            else:
                logger.info(f"Dry run: would save {len(df)} rows for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            results.append({'symbol': symbol, 'error': str(e)})
    
    # Print summary
    print("\n" + "="*60)
    print("FETCH SUMMARY")
    print("="*60)
    for r in results:
        if 'error' in r:
            print(f"❌ {r['symbol']}: {r['error']}")
        else:
            status = "✅" if r.get('schema_valid') and r.get('missing_ok') else "⚠️"
            print(f"{status} {r['symbol']}: {r['rows']} rows -> {r.get('filepath', 'not saved')}")


if __name__ == "__main__":
    main()
