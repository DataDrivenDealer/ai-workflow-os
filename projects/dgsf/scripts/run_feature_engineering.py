#!/usr/bin/env python3
"""
Feature Engineering Pipeline for DGSF SDF Layer

This script orchestrates the feature engineering pipeline for the SDF (Stochastic Discount Factor) model.
It computes firm characteristics, cross-sectional spreads, and factors according to SDF_SPEC v3.1.

Usage:
    python run_feature_engineering.py --config config.yaml --dry-run
    python run_feature_engineering.py --config config.yaml --start-date 2020-01-01 --end-date 2023-12-31

Author: DGSF Research Team
Version: 0.1.0
Date: 2026-02-03
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


# ============================================================================
# Configuration Schema Validation
# ============================================================================

REQUIRED_CONFIG_KEYS = [
    'data_sources',
    'output_dir',
    'feature_settings'
]

REQUIRED_DATA_SOURCES = [
    'price_data',
    'shares_outstanding',
    'financial_statements',
    'monthly_returns',
    'risk_free_rate'
]


def validate_config(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Validate configuration file against schema.
    
    Args:
        config: Parsed YAML configuration
        dry_run: If True, skip path existence checks
        
    Returns:
        True if valid, raises ValueError otherwise
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level keys
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check data sources
    data_sources = config.get('data_sources', {})
    for source in REQUIRED_DATA_SOURCES:
        if source not in data_sources:
            raise ValueError(f"Missing required data source: {source}")
        
        # Validate path exists (skip in dry-run mode)
        if not dry_run:
            source_path = data_sources[source].get('path')
            if source_path and not Path(source_path).exists():
                raise ValueError(f"Data source path does not exist: {source_path}")
    
    # Check output directory
    output_dir = config.get('output_dir')
    if not output_dir:
        raise ValueError("output_dir cannot be empty")
    
    return True


def validate_date_format(date_str: str) -> datetime:
    """
    Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


# ============================================================================
# Execution Plan (from SDF_FEATURE_DEFINITIONS.md)
# ============================================================================

EXECUTION_STEPS = [
    {
        'step': 1,
        'name': 'Load Raw Data',
        'parallel': True,
        'substeps': [
            'load_price_data(start_date, end_date) → price[firm, t]',
            'load_shares_outstanding(start_date, end_date) → shares[firm, t]',
            'load_financial_statements(start_date, end_date) → financials[firm, t]',
            'load_monthly_returns(start_date, end_date) → returns[firm, t]',
            'load_risk_free_rate(start_date, end_date) → risk_free[t]'
        ]
    },
    {
        'step': 2,
        'name': 'Compute Independent Firm Characteristics',
        'parallel': True,
        'substeps': [
            'compute_size(price, shares) → size[firm, t]',
            'compute_momentum(returns) → momentum[firm, t]',
            'compute_profitability(financials) → profitability[firm, t]',
            'compute_volatility(returns) → volatility[firm, t]'
        ]
    },
    {
        'step': 3,
        'name': 'Compute Dependent Firm Characteristics',
        'parallel': False,
        'substeps': [
            'compute_book_to_market(financials, size) → book_to_market[firm, t]'
        ]
    },
    {
        'step': 4,
        'name': 'Compute Cross-Sectional Spreads',
        'parallel': False,
        'substeps': [
            'compute_style_spreads(size, book_to_market, momentum, profitability, volatility) → style_spreads[t, 5]',
            'compute_market_factor(returns, risk_free) → market_factor[t] (independent)'
        ]
    },
    {
        'step': 5,
        'name': 'Compute Factors',
        'parallel': True,
        'substeps': [
            'compute_smb_hml(size, book_to_market, returns) → SMB[t], HML[t] (shared 2×3 sorts)',
            'compute_momentum_factor(momentum, returns) → momentum_factor[t]',
            'compute_reversal(returns) → reversal[t]'
        ]
    },
    {
        'step': 6,
        'name': 'Assemble SDF Inputs',
        'parallel': False,
        'substeps': [
            'assemble_X_state(macro, microstructure, style_spreads, leaf_embeddings, market_structure, time_encoding) → X_state[t, d]',
            'assemble_P_tree_factors([market_factor, SMB, HML, momentum_factor, reversal]) → P_tree_factors[t, 5] (OPTIONAL)'
        ]
    },
    {
        'step': 7,
        'name': 'Save Outputs',
        'parallel': False,
        'substeps': [
            'save_X_state(X_state, output_dir)',
            'save_P_tree_factors(P_tree_factors, output_dir) (if enabled)',
            'save_intermediate_features(size, book_to_market, ..., output_dir) (if debug mode)'
        ]
    }
]


def print_execution_plan(start_date: str, end_date: str, config: Dict[str, Any]) -> None:
    """
    Print the 7-step execution plan in dry-run mode.
    
    Args:
        start_date: Start date for feature computation
        end_date: End date for feature computation
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("DGSF FEATURE ENGINEERING PIPELINE - EXECUTION PLAN")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Compute P-tree Factors: {config.get('feature_settings', {}).get('compute_factors', True)}")
    print(f"  Debug Mode: {config.get('debug', False)}")
    
    print(f"\n{'='*80}")
    print("EXECUTION STEPS (7 steps, aligned with SDF_FEATURE_DEFINITIONS.md)")
    print("="*80)
    
    for step_info in EXECUTION_STEPS:
        step_num = step_info['step']
        step_name = step_info['name']
        is_parallel = step_info['parallel']
        substeps = step_info['substeps']
        
        parallel_indicator = "[PARALLEL]" if is_parallel else "[SEQUENTIAL]"
        print(f"\n{'─'*80}")
        print(f"Step {step_num}: {step_name} {parallel_indicator}")
        print(f"{'─'*80}")
        
        for i, substep in enumerate(substeps, 1):
            print(f"  {step_num}.{i} {substep}")
    
    print(f"\n{'='*80}")
    print("ESTIMATED RESOURCE REQUIREMENTS")
    print("="*80)
    print(f"  Memory: ~4-8 GB (depends on universe size)")
    print(f"  Disk Space: ~1-2 GB (for intermediate + final outputs)")
    print(f"  Computation Time: ~10-30 minutes (depends on date range)")
    print(f"  Parallelization: Step 2 (4-way), Step 5 (3-way)")
    
    print(f"\n{'='*80}")
    print("DRY-RUN COMPLETE - No actual computation performed")
    print("="*80)
    print(f"\nTo execute the pipeline, remove the --dry-run flag:")
    print(f"  python run_feature_engineering.py --config {config.get('_config_path', 'config.yaml')} --start-date {start_date} --end-date {end_date}")
    print()


# ============================================================================
# Main Pipeline Execution
# ============================================================================

def run_pipeline(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    dry_run: bool = False
) -> int:
    """
    Execute the feature engineering pipeline.
    
    Args:
        config: Configuration dictionary
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dry_run: If True, only print execution plan without running
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Validate configuration
        print("Validating configuration...")
        validate_config(config, dry_run=dry_run)
        print("✓ Configuration valid")
        
        # Validate dates
        print("Validating date range...")
        start_dt = validate_date_format(start_date)
        end_dt = validate_date_format(end_date)
        
        if start_dt >= end_dt:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
        
        print(f"✓ Date range valid: {start_date} to {end_date}")
        
        # Dry-run mode: print execution plan and exit
        if dry_run:
            print_execution_plan(start_date, end_date, config)
            return 0
        
        # Step 1: Load Raw Data (implemented in T3.3.2)
        from data_loaders import load_all_data
        
        print("\n" + "="*80)
        print("Step 1: Loading Raw Data")
        print("="*80)
        data = load_all_data(start_date, end_date, config)
        print(f"✓ All 5 data sources loaded successfully\n")
        
        # Steps 2-7: TODO in T3.3.3-T3.3.4
        print("="*80)
        print("PIPELINE EXECUTION PAUSED")
        print("="*80)
        print("\nData loading (Step 1) complete. Next steps:")
        print("  - T3.3.3: Firm characteristics computation (Step 2-3)")
        print("  - T3.3.4: Spreads + Factors computation (Step 4-6)")
        print("\nData loaded:")
        for source_name, df in data.items():
            print(f"  • {source_name}: {len(df)} rows")
        print("="*80)
        
        return 0
        
    except ValueError as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='DGSF Feature Engineering Pipeline for SDF Layer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dry-run to see execution plan
  python run_feature_engineering.py --config config.yaml --dry-run
  
  # Execute pipeline for specific date range
  python run_feature_engineering.py --config config.yaml --start-date 2020-01-01 --end-date 2023-12-31
  
  # Execute with custom output directory
  python run_feature_engineering.py --config config.yaml --output-dir /path/to/output --start-date 2020-01-01 --end-date 2023-12-31

For more information, see:
  - docs/SDF_FEATURE_DEFINITIONS.md (feature specifications)
  - tasks/active/SDF_FEATURE_ENG_001.md (task card)
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (required)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for feature computation (YYYY-MM-DD, default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date for feature computation (YYYY-MM-DD, default: 2023-12-31)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print execution plan without running pipeline'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='DGSF Feature Engineering Pipeline v0.1.0'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is not valid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Store config path for dry-run output
    config['_config_path'] = config_path
    
    return config


def main() -> int:
    """
    Main entry point for the feature engineering pipeline.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config['output_dir'] = args.output_dir
            print(f"Output directory overridden: {args.output_dir}")
        
        # Run pipeline
        return run_pipeline(
            config=config,
            start_date=args.start_date,
            end_date=args.end_date,
            dry_run=args.dry_run
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"\n❌ YAML PARSE ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT


if __name__ == '__main__':
    sys.exit(main())
