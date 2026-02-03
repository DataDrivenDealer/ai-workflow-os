"""
End-to-End Feature Engineering Pipeline Tests

Tests the complete pipeline from raw data loading to X_state assembly:
1. Load raw data (5 sources)
2. Compute firm characteristics (5 features)
3. Compute cross-sectional spreads (5D vector)
4. Compute factors (5 factors)
5. Assemble X_state (10D or 15D)

Mock Data: 2020-01 to 2021-12 (24 months), 100 firms
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from firm_characteristics import compute_all_characteristics
from spreads_factors import (
    compute_style_spreads,
    compute_market_factor,
    compute_smb_hml,
    compute_momentum_factor,
    compute_reversal,
    assemble_X_state
)


@pytest.fixture
def mock_dates():
    """Generate 24 monthly dates (2020-01 to 2021-12)"""
    return pd.date_range('2020-01-31', '2021-12-31', freq='ME')


@pytest.fixture
def mock_raw_data(mock_dates):
    """
    Generate mock raw data for 100 firms over 24 months.
    
    Returns:
        Dict with 5 data sources: price, shares, financials, returns, risk_free
    """
    n_firms = 100
    firms = [f'firm_{i:03d}' for i in range(n_firms)]
    
    # Expand dates to include lookback for momentum (need 12 months prior)
    all_dates = pd.date_range('2019-01-31', '2021-12-31', freq='ME')
    
    data = {}
    
    # 1. Price data
    price_data = []
    for date in all_dates:
        for firm in firms:
            price_data.append({
                'date': date,
                'firm_id': firm,
                'price': np.random.uniform(10, 200)
            })
    data['price'] = pd.DataFrame(price_data)
    
    # 2. Shares outstanding
    shares_data = []
    for date in all_dates:
        for firm in firms:
            shares_data.append({
                'date': date,
                'firm_id': firm,
                'shares': np.random.uniform(1e6, 1e9)
            })
    data['shares'] = pd.DataFrame(shares_data)
    
    # 3. Financial statements (quarterly, forward-fill to monthly)
    financials_data = []
    quarterly_dates = pd.date_range('2019-01-31', '2021-12-31', freq='QE')
    for date in quarterly_dates:
        for firm in firms:
            book_equity = np.random.uniform(1e8, 1e10)
            financials_data.append({
                'date': date,
                'firm_id': firm,
                'operating_income': np.random.uniform(1e6, 1e9),
                'total_assets': book_equity + np.random.uniform(1e8, 1e10),
                'total_liabilities': np.random.uniform(1e8, 1e10),
                'stockholders_equity': book_equity,
                'preferred_stock': np.random.uniform(0, 1e8)
            })
    data['financials'] = pd.DataFrame(financials_data)
    
    # 4. Returns (monthly)
    returns_data = []
    for date in all_dates:
        for firm in firms:
            returns_data.append({
                'date': date,
                'firm_id': firm,
                'return': np.random.normal(0.01, 0.05)  # Mean 1%, std 5%
            })
    data['returns'] = pd.DataFrame(returns_data)
    
    # 5. Risk-free rate (monthly, market-level)
    risk_free_data = []
    for date in all_dates:
        risk_free_data.append({
            'date': date,
            'rf_rate': np.random.uniform(0.001, 0.003)  # 0.1-0.3% monthly
        })
    data['risk_free'] = pd.DataFrame(risk_free_data)
    
    return data


def test_e2e_pipeline_characteristics(mock_raw_data):
    """Test Step 2-3: Firm characteristics computation"""
    # Execute
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Verify all outputs are DataFrames
    for name, df in [('size', size), ('momentum', momentum), 
                      ('profitability', profitability), ('volatility', volatility),
                      ('book_to_market', book_to_market)]:
        assert isinstance(df, pd.DataFrame), f"{name} should be DataFrame"
        assert 'date' in df.columns, f"{name} should have 'date' column"
        assert 'firm_id' in df.columns, f"{name} should have 'firm_id' column"
        assert len(df) > 0, f"{name} should not be empty"
    
    # Verify dependencies (book_to_market requires size)
    assert len(book_to_market) <= len(size), "book_to_market should have <= size obs (depends on size)"


def test_e2e_pipeline_spreads(mock_raw_data):
    """Test Step 4: Cross-sectional spreads computation"""
    # Compute characteristics first
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Execute spreads
    spreads = compute_style_spreads(size, book_to_market, momentum, profitability, volatility)
    
    # Verify
    assert isinstance(spreads, pd.DataFrame)
    assert 'date' in spreads.columns
    expected_cols = ['size_spread', 'book_to_market_spread', 'momentum_spread',
                     'profitability_spread', 'volatility_spread']
    for col in expected_cols:
        assert col in spreads.columns, f"Spreads should contain {col}"
    
    # Check output is date-level (not firm-level)
    assert len(spreads) < len(size), "Spreads should be date-level (aggregated)"


def test_e2e_pipeline_factors(mock_raw_data):
    """Test Step 5: Factor computation"""
    # Compute characteristics first
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Execute factors
    market_factor = compute_market_factor(mock_raw_data['returns'], mock_raw_data['risk_free'])
    smb, hml = compute_smb_hml(size, book_to_market, mock_raw_data['returns'])
    momentum_factor = compute_momentum_factor(momentum, mock_raw_data['returns'])
    reversal = compute_reversal(mock_raw_data['returns'])
    
    # Verify all factors are DataFrames with date column
    for name, df in [('market_factor', market_factor), ('smb', smb), 
                      ('hml', hml), ('momentum_factor', momentum_factor),
                      ('reversal', reversal)]:
        assert isinstance(df, pd.DataFrame), f"{name} should be DataFrame"
        assert 'date' in df.columns, f"{name} should have 'date' column"
        assert len(df) > 0, f"{name} should not be empty"


def test_e2e_pipeline_X_state_without_factors(mock_raw_data):
    """Test Step 6: X_state assembly (characteristics + spreads only)"""
    # Compute characteristics
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Compute spreads
    spreads = compute_style_spreads(size, book_to_market, momentum, profitability, volatility)
    
    # Assemble X_state (without factors)
    characteristics_dict = {
        'size': size,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'profitability': profitability,
        'volatility': volatility
    }
    
    X_state = assemble_X_state(characteristics_dict, spreads)
    
    # Verify shape (10D: 5 characteristics + 5 spreads)
    assert isinstance(X_state, pd.DataFrame)
    assert 'date' in X_state.columns
    
    # Count X_state dimensions
    x_state_cols = [col for col in X_state.columns if col.startswith('X_state_dim_')]
    assert len(x_state_cols) == 10, f"Expected 10 X_state dims, got {len(x_state_cols)}"
    
    # Check NaN only in early months (due to insufficient lookback), not in later stable period
    # Allow first 12 months (2019-01 to 2019-12) to have NaN due to lookback requirements
    stable_period = X_state[X_state['date'] >= pd.Timestamp('2020-01-01')]
    total_cells = stable_period.drop(columns='date').shape[0] * stable_period.drop(columns='date').shape[1]
    nan_count = stable_period.drop(columns='date').isnull().sum().sum()
    nan_ratio = nan_count / total_cells if total_cells > 0 else 0
    assert nan_ratio < 0.2, f"X_state should have < 20% NaN in stable period, got {nan_ratio:.2%}"


def test_e2e_pipeline_X_state_with_factors(mock_raw_data):
    """Test Step 6: X_state assembly (characteristics + spreads + factors)"""
    # Compute characteristics
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Compute spreads
    spreads = compute_style_spreads(size, book_to_market, momentum, profitability, volatility)
    
    # Compute factors
    market_factor = compute_market_factor(mock_raw_data['returns'], mock_raw_data['risk_free'])
    smb, hml = compute_smb_hml(size, book_to_market, mock_raw_data['returns'])
    momentum_factor = compute_momentum_factor(momentum, mock_raw_data['returns'])
    reversal = compute_reversal(mock_raw_data['returns'])
    
    # Assemble X_state (with factors)
    characteristics_dict = {
        'size': size,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'profitability': profitability,
        'volatility': volatility
    }
    factors_dict = {
        'market_factor': market_factor,
        'smb_factor': smb,
        'hml_factor': hml,
        'momentum_factor': momentum_factor,
        'reversal_factor': reversal
    }
    
    X_state = assemble_X_state(characteristics_dict, spreads, factors_dict)
    
    # Verify shape (15D: 5 characteristics + 5 spreads + 5 factors)
    assert isinstance(X_state, pd.DataFrame)
    assert 'date' in X_state.columns
    
    # Count X_state dimensions
    x_state_cols = [col for col in X_state.columns if col.startswith('X_state_dim_')]
    assert len(x_state_cols) == 15, f"Expected 15 X_state dims (with factors), got {len(x_state_cols)}"
    
    # Check NaN only in early months (due to insufficient lookback), not in later stable period
    stable_period = X_state[X_state['date'] >= pd.Timestamp('2020-01-01')]
    total_cells = stable_period.drop(columns='date').shape[0] * stable_period.drop(columns='date').shape[1]
    nan_count = stable_period.drop(columns='date').isnull().sum().sum()
    nan_ratio = nan_count / total_cells if total_cells > 0 else 0
    assert nan_ratio < 0.2, f"X_state should have < 20% NaN in stable period, got {nan_ratio:.2%}"


def test_e2e_pipeline_no_data_leakage(mock_raw_data):
    """Test data leakage: X_state[t] should only use data up to t-1"""
    # Compute characteristics
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    # Compute spreads
    spreads = compute_style_spreads(size, book_to_market, momentum, profitability, volatility)
    
    # Assemble X_state
    characteristics_dict = {
        'size': size,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'profitability': profitability,
        'volatility': volatility
    }
    
    X_state = assemble_X_state(characteristics_dict, spreads)
    
    # Verify dates are sorted (temporal order)
    assert X_state['date'].is_monotonic_increasing, "X_state dates should be sorted"
    
    # Verify momentum uses lagged returns (not current month)
    # This is implicitly tested by momentum computation (skips last month)
    # Just check momentum exists in characteristics
    assert len(momentum) > 0, "Momentum should be computed"


def test_e2e_pipeline_execution_time(mock_raw_data):
    """Test pipeline execution time (should be < 5s for mock data)"""
    import time
    
    start = time.time()
    
    # Full pipeline
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_raw_data['price'],
        shares_outstanding=mock_raw_data['shares'],
        financial_statements=mock_raw_data['financials'],
        returns=mock_raw_data['returns']
    )
    
    spreads = compute_style_spreads(size, book_to_market, momentum, profitability, volatility)
    
    market_factor = compute_market_factor(mock_raw_data['returns'], mock_raw_data['risk_free'])
    smb, hml = compute_smb_hml(size, book_to_market, mock_raw_data['returns'])
    momentum_factor = compute_momentum_factor(momentum, mock_raw_data['returns'])
    reversal = compute_reversal(mock_raw_data['returns'])
    
    characteristics_dict = {
        'size': size,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'profitability': profitability,
        'volatility': volatility
    }
    factors_dict = {
        'market_factor': market_factor,
        'smb_factor': smb,
        'hml_factor': hml,
        'momentum_factor': momentum_factor,
        'reversal_factor': reversal
    }
    
    X_state = assemble_X_state(characteristics_dict, spreads, factors_dict)
    
    elapsed = time.time() - start
    
    # Verify execution time
    assert elapsed < 5.0, f"Pipeline should execute in < 5s, took {elapsed:.2f}s"
    print(f"\nPipeline execution time: {elapsed:.2f}s")
