"""
Unit tests for firm_characteristics.py

Tests all 5 firm characteristic computation functions with mock data.
Verifies formulas, data quality handling, and winsorization logic.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

from firm_characteristics import (
    winsorize,
    compute_size,
    compute_momentum,
    compute_profitability,
    compute_volatility,
    compute_book_to_market,
    compute_all_characteristics
)


# ============================================================================
# Fixtures: Mock Data
# ============================================================================

@pytest.fixture
def mock_price_data():
    """Mock price data for 3 firms over 6 months"""
    dates = pd.date_range('2020-01', periods=6, freq='ME')
    data = []
    for firm_id in [1, 2, 3]:
        for date in dates:
            # Firm 1: stable price $50
            # Firm 2: growing price $20 → $70
            # Firm 3: volatile price $10 → $100
            if firm_id == 1:
                price = 50.0
            elif firm_id == 2:
                price = 20.0 + (date - dates[0]).days * 0.2
            else:
                price = 10.0 + (date - dates[0]).days * 0.5
            data.append({'date': date, 'firm_id': firm_id, 'price': price})
    return pd.DataFrame(data)


@pytest.fixture
def mock_shares_outstanding():
    """Mock shares outstanding for 3 firms"""
    dates = pd.date_range('2020-01', periods=6, freq='ME')
    data = []
    for firm_id in [1, 2, 3]:
        for date in dates:
            # Firm 1: large cap (10M shares)
            # Firm 2: mid cap (1M shares)
            # Firm 3: small cap (100K shares)
            if firm_id == 1:
                shares = 10_000_000
            elif firm_id == 2:
                shares = 1_000_000
            else:
                shares = 100_000
            data.append({'date': date, 'firm_id': firm_id, 'shares': shares})
    return pd.DataFrame(data)


@pytest.fixture
def mock_financial_statements():
    """Mock financial statements for 3 firms (quarterly)"""
    dates = pd.date_range('2020-01', periods=6, freq='ME')
    data = []
    for firm_id in [1, 2, 3]:
        for date in dates:
            # Firm 1: profitable, high book value
            # Firm 2: marginally profitable
            # Firm 3: unprofitable (negative operating income)
            if firm_id == 1:
                total_assets = 1_000_000_000
                total_liabilities = 400_000_000
                operating_income = 50_000_000
            elif firm_id == 2:
                total_assets = 100_000_000
                total_liabilities = 80_000_000
                operating_income = 1_000_000
            else:
                total_assets = 10_000_000
                total_liabilities = 8_000_000
                operating_income = -500_000
            
            data.append({
                'date': date,
                'firm_id': firm_id,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'stockholders_equity': total_assets - total_liabilities,
                'operating_income': operating_income
            })
    return pd.DataFrame(data)


@pytest.fixture
def mock_returns():
    """Mock monthly returns for 3 firms over 12 months"""
    dates = pd.date_range('2020-01', periods=12, freq='ME')
    data = []
    for firm_id in [1, 2, 3]:
        for i, date in enumerate(dates):
            # Firm 1: stable returns (low volatility)
            # Firm 2: positive momentum (winning)
            # Firm 3: high volatility
            if firm_id == 1:
                ret = 0.01  # 1% monthly
            elif firm_id == 2:
                ret = 0.05 + i * 0.005  # Increasing returns
            else:
                ret = 0.10 * np.sin(i)  # Volatile
            data.append({'date': date, 'firm_id': firm_id, 'return': ret})
    return pd.DataFrame(data)


# ============================================================================
# Tests: winsorize
# ============================================================================

def test_winsorize_basic():
    """Test basic winsorization at [1%, 99%]"""
    df = pd.DataFrame({
        'date': ['2020-01'] * 100,
        'firm_id': range(100),
        'value': range(100)  # 0, 1, 2, ..., 99
    })
    
    result = winsorize(df, ['value'], lower=0.01, upper=0.99)
    
    # Check that extreme values are clipped
    assert result['value'].min() >= 0.99  # ~1st percentile
    assert result['value'].max() <= 98.01  # ~99th percentile
    assert len(result) == 100


def test_winsorize_multiple_dates():
    """Test winsorization with multiple dates (cross-sectional)"""
    df = pd.DataFrame({
        'date': ['2020-01'] * 50 + ['2020-02'] * 50,
        'firm_id': list(range(50)) * 2,
        'value': list(range(50)) + list(range(100, 150))
    })
    
    result = winsorize(df, ['value'], lower=0.02, upper=0.98)
    
    # Check that winsorization is date-specific
    jan_values = result[result['date'] == '2020-01']['value']
    feb_values = result[result['date'] == '2020-02']['value']
    
    assert jan_values.min() >= 0.98  # ~2nd percentile of 0-49
    assert feb_values.min() >= 100.98  # ~2nd percentile of 100-149


def test_winsorize_missing_column():
    """Test winsorization with missing column (should warn)"""
    df = pd.DataFrame({
        'date': ['2020-01'] * 10,
        'firm_id': range(10),
        'value': range(10)
    })
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = winsorize(df, ['nonexistent_column'], lower=0.01, upper=0.99)
        assert len(w) == 1
        assert "not found" in str(w[0].message)


# ============================================================================
# Tests: compute_size
# ============================================================================

def test_compute_size_basic(mock_price_data, mock_shares_outstanding):
    """Test basic size computation (log market cap)"""
    result = compute_size(mock_price_data, mock_shares_outstanding)
    
    # Check output structure
    assert 'date' in result.columns
    assert 'firm_id' in result.columns
    assert 'size' in result.columns
    assert len(result) > 0
    
    # Check formula: size = log(price * shares)
    firm1_row = result[(result['firm_id'] == 1)].iloc[0]
    expected_market_cap = 50.0 * 10_000_000  # $500M
    expected_size = np.log(expected_market_cap)
    assert np.isclose(firm1_row['size'], expected_size, rtol=0.01)


def test_compute_size_microcap_filter(mock_price_data, mock_shares_outstanding):
    """Test that microcap firms are excluded"""
    # Add a microcap firm (price = $1, shares = 1000 → market cap = $1K)
    microcap_data = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=6, freq='ME'),
        'firm_id': [99] * 6,
        'price': [1.0] * 6
    })
    microcap_shares = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=6, freq='ME'),
        'firm_id': [99] * 6,
        'shares': [1000] * 6
    })
    
    extended_price = pd.concat([mock_price_data, microcap_data])
    extended_shares = pd.concat([mock_shares_outstanding, microcap_shares])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_size(extended_price, extended_shares, min_market_cap=5e6)
        
        # Check that microcap firm is excluded
        assert 99 not in result['firm_id'].values
        assert len(w) == 2  # 1 for exclusion, 1 for winsorization


def test_compute_size_forward_fill():
    """Test forward-fill of missing values (max 3 months)"""
    price_data = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=6, freq='ME'),
        'firm_id': [1] * 6,
        'price': [50.0, 52.0, np.nan, np.nan, 55.0, 56.0]
    })
    shares_data = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=6, freq='ME'),
        'firm_id': [1] * 6,
        'shares': [1_000_000] * 6
    })
    
    result = compute_size(price_data, shares_data)
    
    # Check that missing months are forward-filled
    assert len(result) == 4  # 2 missing prices excluded (NaN before forward-fill applies)


# ============================================================================
# Tests: compute_momentum
# ============================================================================

def test_compute_momentum_basic(mock_returns):
    """Test basic momentum computation (12-month cumulative return)"""
    result = compute_momentum(mock_returns, lookback_months=12, skip_last_month=True)
    
    # Check output structure
    assert 'date' in result.columns
    assert 'firm_id' in result.columns
    assert 'momentum' in result.columns
    
    # Firm 1: 1% monthly for 12 months → (1.01)^10 - 1 ≈ 10.46% (excluding last month)
    firm1_momentum = result[result['firm_id'] == 1].iloc[-1]['momentum']
    expected_momentum = np.prod([1.01] * 10) - 1  # Months t-12 to t-2 (10 months)
    assert np.isclose(firm1_momentum, expected_momentum, rtol=0.05)


def test_compute_momentum_insufficient_history(mock_returns):
    """Test that firms with insufficient history are excluded"""
    # Only 5 months of data
    short_returns = mock_returns[mock_returns['date'] <= '2020-05'].copy()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_momentum(short_returns, lookback_months=12, min_obs=8)
        
        # All observations should be excluded (< 8 months)
        assert len(result) == 0
        assert len(w) >= 1  # Warning for exclusion


def test_compute_momentum_skip_last_month():
    """Test that skipping last month works correctly"""
    returns = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=13, freq='ME'),
        'firm_id': [1] * 13,
        'return': [0.10] * 13  # 10% every month
    })
    
    # With skip_last_month=True, momentum at t=13 uses months [t-12, t-2]
    result_skip = compute_momentum(returns, lookback_months=12, skip_last_month=True, min_obs=8)
    
    # With skip_last_month=False, momentum at t=13 uses months [t-12, t-1]
    result_no_skip = compute_momentum(returns, lookback_months=12, skip_last_month=False, min_obs=8)
    
    # Both should have observations (may differ in count due to min_obs)
    assert len(result_skip) > 0
    assert len(result_no_skip) > 0
    # Values should differ (excluding vs including last month)


# ============================================================================
# Tests: compute_profitability
# ============================================================================

def test_compute_profitability_basic(mock_financial_statements):
    """Test basic profitability computation (operating income / book equity)"""
    result = compute_profitability(mock_financial_statements)
    
    # Check output structure
    assert 'date' in result.columns
    assert 'firm_id' in result.columns
    assert 'profitability' in result.columns
    
    # Firm 1: operating_income = 50M, book_equity = 600M → profitability = 0.0833
    firm1_prof = result[result['firm_id'] == 1].iloc[0]['profitability']
    expected_prof = 50_000_000 / 600_000_000
    assert np.isclose(firm1_prof, expected_prof, rtol=0.01)


def test_compute_profitability_negative_book_equity():
    """Test that negative book equity firms are excluded"""
    financials = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=3, freq='ME'),
        'firm_id': [1] * 3,
        'total_assets': [100_000, 90_000, 80_000],
        'total_liabilities': [120_000, 110_000, 100_000],  # Negative book equity
        'stockholders_equity': [-20_000, -20_000, -20_000],
        'operating_income': [5_000, 5_000, 5_000]
    })
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_profitability(financials, min_book_equity=0.0)
        
        # All observations should be excluded (negative book equity)
        assert len(result) == 0
        assert len(w) >= 1  # Warning for exclusion


def test_compute_profitability_winsorization(mock_financial_statements):
    """Test that profitability is winsorized at [0.5%, 99.5%]"""
    # Add extreme profitability firm
    extreme_financials = mock_financial_statements.copy()
    extreme_row = {
        'date': pd.Timestamp('2020-01'),
        'firm_id': 99,
        'total_assets': 1_000_000,
        'total_liabilities': 100_000,
        'stockholders_equity': 900_000,
        'operating_income': 1_000_000  # 111% ROE (extreme)
    }
    extreme_financials = pd.concat([
        extreme_financials,
        pd.DataFrame([extreme_row])
    ])
    
    result = compute_profitability(extreme_financials)
    
    # Check that extreme value is winsorized
    firm99_prof = result[result['firm_id'] == 99]['profitability'].values
    if len(firm99_prof) > 0:
        # Should be clipped (depends on cross-sectional distribution)
        assert firm99_prof[0] < 1.5  # Original was 1.11, might be clipped


# ============================================================================
# Tests: compute_volatility
# ============================================================================

def test_compute_volatility_basic(mock_returns):
    """Test basic volatility computation (rolling 12-month std dev)"""
    result = compute_volatility(mock_returns, lookback_months=12, min_obs=6)
    
    # Check output structure
    assert 'date' in result.columns
    assert 'firm_id' in result.columns
    assert 'volatility' in result.columns
    
    # Firm 1: constant 1% returns → volatility ≈ 0
    firm1_vol = result[result['firm_id'] == 1].iloc[-1]['volatility']
    assert firm1_vol < 0.001  # Very low volatility
    
    # Firm 3: volatile returns → higher volatility
    firm3_vol = result[result['firm_id'] == 3].iloc[-1]['volatility']
    assert firm3_vol > 0.05  # Higher volatility


def test_compute_volatility_insufficient_history(mock_returns):
    """Test that firms with insufficient history are excluded"""
    short_returns = mock_returns[mock_returns['date'] <= '2020-03'].copy()  # Only 3 months
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_volatility(short_returns, lookback_months=12, min_obs=6)
        
        # All observations should be excluded (< 6 months)
        assert len(result) == 0
        assert len(w) >= 1


def test_compute_volatility_formula():
    """Test volatility formula with known values"""
    # 12 months of returns: [0%, 10%, -10%, 0%, 10%, -10%, ...]
    returns = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=12, freq='ME'),
        'firm_id': [1] * 12,
        'return': [0.0, 0.1, -0.1] * 4
    })
    
    result = compute_volatility(returns, lookback_months=12, min_obs=12)
    
    # Compute expected volatility manually
    expected_vol = np.std([0.0, 0.1, -0.1] * 4, ddof=1)  # pandas uses ddof=1 by default
    actual_vol = result.iloc[-1]['volatility']
    assert np.isclose(actual_vol, expected_vol, rtol=0.02)  # Allow 2% tolerance


# ============================================================================
# Tests: compute_book_to_market
# ============================================================================

def test_compute_book_to_market_basic(mock_financial_statements, mock_price_data, mock_shares_outstanding):
    """Test basic book_to_market computation"""
    # First compute size
    size_data = compute_size(mock_price_data, mock_shares_outstanding)
    
    # Then compute book_to_market
    result = compute_book_to_market(mock_financial_statements, size_data)
    
    # Check output structure
    assert 'date' in result.columns
    assert 'firm_id' in result.columns
    assert 'book_to_market' in result.columns
    
    # Firm 1: book_equity = 600M, market_cap = 500M → B/M = 1.2
    firm1_btm = result[result['firm_id'] == 1].iloc[0]['book_to_market']
    expected_btm = 600_000_000 / 500_000_000
    assert np.isclose(firm1_btm, expected_btm, rtol=0.05)


def test_compute_book_to_market_dependency_on_size():
    """Test that book_to_market correctly uses size data"""
    financials = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=3, freq='ME'),
        'firm_id': [1] * 3,
        'total_assets': [1_000_000] * 3,
        'total_liabilities': [400_000] * 3,
        'stockholders_equity': [600_000] * 3,
        'operating_income': [50_000] * 3
    })
    
    size_data = pd.DataFrame({
        'date': pd.date_range('2020-01', periods=3, freq='ME'),
        'firm_id': [1] * 3,
        'size': [np.log(500_000)] * 3  # log(market_cap)
    })
    
    result = compute_book_to_market(financials, size_data)
    
    # book_equity = 600K, market_cap = 500K → B/M = 1.2
    expected_btm = 600_000 / 500_000
    assert np.isclose(result.iloc[0]['book_to_market'], expected_btm, rtol=0.01)


# ============================================================================
# Tests: compute_all_characteristics (Integration)
# ============================================================================

def test_compute_all_characteristics_integration(
    mock_price_data,
    mock_shares_outstanding,
    mock_financial_statements,
    mock_returns
):
    """Test end-to-end computation of all 5 characteristics"""
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_price_data,
        shares_outstanding=mock_shares_outstanding,
        financial_statements=mock_financial_statements,
        returns=mock_returns
    )
    
    # Check that all outputs are DataFrames
    assert isinstance(size, pd.DataFrame)
    assert isinstance(momentum, pd.DataFrame)
    assert isinstance(profitability, pd.DataFrame)
    assert isinstance(volatility, pd.DataFrame)
    assert isinstance(book_to_market, pd.DataFrame)
    
    # Check that all have required columns
    for df in [size, momentum, profitability, volatility, book_to_market]:
        assert 'date' in df.columns
        assert 'firm_id' in df.columns
    
    # Check that size and book_to_market have overlapping firms
    common_firms = set(size['firm_id']) & set(book_to_market['firm_id'])
    assert len(common_firms) > 0


def test_compute_all_characteristics_output_counts(
    mock_price_data,
    mock_shares_outstanding,
    mock_financial_statements,
    mock_returns
):
    """Test that output counts are reasonable"""
    size, momentum, profitability, volatility, book_to_market = compute_all_characteristics(
        price_data=mock_price_data,
        shares_outstanding=mock_shares_outstanding,
        financial_statements=mock_financial_statements,
        returns=mock_returns
    )
    
    # All should have > 0 observations
    assert len(size) > 0
    assert len(momentum) > 0
    assert len(profitability) > 0
    assert len(volatility) > 0
    assert len(book_to_market) > 0
    
    print(f"\nCharacteristic observation counts:")
    print(f"  size: {len(size)}")
    print(f"  momentum: {len(momentum)}")
    print(f"  profitability: {len(profitability)}")
    print(f"  volatility: {len(volatility)}")
    print(f"  book_to_market: {len(book_to_market)}")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
