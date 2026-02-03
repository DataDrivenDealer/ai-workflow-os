"""
Unit tests for data_loaders.py

Tests cover:
1. Date range validation
2. Month-end alignment
3. Column validation
4. Missing value handling
5. Data quality checks

Test Strategy: Use mock CSV data to test logic without requiring actual data files.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from data_loaders import (
    load_price_data,
    load_shares_outstanding,
    load_financial_statements,
    load_monthly_returns,
    load_risk_free_rate,
    load_all_data,
    DataValidationError,
    _validate_date_range,
    _align_to_month_end,
    _validate_required_columns
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config(temp_dir):
    """Create mock configuration with test data paths"""
    return {
        'data_sources': {
            'price_data': {
                'path': os.path.join(temp_dir, 'prices.csv'),
                'columns': {
                    'date': 'trade_date',
                    'firm_id': 'stock_code',
                    'price': 'close_adj'
                }
            },
            'shares_outstanding': {
                'path': os.path.join(temp_dir, 'shares.csv'),
                'columns': {
                    'date': 'report_date',
                    'firm_id': 'stock_code',
                    'shares': 'shares_outstanding'
                }
            },
            'financial_statements': {
                'path': os.path.join(temp_dir, 'financials.csv'),
                'columns': {
                    'date': 'report_date',
                    'firm_id': 'stock_code',
                    'total_assets': 'total_assets',
                    'total_liabilities': 'total_liabilities',
                    'stockholders_equity': 'stockholders_equity',
                    'operating_income': 'operating_income',
                    'net_income': 'net_income'
                }
            },
            'monthly_returns': {
                'path': os.path.join(temp_dir, 'returns.csv'),
                'columns': {
                    'date': 'month_end',
                    'firm_id': 'stock_code',
                    'return': 'monthly_return'
                }
            },
            'risk_free_rate': {
                'path': os.path.join(temp_dir, 'rf_rate.csv'),
                'columns': {
                    'date': 'month_end',
                    'rate': 'tbill_rate'
                }
            }
        }
    }


@pytest.fixture
def sample_price_data(temp_dir):
    """Create sample price data CSV"""
    data = pd.DataFrame({
        'trade_date': ['2020-01-15', '2020-02-20', '2020-03-10', '2020-04-25'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'close_adj': [100.0, 105.5, 200.0, 210.0]
    })
    path = os.path.join(temp_dir, 'prices.csv')
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_shares_data(temp_dir):
    """Create sample shares outstanding CSV"""
    data = pd.DataFrame({
        'report_date': ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'shares_outstanding': [1000.0, 1010.0, 2000.0, 2020.0]
    })
    path = os.path.join(temp_dir, 'shares.csv')
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_financial_data(temp_dir):
    """Create sample financial statements CSV"""
    data = pd.DataFrame({
        'report_date': ['2019-12-31', '2020-03-31', '2020-06-30'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT'],
        'total_assets': [1000.0, 1100.0, 2000.0],
        'total_liabilities': [400.0, 420.0, 800.0],
        'stockholders_equity': [600.0, 680.0, 1200.0],
        'operating_income': [50.0, 55.0, 100.0],
        'net_income': [40.0, 44.0, 80.0]
    })
    path = os.path.join(temp_dir, 'financials.csv')
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_returns_data(temp_dir):
    """Create sample monthly returns CSV"""
    data = pd.DataFrame({
        'month_end': ['2019-12-31', '2020-01-31', '2020-02-29', '2020-03-31'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'monthly_return': [0.05, 0.03, -0.02, 0.10]
    })
    path = os.path.join(temp_dir, 'returns.csv')
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_rf_rate_data(temp_dir):
    """Create sample risk-free rate CSV"""
    data = pd.DataFrame({
        'month_end': ['2019-12-31', '2020-01-31', '2020-02-29', '2020-03-31'],
        'tbill_rate': [0.002, 0.0018, 0.0015, 0.0012]
    })
    path = os.path.join(temp_dir, 'rf_rate.csv')
    data.to_csv(path, index=False)
    return path


# ============================================================================
# Test Helper Functions
# ============================================================================

def test_validate_date_range_valid():
    """Test valid date range validation"""
    start_dt, end_dt = _validate_date_range('2020-01-01', '2020-12-31')
    assert start_dt < end_dt
    assert start_dt == pd.Timestamp('2020-01-01')
    assert end_dt == pd.Timestamp('2020-12-31')


def test_validate_date_range_invalid_format():
    """Test invalid date format raises ValueError"""
    with pytest.raises(ValueError, match="Invalid date format"):
        _validate_date_range('invalid-date', '2020-12-31')


def test_validate_date_range_start_after_end():
    """Test start date after end date raises ValueError"""
    with pytest.raises(ValueError, match="must be before"):
        _validate_date_range('2020-12-31', '2020-01-01')


def test_align_to_month_end():
    """Test month-end alignment"""
    df = pd.DataFrame({
        'date': ['2020-01-15', '2020-02-20', '2020-03-10'],
        'value': [1, 2, 3]
    })
    result = _align_to_month_end(df, 'date')
    
    # Check all dates are month-end
    assert all(result['date'].dt.is_month_end)
    assert result['date'].tolist() == [
        pd.Timestamp('2020-01-31'),
        pd.Timestamp('2020-02-29'),
        pd.Timestamp('2020-03-31')
    ]


def test_validate_required_columns_success():
    """Test column validation succeeds with all columns present"""
    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    _validate_required_columns(df, ['a', 'b', 'c'], 'test_source')


def test_validate_required_columns_missing():
    """Test column validation fails with missing columns"""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(DataValidationError, match="Missing required columns: \\['c'\\]"):
        _validate_required_columns(df, ['a', 'b', 'c'], 'test_source')


# ============================================================================
# Test Data Loaders
# ============================================================================

def test_load_price_data_success(mock_config, sample_price_data):
    """Test successful price data loading"""
    df = load_price_data('2020-01-01', '2020-12-31', mock_config)
    
    # Check columns
    assert list(df.columns) == ['date', 'firm_id', 'price']
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert pd.api.types.is_string_dtype(df['firm_id']) or df['firm_id'].dtype == object
    assert df['price'].dtype == float
    
    # Check date range
    assert df['date'].min() >= pd.Timestamp('2020-01-01')
    assert df['date'].max() <= pd.Timestamp('2020-12-31')
    
    # Check month-end alignment
    assert all(df['date'].dt.is_month_end)


def test_load_price_data_file_not_found(mock_config):
    """Test FileNotFoundError when data file missing"""
    with pytest.raises(FileNotFoundError, match="Price data file not found"):
        load_price_data('2020-01-01', '2020-12-31', mock_config)


def test_load_shares_outstanding_success(mock_config, sample_shares_data):
    """Test successful shares outstanding loading"""
    df = load_shares_outstanding('2020-01-01', '2020-12-31', mock_config)
    
    # Check columns
    assert list(df.columns) == ['date', 'firm_id', 'shares']
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert df['shares'].dtype == float
    
    # Check no negative/zero shares
    assert (df['shares'] > 0).all()


def test_load_financial_statements_success(mock_config, sample_financial_data):
    """Test successful financial statements loading"""
    df = load_financial_statements('2020-01-01', '2020-12-31', mock_config)
    
    # Check columns
    expected_cols = ['date', 'firm_id', 'total_assets', 'total_liabilities',
                     'stockholders_equity', 'operating_income', 'net_income']
    assert list(df.columns) == expected_cols
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    for col in ['total_assets', 'total_liabilities', 'stockholders_equity']:
        assert df[col].dtype == float


def test_load_financial_statements_extended_range(mock_config, sample_financial_data):
    """Test financial statements load with 90-day lag extension"""
    # Request 2020-01-01 to 2020-12-31, should include 2019-12-31 data
    df = load_financial_statements('2020-01-01', '2020-12-31', mock_config)
    
    # Should include data from late 2019 (within 90-day lag)
    assert df['date'].min() <= pd.Timestamp('2020-01-31')


def test_load_monthly_returns_success(mock_config, sample_returns_data):
    """Test successful monthly returns loading"""
    df = load_monthly_returns('2020-01-01', '2020-12-31', mock_config)
    
    # Check columns
    assert list(df.columns) == ['date', 'firm_id', 'return']
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert df['return'].dtype == float


def test_load_monthly_returns_extended_range(mock_config, sample_returns_data):
    """Test monthly returns load with 12-month extension for momentum"""
    # Request 2020-01-01 to 2020-12-31, should include 2019 data
    df = load_monthly_returns('2020-01-01', '2020-12-31', mock_config)
    
    # Should include data from 2019 (for momentum computation)
    assert df['date'].min() <= pd.Timestamp('2020-01-31')


def test_load_risk_free_rate_success(mock_config, sample_rf_rate_data):
    """Test successful risk-free rate loading"""
    df = load_risk_free_rate('2020-01-01', '2020-12-31', mock_config)
    
    # Check columns
    assert list(df.columns) == ['date', 'risk_free_rate']
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert df['risk_free_rate'].dtype == float


def test_load_all_data_success(mock_config, sample_price_data, sample_shares_data,
                                sample_financial_data, sample_returns_data, 
                                sample_rf_rate_data):
    """Test loading all data sources together"""
    data = load_all_data('2020-01-01', '2020-12-31', mock_config)
    
    # Check all 5 sources loaded
    assert len(data) == 5
    assert 'price_data' in data
    assert 'shares_outstanding' in data
    assert 'financial_statements' in data
    assert 'monthly_returns' in data
    assert 'risk_free_rate' in data
    
    # Check each is a DataFrame
    for source_name, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ============================================================================
# Test Data Quality Validation
# ============================================================================

def test_price_data_removes_invalid_prices(mock_config, temp_dir):
    """Test that negative/zero prices are filtered out"""
    # Create data with invalid prices
    data = pd.DataFrame({
        'trade_date': ['2020-01-15', '2020-02-20', '2020-03-10'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT'],
        'close_adj': [100.0, 0.0, -10.0]  # Invalid prices
    })
    path = os.path.join(temp_dir, 'prices.csv')
    data.to_csv(path, index=False)
    
    df = load_price_data('2020-01-01', '2020-12-31', mock_config)
    
    # Should only keep valid prices
    assert len(df) == 1
    assert (df['price'] > 0).all()


def test_shares_outstanding_removes_invalid_shares(mock_config, temp_dir):
    """Test that negative/zero shares are filtered out"""
    # Create data with invalid shares
    data = pd.DataFrame({
        'report_date': ['2020-01-31', '2020-02-29', '2020-03-31'],
        'stock_code': ['AAPL', 'AAPL', 'MSFT'],
        'shares_outstanding': [1000.0, 0.0, -100.0]  # Invalid shares
    })
    path = os.path.join(temp_dir, 'shares.csv')
    data.to_csv(path, index=False)
    
    df = load_shares_outstanding('2020-01-01', '2020-12-31', mock_config)
    
    # Should only keep valid shares
    assert len(df) == 1
    assert (df['shares'] > 0).all()


def test_price_data_handles_missing_values(mock_config, temp_dir, capsys):
    """Test warning for missing price values"""
    # Create data with NaN
    data = pd.DataFrame({
        'trade_date': ['2020-01-15', '2020-02-20'],
        'stock_code': ['AAPL', 'AAPL'],
        'close_adj': [100.0, None]
    })
    path = os.path.join(temp_dir, 'prices.csv')
    data.to_csv(path, index=False)
    
    df = load_price_data('2020-01-01', '2020-12-31', mock_config)
    
    # Check warning was printed
    captured = capsys.readouterr()
    assert "missing values" in captured.out


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_date_filtering_works(mock_config, sample_price_data):
    """Test date range filtering excludes out-of-range dates"""
    # Load only Q1 2020
    df = load_price_data('2020-02-01', '2020-03-31', mock_config)
    
    # Should only include Feb-Mar data
    assert df['date'].min() >= pd.Timestamp('2020-02-01')
    assert df['date'].max() <= pd.Timestamp('2020-03-31')


def test_empty_result_after_filtering(mock_config, sample_price_data):
    """Test behavior when date filter results in empty DataFrame"""
    # Request future dates
    df = load_price_data('2025-01-01', '2025-12-31', mock_config)
    
    # Should return empty DataFrame
    assert len(df) == 0
    assert list(df.columns) == ['date', 'firm_id', 'price']


def test_column_mapping_works(mock_config, temp_dir):
    """Test that column renaming works correctly"""
    # Create data with different column names
    data = pd.DataFrame({
        'custom_date_col': ['2020-01-15'],
        'custom_id_col': ['AAPL'],
        'custom_price_col': [100.0]
    })
    path = os.path.join(temp_dir, 'prices.csv')
    data.to_csv(path, index=False)
    
    # Update config with custom column names
    custom_config = mock_config.copy()
    custom_config['data_sources']['price_data']['columns'] = {
        'date': 'custom_date_col',
        'firm_id': 'custom_id_col',
        'price': 'custom_price_col'
    }
    
    df = load_price_data('2020-01-01', '2020-12-31', custom_config)
    
    # Should have standardized column names
    assert list(df.columns) == ['date', 'firm_id', 'price']
    assert df['firm_id'].iloc[0] == 'AAPL'
    assert df['price'].iloc[0] == 100.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
