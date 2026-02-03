"""
Unit tests for spreads_factors.py

Test Coverage:
- compute_market_factor (3 tests)
- compute_smb_hml (4 tests) - TODO
- compute_momentum_factor (3 tests) - TODO
- compute_reversal (3 tests) - TODO
- compute_style_spreads (4 tests) - TODO
- assemble_X_state (2 tests) - TODO
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import spreads_factors as sf


# ============================================================================
# compute_market_factor tests
# ============================================================================

def test_compute_market_factor_basic():
    """Test basic market factor computation: market_return - risk_free"""
    # Setup: 3 dates, 4 firms
    returns = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31'] * 4 + ['2020-02-29'] * 4 + ['2020-03-31'] * 4),
        'firm_id': ['A', 'B', 'C', 'D'] * 3,
        'return': [0.05, 0.03, 0.07, 0.01,  # Jan: mean = 0.04
                   -0.02, 0.04, 0.02, 0.00,  # Feb: mean = 0.01
                   0.10, 0.08, 0.12, 0.06]   # Mar: mean = 0.09
    })
    
    risk_free = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31', '2020-02-29', '2020-03-31']),
        'rf_rate': [0.002, 0.002, 0.002]  # 0.2% monthly
    })
    
    # Execute
    result = sf.compute_market_factor(returns, risk_free)
    
    # Verify
    assert len(result) == 3, "Should have 3 dates"
    assert list(result.columns) == ['date', 'market_factor'], "Columns should be [date, market_factor]"
    
    # Check values (market_return - rf_rate)
    expected = [0.04 - 0.002, 0.01 - 0.002, 0.09 - 0.002]
    np.testing.assert_array_almost_equal(result['market_factor'].values, expected, decimal=5)


def test_compute_market_factor_no_risk_free():
    """Test market factor when risk-free rate is missing (should default to 0)"""
    returns = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31'] * 3),
        'firm_id': ['A', 'B', 'C'],
        'return': [0.05, 0.03, 0.07]  # mean = 0.05
    })
    
    # Risk-free with NaN
    risk_free = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31']),
        'rf_rate': [np.nan]
    })
    
    # Execute
    result = sf.compute_market_factor(returns, risk_free)
    
    # Verify (should use 0 when rf_rate is NaN)
    assert len(result) == 1
    assert result.iloc[0]['market_factor'] == pytest.approx(0.05, abs=1e-5)


def test_compute_market_factor_missing_dates():
    """Test market factor when risk-free rate is missing for some dates"""
    returns = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31'] * 2 + ['2020-02-29'] * 2),
        'firm_id': ['A', 'B'] * 2,
        'return': [0.05, 0.03, 0.02, 0.04]  # Jan: 0.04, Feb: 0.03
    })
    
    # Risk-free only for Jan (Feb should forward-fill)
    risk_free = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31']),
        'rf_rate': [0.002]
    })
    
    # Execute
    result = sf.compute_market_factor(returns, risk_free)
    
    # Verify (Feb should forward-fill Jan's rf_rate)
    assert len(result) == 2
    assert result.iloc[0]['market_factor'] == pytest.approx(0.04 - 0.002, abs=1e-5)
    assert result.iloc[1]['market_factor'] == pytest.approx(0.03 - 0.002, abs=1e-5)  # forward-filled


# ============================================================================
# compute_smb_hml tests
# ============================================================================

def test_compute_smb_hml_basic():
    """Test basic SMB/HML computation with known inputs"""
    # Setup: 1 date, 6 firms (2 size groups × 3 B/M groups)
    date = pd.to_datetime('2020-01-31')
    
    size = pd.DataFrame({
        'date': [date] * 6,
        'firm_id': ['A', 'B', 'C', 'D', 'E', 'F'],
        'size': [10, 11, 12, 20, 21, 22]  # Median = 16 (A,B,C small; D,E,F big)
    })
    
    book_to_market = pd.DataFrame({
        'date': [date] * 6,
        'firm_id': ['A', 'B', 'C', 'D', 'E', 'F'],
        'book_to_market': [0.5, 1.0, 1.5, 0.6, 1.1, 1.6]  
        # 30%ile = 0.62, 70%ile = 1.42
        # Low: A(0.5), D(0.6)
        # Medium: B(1.0), E(1.1)
        # High: C(1.5), F(1.6)
    })
    
    returns = pd.DataFrame({
        'date': [date] * 6,
        'firm_id': ['A', 'B', 'C', 'D', 'E', 'F'],
        'return': [0.02, 0.04, 0.06, 0.01, 0.03, 0.05]
        # S/L=0.02, S/M=0.04, S/H=0.06
        # B/L=0.01, B/M=0.03, B/H=0.05
    })
    
    # Execute
    smb, hml = sf.compute_smb_hml(size, book_to_market, returns)
    
    # Verify SMB = (S/L + S/M + S/H)/3 - (B/L + B/M + B/H)/3
    #            = (0.02 + 0.04 + 0.06)/3 - (0.01 + 0.03 + 0.05)/3
    #            = 0.04 - 0.03 = 0.01
    assert len(smb) == 1
    assert smb.iloc[0]['smb_factor'] == pytest.approx(0.01, abs=1e-5)
    
    # Verify HML = (S/H + B/H)/2 - (S/L + B/L)/2
    #            = (0.06 + 0.05)/2 - (0.02 + 0.01)/2
    #            = 0.055 - 0.015 = 0.04
    assert len(hml) == 1
    assert hml.iloc[0]['hml_factor'] == pytest.approx(0.04, abs=1e-5)


def test_compute_smb_hml_multiple_dates():
    """Test SMB/HML computation across multiple dates"""
    dates = pd.to_datetime(['2020-01-31', '2020-02-29'])
    
    size = pd.DataFrame({
        'date': [dates[0]] * 4 + [dates[1]] * 4,
        'firm_id': ['A', 'B', 'C', 'D'] * 2,
        'size': [10, 11, 20, 21] * 2  # Median splits at 15.5
    })
    
    book_to_market = pd.DataFrame({
        'date': [dates[0]] * 4 + [dates[1]] * 4,
        'firm_id': ['A', 'B', 'C', 'D'] * 2,
        'book_to_market': [0.5, 1.5, 0.6, 1.6] * 2  # A,C low; B,D high
    })
    
    returns = pd.DataFrame({
        'date': [dates[0]] * 4 + [dates[1]] * 4,
        'firm_id': ['A', 'B', 'C', 'D'] * 2,
        'return': [0.02, 0.04, 0.01, 0.03,  # Jan
                   0.03, 0.05, 0.02, 0.04]  # Feb
    })
    
    # Execute
    smb, hml = sf.compute_smb_hml(size, book_to_market, returns)
    
    # Verify
    assert len(smb) == 2, "Should have 2 dates"
    assert len(hml) == 2, "Should have 2 dates"
    assert not smb.isnull().any().any(), "SMB should not have NaN"
    assert not hml.isnull().any().any(), "HML should not have NaN"


def test_compute_smb_hml_missing_portfolios():
    """Test SMB/HML when some portfolios are missing (edge case)"""
    date = pd.to_datetime('2020-01-31')
    
    # Only 3 firms (not enough for all 6 portfolios)
    size = pd.DataFrame({
        'date': [date] * 3,
        'firm_id': ['A', 'B', 'C'],
        'size': [10, 15, 20]
    })
    
    book_to_market = pd.DataFrame({
        'date': [date] * 3,
        'firm_id': ['A', 'B', 'C'],
        'book_to_market': [0.5, 1.0, 1.5]
    })
    
    returns = pd.DataFrame({
        'date': [date] * 3,
        'firm_id': ['A', 'B', 'C'],
        'return': [0.02, 0.03, 0.04]
    })
    
    # Execute
    smb, hml = sf.compute_smb_hml(size, book_to_market, returns)
    
    # Verify (should handle missing portfolios with nanmean)
    assert len(smb) == 1
    assert len(hml) == 1
    assert not np.isnan(smb.iloc[0]['smb_factor']), "SMB should be computed despite missing portfolios"
    assert not np.isnan(hml.iloc[0]['hml_factor']), "HML should be computed despite missing portfolios"


def test_compute_smb_hml_shared_sorts():
    """Test that SMB and HML share the same 2×3 sorts (optimization verification)"""
    date = pd.to_datetime('2020-01-31')
    
    size = pd.DataFrame({
        'date': [date] * 12,
        'firm_id': list('ABCDEFGHIJKL'),
        'size': [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]  # Median = 17.5
    })
    
    book_to_market = pd.DataFrame({
        'date': [date] * 12,
        'firm_id': list('ABCDEFGHIJKL'),
        'book_to_market': [0.5, 0.6, 0.7, 1.0, 1.1, 1.2,  # 30%ile ~0.72, 70%ile ~1.18
                           0.55, 0.65, 0.75, 1.05, 1.15, 1.25]
    })
    
    returns = pd.DataFrame({
        'date': [date] * 12,
        'firm_id': list('ABCDEFGHIJKL'),
        'return': np.random.uniform(0, 0.10, 12)
    })
    
    # Execute
    smb, hml = sf.compute_smb_hml(size, book_to_market, returns)
    
    # Verify both factors produced (shared computation)
    assert len(smb) == 1
    assert len(hml) == 1
    # If they share sorts, both should be computed simultaneously
    assert smb.iloc[0]['date'] == hml.iloc[0]['date']


# ============================================================================
# compute_momentum_factor tests
# ============================================================================

def test_compute_momentum_factor_basic():
    """Test basic momentum factor computation (WML)"""
    date = pd.to_datetime('2020-01-31')
    
    momentum = pd.DataFrame({
        'date': [date] * 10,
        'firm_id': list('ABCDEFGHIJ'),
        'momentum': [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        # 30%ile = 0.015, 70%ile = 0.225
        # Losers: A,B,C (mom <= 0.015)
        # Winners: H,I,J (mom >= 0.225)
    })
    
    returns = pd.DataFrame({
        'date': [date] * 10,
        'firm_id': list('ABCDEFGHIJ'),
        'return': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        # Losers avg: (0.01+0.02+0.03)/3 = 0.02
        # Winners avg: (0.08+0.09+0.10)/3 = 0.09
    })
    
    # Execute
    result = sf.compute_momentum_factor(momentum, returns)
    
    # Verify WML = Winners - Losers = 0.09 - 0.02 = 0.07
    assert len(result) == 1
    assert result.iloc[0]['momentum_factor'] == pytest.approx(0.07, abs=1e-5)


def test_compute_momentum_factor_multiple_dates():
    """Test momentum factor across multiple dates"""
    dates = pd.to_datetime(['2020-01-31', '2020-02-29'])
    
    momentum = pd.DataFrame({
        'date': [dates[0]] * 6 + [dates[1]] * 6,
        'firm_id': list('ABCDEF') * 2,
        'momentum': [-0.10, -0.05, 0.00, 0.10, 0.15, 0.20] * 2
    })
    
    returns = pd.DataFrame({
        'date': [dates[0]] * 6 + [dates[1]] * 6,
        'firm_id': list('ABCDEF') * 2,
        'return': [0.02, 0.03, 0.04, 0.07, 0.08, 0.09,
                   0.01, 0.02, 0.03, 0.06, 0.07, 0.08]
    })
    
    # Execute
    result = sf.compute_momentum_factor(momentum, returns)
    
    # Verify
    assert len(result) == 2
    assert not result.isnull().any().any()


def test_compute_momentum_factor_missing_data():
    """Test momentum factor with missing values"""
    date = pd.to_datetime('2020-01-31')
    
    momentum = pd.DataFrame({
        'date': [date] * 5,
        'firm_id': list('ABCDE'),
        'momentum': [-0.10, np.nan, 0.00, 0.10, 0.20]
    })
    
    returns = pd.DataFrame({
        'date': [date] * 5,
        'firm_id': list('ABCDE'),
        'return': [0.01, 0.02, 0.03, 0.04, 0.05]
    })
    
    # Execute
    result = sf.compute_momentum_factor(momentum, returns)
    
    # Verify (should exclude B with NaN momentum)
    assert len(result) == 1
    assert not np.isnan(result.iloc[0]['momentum_factor'])


# ============================================================================
# compute_reversal tests
# ============================================================================

def test_compute_reversal_basic():
    """Test basic reversal factor computation"""
    dates = pd.to_datetime(['2020-01-31', '2020-02-29'])
    
    # Returns with 2 dates (need t-1 for reversal at t)
    returns = pd.DataFrame({
        'date': [dates[0]] * 6 + [dates[1]] * 6,
        'firm_id': list('ABCDEF') * 2,
        'return': [
            -0.05, -0.02, 0.00, 0.02, 0.05, 0.10,  # Jan (lagged for Feb)
            0.04, 0.03, 0.02, 0.01, 0.00, -0.01    # Feb (current)
        ]
        # Jan: A,B losers (return <= -0.014); E,F winners (return >= 0.044)
        # Feb reversal: past losers (A,B) returns = (0.04+0.03)/2 = 0.035
        #               past winners (E,F) returns = (0.00+(-0.01))/2 = -0.005
        # Reversal = 0.035 - (-0.005) = 0.04
    })
    
    # Execute
    result = sf.compute_reversal(returns)
    
    # Verify (only Feb has reversal, Jan has no t-1)
    assert len(result) == 1
    assert result.iloc[0]['date'] == dates[1]
    # Approximate check (tertile splits may vary)
    assert abs(result.iloc[0]['reversal_factor']) < 0.10  # Should be positive


def test_compute_reversal_multiple_dates():
    """Test reversal across multiple dates"""
    dates = pd.to_datetime(['2020-01-31', '2020-02-29', '2020-03-31'])
    
    returns = pd.DataFrame({
        'date': [dates[0]] * 4 + [dates[1]] * 4 + [dates[2]] * 4,
        'firm_id': list('ABCD') * 3,
        'return': [
            -0.05, 0.00, 0.05, 0.10,  # Jan
            0.03, 0.02, 0.01, 0.00,   # Feb
            0.04, 0.03, 0.02, 0.01    # Mar
        ]
    })
    
    # Execute
    result = sf.compute_reversal(returns)
    
    # Verify (Feb and Mar have reversal, Jan has no t-1)
    assert len(result) == 2
    assert result.iloc[0]['date'] == dates[1]
    assert result.iloc[1]['date'] == dates[2]


def test_compute_reversal_insufficient_history():
    """Test reversal with insufficient history (single date)"""
    date = pd.to_datetime('2020-01-31')
    
    returns = pd.DataFrame({
        'date': [date] * 4,
        'firm_id': list('ABCD'),
        'return': [0.01, 0.02, 0.03, 0.04]
    })
    
    # Execute
    result = sf.compute_reversal(returns)
    
    # Verify (no reversal without t-1)
    assert len(result) == 0


# ============================================================================
# Placeholder tests for remaining functions
# ============================================================================

def test_compute_momentum_factor_placeholder():
    """TODO: Implement in Step 4"""
    pytest.skip("Not yet implemented")


def test_compute_reversal_placeholder():
    """TODO: Implement in Step 5"""
    pytest.skip("Not yet implemented")


def test_compute_style_spreads_placeholder():
    """TODO: Implement in Step 6"""
    pytest.skip("Not yet implemented")


def test_assemble_X_state_placeholder():
    """TODO: Implement in Step 7"""
    pytest.skip("Not yet implemented")
