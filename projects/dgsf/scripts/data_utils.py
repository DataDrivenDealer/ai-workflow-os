"""
DGSF Data Utilities - Real Data Loading (DATA-001 Fix)

Purpose: Provide unified real data loading for all DGSF scripts.

Data Sources:
    - xstate_monthly_final.parquet: Monthly state features (nested x_state_vector column)
    - monthly_returns.parquet: Panel returns (month_end, ts_code, ret)

Usage:
    from data_utils import load_real_data, RealDataLoader
    
    # Simple usage
    X, R, is_real = load_real_data()
    
    # With split
    loader = RealDataLoader()
    X_train, R_train, X_val, R_val, X_test, R_test = loader.load_split()

Author: Copilot Agent
Created: 2026-02-03
Updated: 2026-02-03 (DATA-001 fix - expanded nested columns, aligned panel data)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = SCRIPT_DIR.parent / "legacy" / "DGSF"

DEFAULT_XSTATE_PATH = LEGACY_ROOT / "data" / "full" / "xstate_monthly_final.parquet"
DEFAULT_RETURNS_PATH = LEGACY_ROOT / "data" / "final" / "monthly_returns.parquet"


class RealDataLoader:
    """
    Unified real data loader for DGSF project.
    
    Handles the specific data formats:
    - xstate: (n_months, 2) with nested x_state_vector column
    - returns: (n_obs, 3) panel data needing aggregation
    """
    
    def __init__(
        self,
        xstate_path: Optional[Path] = None,
        returns_path: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.xstate_path = xstate_path or DEFAULT_XSTATE_PATH
        self.returns_path = returns_path or DEFAULT_RETURNS_PATH
        self.verbose = verbose
        self._cache: Dict[str, np.ndarray] = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Load and align real data.
        
        Returns:
            X: Features (n_samples, n_features) - monthly state features
            R: Returns (n_samples,) - monthly cross-sectional mean returns
            is_real: True if real data loaded successfully
        """
        if 'X' in self._cache and 'R' in self._cache:
            return self._cache['X'], self._cache['R'], True
            
        try:
            import pandas as pd
            
            if not self.xstate_path.exists() or not self.returns_path.exists():
                self._log(f"Data files not found:")
                self._log(f"  xstate: {self.xstate_path.exists()}")
                self._log(f"  returns: {self.returns_path.exists()}")
                return self._fallback_synthetic()
            
            self._log(f"Loading real data from:")
            self._log(f"  X_state: {self.xstate_path}")
            self._log(f"  Returns: {self.returns_path}")
            
            # Load xstate - contains nested x_state_vector column
            X_df = pd.read_parquet(self.xstate_path)
            self._log(f"  xstate raw shape: {X_df.shape}, columns: {X_df.columns.tolist()}")
            
            # Expand the nested x_state_vector column
            if 'x_state_vector' in X_df.columns:
                x_vectors = X_df['x_state_vector'].tolist()
                X = np.array(x_vectors, dtype=np.float32)
                
                if 'date' in X_df.columns:
                    dates = X_df['date'].astype(str).values
                else:
                    dates = X_df.index.astype(str).values
                    
                self._log(f"  xstate expanded shape: {X.shape} ({X.shape[0]} months, {X.shape[1]} features)")
            else:
                X = X_df.values.astype(np.float32)
                dates = X_df.index.astype(str).values
            
            # Load returns - panel data (month x firm)
            R_df = pd.read_parquet(self.returns_path)
            self._log(f"  returns raw shape: {R_df.shape}, columns: {R_df.columns.tolist()}")
            
            # Aggregate panel returns to monthly cross-sectional mean
            if 'month_end' in R_df.columns and 'ret' in R_df.columns:
                R_monthly = R_df.groupby('month_end')['ret'].mean()
                R_dates = R_monthly.index.astype(str).values
                R_values = R_monthly.values.astype(np.float32)
                self._log(f"  returns aggregated: {len(R_values)} months")
            elif 'ret' in R_df.columns:
                R_values = R_df['ret'].values.astype(np.float32)
                R_dates = R_df.index.astype(str).values
            else:
                R_values = R_df.iloc[:, -1].values.astype(np.float32)
                R_dates = R_df.index.astype(str).values
            
            # Align dates
            xstate_dates_norm = [d[:6] if len(d) >= 6 else d for d in dates]
            returns_dates_norm = [d[:6] if len(d) >= 6 else d for d in R_dates]
            
            xstate_date_set = set(xstate_dates_norm)
            common_dates = [d for d in returns_dates_norm if d in xstate_date_set]
            
            if len(common_dates) == 0:
                self._log(f"No overlapping dates found")
                return self._fallback_synthetic()
            
            # Build aligned arrays
            xstate_date_to_idx = {d: i for i, d in enumerate(xstate_dates_norm)}
            returns_date_to_idx = {d: i for i, d in enumerate(returns_dates_norm)}
            
            aligned_X = []
            aligned_R = []
            for d in common_dates:
                if d in xstate_date_to_idx and d in returns_date_to_idx:
                    aligned_X.append(X[xstate_date_to_idx[d]])
                    aligned_R.append(R_values[returns_date_to_idx[d]])
            
            X = np.array(aligned_X, dtype=np.float32)
            R = np.array(aligned_R, dtype=np.float32)
            
            self._log(f"[OK] Loaded {len(X)} aligned samples with {X.shape[1]} features")
            self._log(f"   Date range: {common_dates[0]} to {common_dates[-1]}")
            
            # Cache
            self._cache['X'] = X
            self._cache['R'] = R
            self._cache['dates'] = common_dates
            
            return X, R, True
            
        except Exception as e:
            self._log(f"Failed to load real data: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_synthetic()
    
    def _fallback_synthetic(
        self,
        n_samples: int = 500,
        n_features: int = 48,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Generate synthetic fallback data."""
        self._log("[WARN] Using synthetic data (fallback)")
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        R = (0.007 + 0.04 * np.random.randn(n_samples)).astype(np.float32)
        return X, R, False
    
    def load_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and split into train/val/test.
        
        Returns:
            X_train, R_train, X_val, R_val, X_test, R_test
        """
        X, R, is_real = self.load()
        n = len(X)
        
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        
        X_train = X[:n_train]
        R_train = R[:n_train]
        X_val = X[n_train:n_train+n_val]
        R_val = R[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        R_test = R[n_train+n_val:]
        
        self._log(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, R_train, X_val, R_val, X_test, R_test
    
    def load_panel_returns(self) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Load full panel returns (not aggregated) for cross-sectional analysis.
        
        Returns:
            returns: (n_months, n_firms) return matrix
            firms: List of firm codes
            dates: List of month dates
            is_real: True if real data
        """
        try:
            import pandas as pd
            
            if not self.returns_path.exists():
                return np.array([]), np.array([]), [], []
            
            R_df = pd.read_parquet(self.returns_path)
            
            # Pivot to (month x firm) matrix
            if 'month_end' in R_df.columns and 'ts_code' in R_df.columns:
                pivot = R_df.pivot_table(
                    index='month_end',
                    columns='ts_code',
                    values='ret',
                    aggfunc='first'
                )
                
                returns = pivot.values.astype(np.float32)
                firms = pivot.columns.tolist()
                dates = pivot.index.astype(str).tolist()
                
                self._log(f"Panel returns: {returns.shape[0]} months x {returns.shape[1]} firms")
                return returns, np.array(firms), dates, True
            
            return np.array([]), np.array([]), [], False
            
        except Exception as e:
            self._log(f"Failed to load panel returns: {e}")
            return np.array([]), np.array([]), [], False


# Convenience function
def load_real_data(
    xstate_path: Optional[Path] = None,
    returns_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Load real DGSF data.
    
    Args:
        xstate_path: Path to xstate parquet (optional)
        returns_path: Path to returns parquet (optional)
        verbose: Print loading progress
        
    Returns:
        X: Features (n_samples, n_features)
        R: Returns (n_samples,)
        is_real: True if real data loaded successfully
    """
    loader = RealDataLoader(xstate_path, returns_path, verbose)
    return loader.load()


if __name__ == "__main__":
    # Test the loader
    print("Testing RealDataLoader...")
    loader = RealDataLoader()
    X, R, is_real = loader.load()
    print(f"\nResult: is_real={is_real}, X.shape={X.shape}, R.shape={R.shape}")
    
    if is_real:
        print("\nStatistics:")
        print(f"  X mean: {X.mean():.4f}, std: {X.std():.4f}")
        print(f"  R mean: {R.mean():.4f}, std: {R.std():.4f}")
        print(f"  R range: [{R.min():.4f}, {R.max():.4f}]")
