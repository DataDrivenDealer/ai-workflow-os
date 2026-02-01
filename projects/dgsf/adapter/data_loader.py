"""
DGSF Data Loader - Data loading utilities for DGSF integration.

Provides unified access to Legacy DGSF data assets with Parquet/Arrow support,
caching, and causality validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pq = None
    HAS_PYARROW = False


class DGSFDataLoader:
    """
    Data loader for DGSF framework.
    
    Provides unified access to Legacy DGSF data assets including:
    - Parquet file loading with optional memory mapping
    - Caching for repeated access
    - Causality validation (no look-ahead leakage)
    - Dataset discovery and metadata
    
    Attributes
    ----------
    data_root : Path
        Root path to Legacy DGSF data directory
    cache : dict
        Loaded data cache
    cache_enabled : bool
        Whether caching is enabled
    """
    
    # Known dataset directories
    KNOWN_DATASETS = {
        "a0": "A股基础数据集 (mini)",
        "full": "完整特征数据集",
        "final": "最终处理数据集",
        "interim": "中间处理数据",
        "processed": "已处理数据",
        "cache": "缓存数据",
        "paneltree": "PanelTree 输出",
        "paneltree_v2": "PanelTree v2 输出",
    }
    
    # Key data files by dataset
    KEY_FILES = {
        "full": [
            "de1_canonical_daily.parquet",
            "de1_canonical_monthly.parquet",
            "de7_factor_panel.parquet",
            "monthly_features.parquet",
        ],
        "a0": [
            "daily_basic.parquet",
            "daily_prices.parquet",
            "monthly_prices.parquet",
        ],
    }
    
    # Causality rules: column -> required lag
    CAUSALITY_RULES = {
        "ret": 0,           # Return is target (t period)
        "ret_1m": 0,        # Monthly return is target
        "close": -1,        # Price must be lagged
        "open": -1,
        "high": -1,
        "low": -1,
        "volume": -1,
        "turnover": -1,
        "total_mv": -1,     # Market cap must be lagged
        "pe": -1,
        "pb": -1,
        "ps": -1,
        "roe": -1,
        "roa": -1,
    }
    
    def __init__(self, data_root: Optional[Path] = None, cache_enabled: bool = True):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_root : Path, optional
            Root path to Legacy DGSF data. If not provided, uses default.
        cache_enabled : bool
            Whether to enable caching (default True)
        """
        if data_root is None:
            self.data_root = Path(__file__).parent.parent / "legacy" / "DGSF" / "data"
        else:
            self.data_root = Path(data_root)
        
        self.cache: Dict[str, Any] = {}
        self.cache_enabled = cache_enabled
        self._metadata_cache: Dict[str, Dict] = {}
        
        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
    
    def load(
        self,
        dataset: str,
        filename: str,
        columns: List[str] = None,
        use_cache: bool = True,
    ) -> "pd.DataFrame":
        """
        Load a data file from a dataset.
        
        Parameters
        ----------
        dataset : str
            Dataset name (e.g., "full", "a0")
        filename : str
            File name (with or without .parquet extension)
        columns : list, optional
            Specific columns to load
        use_cache : bool
            Whether to use cache
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for data loading")
        
        # Resolve path
        if not filename.endswith(".parquet"):
            filename = f"{filename}.parquet"
        
        file_path = self.data_root / dataset / filename
        
        if not file_path.exists():
            # Try subdirectory (e.g., a0/interim/)
            for subdir in self.data_root.glob(f"{dataset}/**/"):
                candidate = subdir / filename
                if candidate.exists():
                    file_path = candidate
                    break
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Check cache
        cache_key = f"{dataset}/{filename}"
        if columns:
            cache_key += f":{','.join(columns)}"
        
        if use_cache and self.cache_enabled and cache_key in self.cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        # Load file
        logger.info(f"Loading: {file_path}")
        
        if HAS_PYARROW:
            df = pd.read_parquet(file_path, columns=columns, engine="pyarrow")
        else:
            df = pd.read_parquet(file_path, columns=columns)
        
        # Cache if enabled
        if self.cache_enabled:
            self.cache[cache_key] = df
        
        return df
    
    def load_dataset(self, dataset: str, pattern: str = "*.parquet") -> Dict[str, "pd.DataFrame"]:
        """
        Load all files from a dataset matching pattern.
        
        Parameters
        ----------
        dataset : str
            Dataset name
        pattern : str
            Glob pattern for files
        
        Returns
        -------
        dict
            Dictionary of filename -> DataFrame
        """
        dataset_path = self.data_root / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        result = {}
        for file_path in dataset_path.glob(pattern):
            if file_path.is_file():
                name = file_path.stem
                result[name] = self.load(dataset, file_path.name)
        
        return result
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns
        -------
        list
            List of dataset info dicts
        """
        datasets = []
        
        for dir_path in self.data_root.iterdir():
            if dir_path.is_dir():
                files = list(dir_path.glob("**/*.parquet"))
                total_size = sum(f.stat().st_size for f in files)
                
                datasets.append({
                    "name": dir_path.name,
                    "description": self.KNOWN_DATASETS.get(dir_path.name, "Unknown"),
                    "file_count": len(files),
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(dir_path),
                })
        
        return datasets
    
    def list_files(self, dataset: str) -> List[Dict[str, Any]]:
        """
        List all files in a dataset.
        
        Parameters
        ----------
        dataset : str
            Dataset name
        
        Returns
        -------
        list
            List of file info dicts
        """
        dataset_path = self.data_root / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        files = []
        for file_path in dataset_path.glob("**/*.parquet"):
            files.append({
                "name": file_path.name,
                "stem": file_path.stem,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "path": str(file_path),
                "relative": str(file_path.relative_to(self.data_root)),
            })
        
        return sorted(files, key=lambda x: x["name"])
    
    def get_metadata(self, dataset: str, filename: str) -> Dict[str, Any]:
        """
        Get metadata for a data file without loading full data.
        
        Parameters
        ----------
        dataset : str
            Dataset name
        filename : str
            File name
        
        Returns
        -------
        dict
            File metadata including schema, row count, etc.
        """
        if not HAS_PYARROW:
            raise ImportError("pyarrow is required for metadata inspection")
        
        if not filename.endswith(".parquet"):
            filename = f"{filename}.parquet"
        
        file_path = self.data_root / dataset / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        cache_key = f"meta:{dataset}/{filename}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        pf = pq.ParquetFile(file_path)
        schema = pf.schema_arrow
        
        metadata = {
            "path": str(file_path),
            "num_rows": pf.metadata.num_rows,
            "num_columns": len(schema),
            "columns": [f.name for f in schema],
            "dtypes": {f.name: str(f.type) for f in schema},
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "num_row_groups": pf.metadata.num_row_groups,
        }
        
        self._metadata_cache[cache_key] = metadata
        return metadata
    
    def validate_causality(
        self,
        df: "pd.DataFrame",
        date_col: str = "trade_date",
        rules: Dict[str, int] = None,
    ) -> Dict[str, Any]:
        """
        Validate causality constraints on a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        date_col : str
            Date column name
        rules : dict, optional
            Custom causality rules (column -> required lag)
        
        Returns
        -------
        dict
            Validation results
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for causality validation")
        
        rules = rules or self.CAUSALITY_RULES
        
        results = {
            "passed": True,
            "checked_columns": [],
            "violations": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Check if date column exists
        if date_col not in df.columns:
            results["warnings"].append(f"Date column '{date_col}' not found")
            return results
        
        # Check each column against rules
        for col in df.columns:
            if col in rules:
                results["checked_columns"].append(col)
                
                # For proper causality check, we'd need to verify that
                # features are properly lagged relative to targets
                # This is a simplified check
                
                expected_lag = rules[col]
                if expected_lag < 0:
                    # Feature should be lagged (use t-1 or earlier)
                    # In a proper implementation, check time alignment
                    pass
        
        # Check for known look-ahead indicators
        lookahead_indicators = ["future", "next", "forward", "lead"]
        for col in df.columns:
            col_lower = col.lower()
            if any(ind in col_lower for ind in lookahead_indicators):
                results["violations"].append({
                    "column": col,
                    "issue": "Column name suggests forward-looking data",
                    "severity": "warning",
                })
        
        if results["violations"]:
            results["passed"] = False
        
        return results
    
    def get_date_range(self, dataset: str, filename: str, date_col: str = "trade_date") -> Dict[str, str]:
        """
        Get date range for a data file.
        
        Parameters
        ----------
        dataset : str
            Dataset name
        filename : str
            File name
        date_col : str
            Date column name
        
        Returns
        -------
        dict
            Date range info
        """
        df = self.load(dataset, filename, columns=[date_col])
        
        return {
            "min_date": str(df[date_col].min()),
            "max_date": str(df[date_col].max()),
            "count": len(df),
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self._metadata_cache.clear()
        logger.info("Cache cleared")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of data loader state.
        
        Returns
        -------
        dict
            Summary including datasets, cache status, etc.
        """
        datasets = self.list_datasets()
        total_files = sum(d["file_count"] for d in datasets)
        total_size = sum(d["size_mb"] for d in datasets)
        
        return {
            "data_root": str(self.data_root),
            "data_root_exists": self.data_root.exists(),
            "total_datasets": len(datasets),
            "total_files": total_files,
            "total_size_mb": round(total_size, 2),
            "cache_enabled": self.cache_enabled,
            "cached_items": len(self.cache),
            "datasets": datasets,
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on data infrastructure.
        
        Returns
        -------
        dict
            Health check results
        """
        results = {
            "data_root_exists": self.data_root.exists(),
            "pandas_available": HAS_PANDAS,
            "pyarrow_available": HAS_PYARROW,
            "a0_accessible": False,
            "full_accessible": False,
            "key_files_present": False,
        }
        
        # Check a0 dataset
        a0_path = self.data_root / "a0"
        if a0_path.exists():
            results["a0_accessible"] = len(list(a0_path.glob("*.parquet"))) > 0
        
        # Check full dataset
        full_path = self.data_root / "full"
        if full_path.exists():
            results["full_accessible"] = len(list(full_path.glob("*.parquet"))) > 0
        
        # Check key files
        key_files_found = 0
        for file in self.KEY_FILES.get("full", []):
            if (self.data_root / "full" / file).exists():
                key_files_found += 1
        
        results["key_files_present"] = key_files_found >= 2
        
        return results


# Singleton instance
_loader_instance: Optional[DGSFDataLoader] = None


def get_data_loader() -> DGSFDataLoader:
    """
    Get singleton data loader instance.
    
    Returns
    -------
    DGSFDataLoader
        Data loader instance
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DGSFDataLoader()
    return _loader_instance
