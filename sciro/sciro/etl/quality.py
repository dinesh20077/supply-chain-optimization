import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set

logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    check_name: str
    passed: bool
    details: Dict[str, Any]
    severity: str  # 'info', 'warning', 'error'

@dataclass 
class DataQualityReport:
    table_name: str
    total_rows: int
    checks: List[QualityCheckResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == 'error')
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [f"Data Quality Report for {self.table_name}"]
        lines.append(f"Total rows: {self.total_rows}")
        lines.append(f"Overall status: {'PASSED' if self.passed else 'FAILED'}")
        lines.append("-" * 40)
        
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"[{status}] {check.check_name} ({check.severity})")
            for k, v in check.details.items():
                lines.append(f"  - {k}: {v}")
                
        return "\\n".join(lines)

def check_completeness(df: pd.DataFrame, required_columns: List[str], threshold: float = 0.95) -> QualityCheckResult:
    """Check that required columns have at least threshold% non-null values."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    total_rows = len(df)
    if total_rows == 0:
        return QualityCheckResult(
            check_name="completeness",
            passed=False,
            details={"error": "DataFrame is empty"},
            severity="error"
        )
        
    details = {}
    passed = True
    
    for col in required_columns:
        if col not in df.columns:
            details[col] = {"error": "Column not found"}
            passed = False
            continue
            
        non_null_count = df[col].count()
        completeness = non_null_count / total_rows
        details[col] = {"completeness": completeness}
        if completeness < threshold:
            passed = False
            
    return QualityCheckResult(
        check_name="completeness",
        passed=passed,
        details=details,
        severity="error"
    )

def check_uniqueness(df: pd.DataFrame, columns: List[str]) -> QualityCheckResult:
    """Check that values in the given columns are unique (no duplicates)."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        return QualityCheckResult(
            check_name="uniqueness",
            passed=False,
            details={"error": f"Columns not found: {missing_cols}"},
            severity="error"
        )
        
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates(subset=columns))
    
    passed = (total_rows == unique_rows)
    details = {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicates": total_rows - unique_rows
    }
    
    return QualityCheckResult(
        check_name="uniqueness",
        passed=passed,
        details=details,
        severity="error"
    )

def check_value_range(df: pd.DataFrame, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> QualityCheckResult:
    """Check that values are within expected range."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    if column not in df.columns:
        return QualityCheckResult(
            check_name=f"value_range_{column}",
            passed=False,
            details={"error": f"Column '{column}' not found"},
            severity="error"
        )
        
    violations = 0
    total_valid = df[column].count()
    
    numeric_col = pd.to_numeric(df[column], errors='coerce')
    
    if min_val is not None:
        violations += (numeric_col < min_val).sum()
    if max_val is not None:
        violations += (numeric_col > max_val).sum()
        
    passed = bool(violations == 0)
    
    return QualityCheckResult(
        check_name=f"value_range_{column}",
        passed=passed,
        details={
            "column": column,
            "min_val": min_val,
            "max_val": max_val,
            "violations": int(violations),
            "total_valid_records": int(total_valid)
        },
        severity="error"
    )

def check_referential_integrity(df: pd.DataFrame, column: str, valid_values: Set[Any]) -> QualityCheckResult:
    """Check that all values in column exist in the valid_values set."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    if column not in df.columns:
        return QualityCheckResult(
            check_name=f"referential_integrity_{column}",
            passed=False,
            details={"error": f"Column '{column}' not found"},
            severity="error"
        )
        
    non_null_values = df[column].dropna()
    invalid_values = non_null_values[~non_null_values.isin(valid_values)]
    num_invalid = len(invalid_values)
    
    passed = (num_invalid == 0)
    
    return QualityCheckResult(
        check_name=f"referential_integrity_{column}",
        passed=passed,
        details={
            "column": column,
            "invalid_count": num_invalid,
            "unique_invalid_values": invalid_values.unique().tolist()[:10]
        },
        severity="error"
    )

def check_valid_values(df: pd.DataFrame, column: str, allowed_values: Set[Any]) -> QualityCheckResult:
    """Check that column only contains allowed categorical values."""
    res = check_referential_integrity(df, column, allowed_values)
    res.check_name = f"valid_values_{column}"
    return res

def run_quality_checks(
    df: pd.DataFrame,
    table_name: str,
    checks_config: Dict[str, Any]
) -> DataQualityReport:
    """Run all configured quality checks and return a report.
    
    checks_config example:
    {
        'completeness': {'columns': ['name', 'email'], 'threshold': 0.95},
        'uniqueness': {'columns': ['sku']},
        'ranges': {'unit_price': {'min': 0}, 'quantity': {'min': 0}},
        'valid_values': {'status': {'allowed': {'active', 'inactive'}}},
        'referential': {'region_name': {'valid': {'North', 'South', ...}}}
    }
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    report = DataQualityReport(table_name=table_name, total_rows=len(df))
    logger.info(f"Running quality checks for table: {table_name}")
    
    if 'completeness' in checks_config:
        cfg = checks_config['completeness']
        report.checks.append(check_completeness(df, cfg.get('columns', []), cfg.get('threshold', 0.95)))
        
    if 'uniqueness' in checks_config:
        cfg = checks_config['uniqueness']
        report.checks.append(check_uniqueness(df, cfg.get('columns', [])))
        
    if 'ranges' in checks_config:
        for col, cfg in checks_config['ranges'].items():
            report.checks.append(check_value_range(df, col, cfg.get('min'), cfg.get('max')))
            
    if 'valid_values' in checks_config:
        for col, cfg in checks_config['valid_values'].items():
            report.checks.append(check_valid_values(df, col, set(cfg.get('allowed', []))))
            
    if 'referential' in checks_config:
        for col, cfg in checks_config['referential'].items():
            report.checks.append(check_referential_integrity(df, col, set(cfg.get('valid', []))))
            
    return report
