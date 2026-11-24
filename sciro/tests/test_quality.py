"""
Tests for SCIRO ETL data quality checks.
"""

import pytest
import pandas as pd

from sciro.etl.quality import (
    QualityCheckResult,
    DataQualityReport,
    check_completeness,
    check_uniqueness,
    check_value_range,
    check_referential_integrity,
    check_valid_values,
    run_quality_checks,
)


# =============================================================================
# QualityCheckResult / DataQualityReport Tests
# =============================================================================
class TestQualityReport:
    def test_report_passes_when_no_errors(self):
        report = DataQualityReport(table_name="test", total_rows=10)
        report.checks.append(
            QualityCheckResult("check1", True, {}, "error")
        )
        report.checks.append(
            QualityCheckResult("check2", False, {}, "warning")  # Warning doesn't fail
        )
        assert report.passed is True

    def test_report_fails_on_error_severity(self):
        report = DataQualityReport(table_name="test", total_rows=10)
        report.checks.append(
            QualityCheckResult("check1", False, {}, "error")
        )
        assert report.passed is False

    def test_summary_output(self):
        report = DataQualityReport(table_name="test", total_rows=10)
        report.checks.append(
            QualityCheckResult("completeness", True, {"pct": "100%"}, "error")
        )
        summary = report.summary()
        assert "test" in summary
        assert "PASS" in summary


# =============================================================================
# check_completeness Tests
# =============================================================================
class TestCheckCompleteness:
    def test_complete_data_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_completeness(df, ["a", "b"])
        assert result.passed is True

    def test_incomplete_data_fails(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [4, 5, 6]})
        result = check_completeness(df, ["a"], threshold=0.95)
        assert result.passed is False

    def test_custom_threshold(self):
        df = pd.DataFrame({"a": [1, None, 3, 4, 5]})
        result = check_completeness(df, ["a"], threshold=0.5)
        assert result.passed is True


# =============================================================================
# check_uniqueness Tests
# =============================================================================
class TestCheckUniqueness:
    def test_unique_values_pass(self):
        df = pd.DataFrame({"sku": ["A", "B", "C"]})
        result = check_uniqueness(df, ["sku"])
        assert result.passed is True

    def test_duplicate_values_fail(self):
        df = pd.DataFrame({"sku": ["A", "A", "B"]})
        result = check_uniqueness(df, ["sku"])
        assert result.passed is False


# =============================================================================
# check_value_range Tests
# =============================================================================
class TestCheckValueRange:
    def test_in_range_passes(self):
        df = pd.DataFrame({"price": [100, 200, 300]})
        result = check_value_range(df, "price", min_val=0, max_val=500)
        assert result.passed is True

    def test_below_min_fails(self):
        df = pd.DataFrame({"price": [-10, 200, 300]})
        result = check_value_range(df, "price", min_val=0)
        assert result.passed is False

    def test_above_max_fails(self):
        df = pd.DataFrame({"price": [100, 200, 1000]})
        result = check_value_range(df, "price", max_val=500)
        assert result.passed is False


# =============================================================================
# check_referential_integrity Tests
# =============================================================================
class TestCheckReferentialIntegrity:
    def test_valid_references_pass(self):
        df = pd.DataFrame({"region_name": ["North", "South"]})
        valid = {"North", "South", "East"}
        result = check_referential_integrity(df, "region_name", valid)
        assert result.passed is True

    def test_invalid_references_fail(self):
        df = pd.DataFrame({"region_name": ["North", "Unknown"]})
        valid = {"North", "South"}
        result = check_referential_integrity(df, "region_name", valid)
        assert result.passed is False


# =============================================================================
# check_valid_values Tests
# =============================================================================
class TestCheckValidValues:
    def test_valid_values_pass(self):
        df = pd.DataFrame({"status": ["active", "inactive"]})
        result = check_valid_values(df, "status", {"active", "inactive"})
        assert result.passed is True

    def test_invalid_values_fail(self):
        df = pd.DataFrame({"status": ["active", "unknown"]})
        result = check_valid_values(df, "status", {"active", "inactive"})
        assert result.passed is False


# =============================================================================
# run_quality_checks Tests
# =============================================================================
class TestRunQualityChecks:
    def test_comprehensive_check(self):
        df = pd.DataFrame({
            "sku": ["A", "B", "C"],
            "price": [100, 200, 300],
            "status": ["active", "active", "inactive"],
            "region": ["North", "South", "East"],
        })
        config = {
            "completeness": {"columns": ["sku", "price"], "threshold": 0.95},
            "uniqueness": {"columns": ["sku"]},
            "ranges": {"price": {"min": 0, "max": 1000}},
            "valid_values": {"status": {"allowed": {"active", "inactive"}}},
            "referential": {
                "region": {"valid": {"North", "South", "East", "West"}}
            },
        }
        report = run_quality_checks(df, "products", config)
        assert report.passed is True
        assert len(report.checks) > 0
