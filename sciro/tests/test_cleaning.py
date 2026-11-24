"""
Tests for SCIRO ETL cleaning module.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.etl.cleaning import (
    handle_missing_values,
    detect_duplicates,
    standardize_text_columns,
    coerce_types,
    clean_dataframe,
)


# =============================================================================
# handle_missing_values Tests
# =============================================================================
class TestHandleMissingValues:
    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = handle_missing_values(df, {"a": "drop"})
        assert len(result) == 2
        assert result["a"].isna().sum() == 0

    def test_mean_strategy(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = handle_missing_values(df, {"a": "mean"})
        assert result["a"].iloc[1] == 2.0

    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1.0, None, 5.0, 3.0]})
        result = handle_missing_values(df, {"a": "median"})
        assert result["a"].iloc[1] == 3.0

    def test_mode_strategy(self):
        df = pd.DataFrame({"a": ["x", None, "x", "y"]})
        result = handle_missing_values(df, {"a": "mode"})
        assert result["a"].iloc[1] == "x"

    def test_default_strategy(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        result = handle_missing_values(df, {"a": "default"}, fill_values={"a": 99})
        assert result["a"].iloc[1] == 99

    def test_nonexistent_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = handle_missing_values(df, {"nonexistent": "drop"})
        assert len(result) == 3

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            handle_missing_values("not a dataframe", {})


# =============================================================================
# detect_duplicates Tests
# =============================================================================
class TestDetectDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        clean, dups = detect_duplicates(df)
        assert len(clean) == 3
        assert len(dups) == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        clean, dups = detect_duplicates(df)
        assert len(clean) == 2
        assert len(dups) == 1

    def test_subset_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
        clean, dups = detect_duplicates(df, subset=["a"])
        assert len(clean) == 2
        assert len(dups) == 1


# =============================================================================
# standardize_text_columns Tests
# =============================================================================
class TestStandardizeText:
    def test_strip_and_lower(self):
        df = pd.DataFrame({"name": ["  HELLO  ", "  World  "]})
        result = standardize_text_columns(df, ["name"], ["strip", "lower"])
        assert result["name"].iloc[0] == "hello"
        assert result["name"].iloc[1] == "world"

    def test_title_case(self):
        df = pd.DataFrame({"name": ["hello world"]})
        result = standardize_text_columns(df, ["name"], ["title"])
        assert result["name"].iloc[0] == "Hello World"

    def test_nonexistent_column_skipped(self):
        df = pd.DataFrame({"a": ["test"]})
        result = standardize_text_columns(df, ["nonexistent"])
        assert len(result) == 1


# =============================================================================
# coerce_types Tests
# =============================================================================
class TestCoerceTypes:
    def test_string_to_int(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = coerce_types(df, {"a": "int"})
        assert result["a"].dtype in (np.int64, np.int32, int)

    def test_string_to_float(self):
        df = pd.DataFrame({"a": ["1.5", "2.5"]})
        result = coerce_types(df, {"a": "float"})
        assert result["a"].dtype == np.float64

    def test_string_to_datetime(self):
        df = pd.DataFrame({"a": ["2025-01-01", "2025-06-15"]})
        result = coerce_types(df, {"a": "datetime"})
        assert pd.api.types.is_datetime64_any_dtype(result["a"])


# =============================================================================
# clean_dataframe Tests
# =============================================================================
class TestCleanDataframe:
    def test_full_cleaning_pipeline(self):
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Alice  "],
            "value": [1.0, None, 3.0],
        })
        clean_df, report = clean_dataframe(
            df,
            duplicate_subset=["name"],
            missing_strategies={"value": "mean"},
            text_columns=["name"],
        )
        assert report["rows_before"] >= 2
        assert "duplicates_removed" in report

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": []})
        clean_df, report = clean_dataframe(df)
        assert len(clean_df) == 0
