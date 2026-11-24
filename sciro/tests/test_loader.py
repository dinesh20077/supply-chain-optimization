"""
Unit tests for SCIRO Data Loader (CSV, Excel, and PDF).
"""

import io
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from sciro.ingestion.loader import (
    detect_format,
    normalize_columns,
    load_csv,
    load_excel,
    get_excel_sheet_names,
    load_pdf,
    load_file,
    SUPPORTED_FORMATS,
)


class TestDetectFormat:
    def test_supported_formats(self):
        assert detect_format("file.csv") == ".csv"
        assert detect_format("path/to/data.XLSX") == ".xlsx"
        assert detect_format("legacy.xls") == ".xls"
        assert detect_format("report.pdf") == ".pdf"

    def test_file_object_with_name(self):
        f = io.BytesIO(b"")
        f.name = "invoice.pdf"
        assert detect_format(f) == ".pdf"

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            detect_format("data.json")


class TestNormalizeColumns:
    def test_normalize_columns_formatting(self):
        df = pd.DataFrame(columns=[" Product SKU ", "Unit-Price", "Order.Date", "lead time"])
        norm_df = normalize_columns(df)
        assert list(norm_df.columns) == ["product_sku", "unit_price", "order_date", "lead_time"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        norm_df = normalize_columns(df)
        assert norm_df.empty


class TestCsvLoader:
    def test_load_csv_from_buffer(self):
        csv_data = "name,lead_time_days\nSupplier A,5\nSupplier B,12\n"
        buffer = io.StringIO(csv_data)
        buffer.name = "suppliers.csv"
        df = load_csv(buffer)
        assert len(df) == 2
        assert "name" in df.columns
        assert df["lead_time_days"].tolist() == [5, 12]


class TestExcelLoader:
    def test_load_excel_single_sheet(self):
        buffer = io.BytesIO()
        df_src = pd.DataFrame({"sku": ["P1", "P2"], "unit_price": [100.0, 250.0]})
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_src.to_excel(writer, sheet_name="Products", index=False)
        buffer.seek(0)
        buffer.name = "products.xlsx"

        sheet_names = get_excel_sheet_names(buffer)
        assert "Products" in sheet_names

        loaded = load_excel(buffer, sheet_name="Products")
        assert len(loaded) == 2
        assert loaded["sku"].tolist() == ["P1", "P2"]

    def test_load_excel_multi_sheet(self):
        buffer = io.BytesIO()
        df1 = pd.DataFrame({"name": ["Tata", "Adani"]})
        df2 = pd.DataFrame({"sku": ["X1", "X2"]})
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="Suppliers", index=False)
            df2.to_excel(writer, sheet_name="Products", index=False)
        buffer.seek(0)
        buffer.name = "combined.xlsx"

        sheets = get_excel_sheet_names(buffer)
        assert sheets == ["Suppliers", "Products"]

        loaded_s = load_excel(buffer, sheet_name="Suppliers")
        assert len(loaded_s) == 2
        assert loaded_s["name"].tolist() == ["Tata", "Adani"]


class TestPdfLoader:
    @patch("pdfplumber.open")
    def test_load_pdf_table_extraction(self, mock_pdf_open):
        # Mock pdfplumber extracting a table
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [
            [
                ["Supplier Name", "Lead Time", "Status"],
                ["Tata Motors", "14", "active"],
                ["Reliance Ind", "7", "active"],
            ]
        ]
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        buffer = io.BytesIO(b"%PDF-mock")
        buffer.name = "suppliers.pdf"

        df = load_pdf(buffer)
        assert len(df) == 2
        assert list(df.columns) == ["Supplier Name", "Lead Time", "Status"]
        assert df["Supplier Name"].tolist() == ["Tata Motors", "Reliance Ind"]

    @patch("pdfplumber.open")
    def test_load_pdf_text_fallback(self, mock_pdf_open):
        # Mock pdfplumber where no tables found, but formatted text is found
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []
        mock_page.extract_text.return_value = (
            "sku,category,price\n"
            "SKU-001,Electronics,1200\n"
            "SKU-002,Machinery,4500\n"
        )
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        buffer = io.BytesIO(b"%PDF-mock")
        buffer.name = "products.pdf"

        df = load_pdf(buffer)
        assert len(df) == 2
        assert "sku" in df.columns
        assert df["sku"].tolist() == ["SKU-001", "SKU-002"]


class TestLoadFile:
    def test_load_file_csv_with_normalization(self):
        csv_data = "Product SKU,Cost Price\nSKU10,50.5\n"
        buffer = io.StringIO(csv_data)
        buffer.name = "items.csv"

        df = load_file(buffer, normalize=True)
        assert "product_sku" in df.columns
        assert "cost_price" in df.columns
        assert len(df) == 1

    def test_load_file_excel_with_normalization(self):
        buffer = io.BytesIO()
        df_src = pd.DataFrame({"Supplier Name": ["Adani"], "Lead Time Days": [10]})
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_src.to_excel(writer, sheet_name="Suppliers", index=False)
        buffer.seek(0)
        buffer.name = "suppliers.xlsx"

        df = load_file(buffer, normalize=True)
        assert "supplier_name" in df.columns
        assert "lead_time_days" in df.columns
        assert len(df) == 1

    def test_load_file_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_file("nonexistent_file_path_123.csv")
