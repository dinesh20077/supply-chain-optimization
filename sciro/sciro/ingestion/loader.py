"""
SCIRO Data Loader.

Handles reading CSV, Excel (.xlsx, .xls), and PDF files into pandas DataFrames
with automatic format detection, encoding handling, table extraction, and
column name normalization.
"""

import os
import io
import re
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".pdf"}


def detect_format(filepath_or_name: Union[str, Path, Any]) -> str:
    """Detect file format from filename or file-like object.

    Args:
        filepath_or_name: Path or object with a name attribute.

    Returns:
        File extension in lowercase (e.g., '.csv', '.xlsx', '.pdf').

    Raises:
        ValueError: If format is not supported.
    """
    if isinstance(filepath_or_name, (str, Path)):
        name = str(filepath_or_name)
    elif hasattr(filepath_or_name, "name"):
        name = getattr(filepath_or_name, "name")
    else:
        name = ""

    _, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported: {sorted(list(SUPPORTED_FORMATS))}"
        )
    return ext


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, strip whitespace, replace spaces with underscores.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with normalized column names.
    """
    if len(df.columns) == 0:
        return df

    original = list(df.columns)
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        for col in df.columns
    ]
    renamed = {o: n for o, n in zip(original, df.columns) if o != n}
    if renamed:
        logger.info("Normalized columns: %s", renamed)
    return df


def load_csv(
    filepath_or_buffer: Any,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Handles encoding fallback (utf-8 → latin-1).

    Args:
        filepath_or_buffer: Path to CSV or file-like buffer.
        encoding: File encoding to try first.
        **kwargs: Additional arguments passed to pd.read_csv().

    Returns:
        Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath_or_buffer, encoding=encoding, **kwargs)
        name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
        logger.info("Loaded CSV: %s (%d rows, %d columns)", name, len(df), len(df.columns))
        return df
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, trying latin-1", filepath_or_buffer)
        if hasattr(filepath_or_buffer, "seek"):
            filepath_or_buffer.seek(0)
        df = pd.read_csv(filepath_or_buffer, encoding="latin-1", **kwargs)
        name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
        logger.info("Loaded CSV (latin-1): %s (%d rows, %d columns)", name, len(df), len(df.columns))
        return df


def get_excel_sheet_names(filepath_or_buffer: Any) -> List[str]:
    """Retrieve all sheet names from an Excel file.

    Args:
        filepath_or_buffer: Path to the Excel file or file-like buffer.

    Returns:
        List of sheet names.
    """
    try:
        if hasattr(filepath_or_buffer, "seek"):
            filepath_or_buffer.seek(0)
        excel_file = pd.ExcelFile(filepath_or_buffer)
        return excel_file.sheet_names
    except Exception as e:
        logger.error("Failed to inspect Excel sheet names: %s", e)
        return ["Sheet1"]


def load_excel(
    filepath_or_buffer: Any,
    sheet_name: Optional[Union[str, int]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load an Excel file (.xlsx, .xls) into a DataFrame.

    Args:
        filepath_or_buffer: Path to Excel file or file-like buffer.
        sheet_name: Sheet to load (None = first sheet / index 0).
        **kwargs: Additional arguments passed to pd.read_excel().

    Returns:
        Loaded DataFrame.
    """
    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)

    sheet = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(filepath_or_buffer, sheet_name=sheet, **kwargs)
    name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
    logger.info(
        "Loaded Excel: %s (sheet='%s', %d rows, %d columns)",
        name, sheet if sheet is not None else "0", len(df), len(df.columns),
    )
    return df


def load_pdf(
    filepath_or_buffer: Any,
    pages: Optional[List[int]] = None,
    table_settings: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Extract tabular data or structured records from a PDF file.

    Uses pdfplumber to extract tables across pages. If tables are found,
    constructs a unified DataFrame. If no tables are detected, parses
    structured text lines.

    Args:
        filepath_or_buffer: Path to PDF or BytesIO / file buffer.
        pages: List of page numbers (0-indexed) to inspect, or None for all.
        table_settings: Optional dictionary passed to page.extract_tables().
        **kwargs: Additional parameters.

    Returns:
        Extracted pandas DataFrame.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required to load PDF files. Install with: pip install pdfplumber"
        ) from exc

    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)

    tables_found: List[List[List[Optional[str]]]] = []
    text_lines: List[str] = []

    with pdfplumber.open(filepath_or_buffer) as pdf:
        num_pages = len(pdf.pages)
        target_pages = [pdf.pages[i] for i in pages if i < num_pages] if pages is not None else pdf.pages

        for page in target_pages:
            extracted = page.extract_tables(table_settings or {})
            if extracted:
                for tbl in extracted:
                    if tbl and len(tbl) > 0:
                        tables_found.append(tbl)
            else:
                raw_text = page.extract_text()
                if raw_text:
                    for line in raw_text.splitlines():
                        s = line.strip()
                        if s:
                            text_lines.append(s)

    # Strategy 1: Tabular data extracted
    if tables_found:
        dfs = []
        for tbl in tables_found:
            if not tbl:
                continue
            if len(tbl) == 1:
                # Single row table
                dfs.append(pd.DataFrame(tbl))
                continue

            # Header from the first row
            raw_header = tbl[0]
            header = [
                str(c).strip().replace("\n", " ") if c is not None and str(c).strip() else f"col_{i}"
                for i, c in enumerate(raw_header)
            ]
            data_rows = tbl[1:]

            cleaned_rows = []
            for row in data_rows:
                cleaned = [str(c).strip() if c is not None else "" for c in row]
                if len(cleaned) < len(header):
                    cleaned += [""] * (len(header) - len(cleaned))
                elif len(cleaned) > len(header):
                    cleaned = cleaned[:len(header)]
                cleaned_rows.append(cleaned)

            table_df = pd.DataFrame(cleaned_rows, columns=header)
            table_df = table_df.replace("", pd.NA).dropna(how="all").fillna("")
            if not table_df.empty:
                dfs.append(table_df)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
            logger.info("Loaded PDF tables: %s (%d rows, %d columns)", name, len(df), len(df.columns))
            return df

    # Strategy 2: Text parsing fallback (delimiter or structured lines)
    if text_lines:
        parsed_rows = []
        for line in text_lines:
            # Check for delimiter: comma, tab, pipe, or multiple spaces
            if "," in line:
                row = [part.strip() for part in next(csv.reader([line]))]
            elif "\t" in line:
                row = [part.strip() for part in line.split("\t")]
            elif "|" in line:
                row = [part.strip() for part in line.split("|") if part.strip()]
            elif re.search(r"\s{2,}", line):
                row = [part.strip() for part in re.split(r"\s{2,}", line)]
            else:
                row = [line]
            parsed_rows.append(row)

        if parsed_rows and len(parsed_rows) > 1:
            header_len = len(parsed_rows[0])
            if header_len > 1 and all(len(r) == header_len for r in parsed_rows[:5]):
                header = [f"col_{c}" if not str(c).strip() else str(c).strip() for c in parsed_rows[0]]
                df = pd.DataFrame(parsed_rows[1:], columns=header)
                name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
                logger.info("Loaded PDF text as table: %s (%d rows, %d columns)", name, len(df), len(df.columns))
                return df

        # Otherwise return raw lines DataFrame
        df = pd.DataFrame({"text_content": text_lines})
        name = getattr(filepath_or_buffer, "name", str(filepath_or_buffer))
        logger.info("Loaded PDF plain text: %s (%d lines)", name, len(df))
        return df

    logger.warning("No tables or text extracted from PDF: %s", filepath_or_buffer)
    return pd.DataFrame()


def load_file(
    filepath_or_buffer: Any,
    filename: Optional[str] = None,
    normalize: bool = True,
    sheet_name: Optional[Union[str, int]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a data file (CSV, Excel, or PDF) into a DataFrame.

    Auto-detects format from file extension or object name. Normalizes column names
    by default.

    Args:
        filepath_or_buffer: Path to the data file or file-like buffer (e.g. UploadedFile).
        filename: Optional filename override if passing a buffer without a name attribute.
        normalize: Whether to normalize column names.
        sheet_name: Sheet to load for Excel workbooks.
        **kwargs: Additional arguments passed to the reader.

    Returns:
        Loaded and optionally normalized DataFrame.

    Raises:
        FileNotFoundError: If the file path doesn't exist on disk.
        ValueError: If the format is not supported.
    """
    # Check disk existence only if it's a string path
    if isinstance(filepath_or_buffer, (str, Path)):
        path_str = str(filepath_or_buffer)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Data file not found: {path_str}")
        file_name = filename or os.path.basename(path_str)
    else:
        file_name = filename or getattr(filepath_or_buffer, "name", "data.csv")

    file_format = detect_format(file_name)

    if file_format == ".csv":
        df = load_csv(filepath_or_buffer, **kwargs)
    elif file_format in (".xlsx", ".xls"):
        df = load_excel(filepath_or_buffer, sheet_name=sheet_name, **kwargs)
    elif file_format == ".pdf":
        df = load_pdf(filepath_or_buffer, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    if normalize:
        df = normalize_columns(df)

    return df
