import pandas as pd
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PDFtoCSVConverter:
    """
    Robust PDF → CSV converter with multi‑page handling.
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- EXTRACTORS ----------

    def extract_with_camelot(self, pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
        """Extract tables using Camelot."""
        try:
            import camelot

            logger.info(f"Extracting with Camelot from {pdf_path}")
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
            if len(tables) == 0:
                logger.info("Lattice found no tables, trying stream")
                tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")

            dfs = [t.df for t in tables]
            logger.info(f"Camelot extracted {len(dfs)} tables")
            return dfs
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []

    def extract_with_tabula(self, pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
        """Extract tables using Tabula."""
        try:
            import tabula

            logger.info(f"Extracting with Tabula from {pdf_path}")
            tables = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
            logger.info(f"Tabula extracted {len(tables)} tables")
            return tables
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []

    def extract_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Pdfplumber extraction, page‑by‑page to keep memory low,
        and stitching tables that continue across pages.
        """
        try:
            import pdfplumber

            logger.info(f"Extracting with pdfplumber (streaming) from {pdf_path}")
            all_tables: List[pd.DataFrame] = []

            current_headers = None
            current_table_data: List[List] = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) == 0:
                            continue

                        header_row = table[0]
                        body_rows = table[1:]

                        if current_headers is None:
                            # first table
                            current_headers = header_row
                            current_table_data = body_rows
                            logger.info(f"Started table on page {page_num}")
                        elif self._is_same_table(current_headers, header_row):
                            # continuation of same table
                            current_table_data.extend(body_rows)
                            logger.info(
                                f"Continued table on page {page_num}, added {len(body_rows)} rows"
                            )
                        else:
                            # flush previous table and start new
                            if current_table_data:
                                df = pd.DataFrame(current_table_data, columns=current_headers)
                                all_tables.append(df)
                                logger.info(
                                    f"Completed table with {len(current_table_data)} rows"
                                )
                            current_headers = header_row
                            current_table_data = body_rows
                            logger.info(f"Started new table on page {page_num}")

                    # free per‑page resources
                    try:
                        page.flush_cache()
                    except Exception:
                        pass

            # flush last table
            if current_headers is not None and current_table_data:
                df = pd.DataFrame(current_table_data, columns=current_headers)
                all_tables.append(df)
                logger.info(f"Completed final table with {len(current_table_data)} rows")

            logger.info(f"pdfplumber extracted {len(all_tables)} tables total")
            return all_tables

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return []

    # ---------- HELPERS ----------

    def _is_same_table(self, headers1: List, headers2: List) -> bool:
        """Heuristic: are these header rows for the same table?"""
        h1 = [str(h).strip().lower() for h in headers1]
        h2 = [str(h).strip().lower() for h in headers2]

        if len(h1) != len(h2):
            return False

        matches = 0
        for a, b in zip(h1, h2):
            if a == b:
                matches += 1
            elif len(a) > 2 and len(b) > 2 and (a in b or b in a):
                matches += 1

        return (matches / len(h1)) >= 0.7 if h1 else False

    def remove_duplicate_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that look like a repeat of the header row."""
        if df.empty:
            return df

        header_vals = [str(c).strip().lower() for c in df.columns]
        rows_to_drop = []

        for idx, row in df.iterrows():
            row_vals = [str(v).strip().lower() for v in row]
            matches = sum(1 for a, b in zip(header_vals, row_vals) if a == b)
            if header_vals and matches / len(header_vals) >= 0.7:
                rows_to_drop.append(idx)

        if rows_to_drop:
            logger.info(f"Dropping {len(rows_to_drop)} duplicate header rows")
            df = df.drop(rows_to_drop)

        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleanup: remove duplicate headers, empty rows/cols, trim text."""
        df = self.remove_duplicate_headers(df)

        # drop all‑NaN rows/cols
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # drop rows that are all empty strings
        df = df[
            ~df.apply(lambda row: all(str(val).strip() == "" for val in row), axis=1)
        ]

        # normalize non‑numeric columns, without ever using df[col].dtype directly
        for col in df.columns:
            s = df[col]
            try:
                if not pd.api.types.is_numeric_dtype(s):
                    s = s.astype(str).str.strip()
                    s = s.replace("nan", "")
                    df[col] = s
            except Exception:
                df[col] = s.astype(str).str.strip()

        return df.reset_index(drop=True)

    def merge_similar_tables(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Merge consecutive tables that share the same set of columns."""
        if len(dfs) <= 1:
            return dfs

        merged: List[pd.DataFrame] = []
        current_group = [dfs[0]]
        current_cols = set(str(c).strip().lower() for c in dfs[0].columns)

        for df in dfs[1:]:
            cols = set(str(c).strip().lower() for c in df.columns)
            if cols == current_cols:
                current_group.append(df)
            else:
                if len(current_group) > 1:
                    merged.append(pd.concat(current_group, ignore_index=True))
                else:
                    merged.append(current_group[0])
                current_group = [df]
                current_cols = cols

        # flush last group
        if len(current_group) > 1:
            merged.append(pd.concat(current_group, ignore_index=True))
        else:
            merged.append(current_group[0])

        return merged

    # ---------- MAIN ENTRY ----------

    def convert(
        self,
        pdf_path: str,
        method: str = "auto",
        pages: str = "all",
        clean: bool = True,
        merge: bool = False,
    ) -> List[pd.DataFrame]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        dfs: List[pd.DataFrame] = []

        if method == "auto":
            logger.info("Auto mode: trying pdfplumber → camelot → tabula")
            dfs = self.extract_with_pdfplumber(str(pdf_path))
            if not dfs:
                dfs = self.extract_with_camelot(str(pdf_path), pages)
            if not dfs:
                dfs = self.extract_with_tabula(str(pdf_path), pages)
        elif method == "pdfplumber":
            dfs = self.extract_with_pdfplumber(str(pdf_path))
        elif method == "camelot":
            dfs = self.extract_with_camelot(str(pdf_path), pages)
        elif method == "tabula":
            dfs = self.extract_with_tabula(str(pdf_path), pages)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not dfs:
            logger.warning("No tables extracted from PDF")
            return []

        if len(dfs) > 1:
            dfs = self.merge_similar_tables(dfs)

        if clean:
            dfs = [self.clean_dataframe(df) for df in dfs if not df.empty]
            dfs = [df for df in dfs if not df.empty]

        base = pdf_path.stem

        if merge and len(dfs) > 1:
            merged_df = pd.concat(dfs, ignore_index=True)
            out = self.output_dir / f"{base}_merged.csv"
            merged_df.to_csv(out, index=False, encoding="utf-8-sig")
            logger.info(f"Saved merged CSV to {out}")
        else:
            for i, df in enumerate(dfs, 1):
                out = self.output_dir / f"{base}_table_{i}.csv"
                df.to_csv(out, index=False, encoding="utf-8-sig")
                logger.info(f"Saved table {i} to {out}")

        return dfs
