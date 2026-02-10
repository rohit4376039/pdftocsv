import pandas as pd
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PDFtoCSVConverter:
    """
    Robust PDF → CSV converter with per‑page tables and safe merging.
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
        Pdfplumber extraction, page‑by‑page to keep memory low.
        Each page/table is treated independently (no cross‑page stitching).
        """
        try:
            import pdfplumber

            logger.info(f"Extracting with pdfplumber (per‑page) from {pdf_path}")
            all_tables: List[pd.DataFrame] = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) == 0:
                            continue

                        header_row = table[0]
                        body_rows = table[1:]

                        df = pd.DataFrame(body_rows, columns=header_row)
                        all_tables.append(df)
                        logger.info(
                            f"Extracted table on page {page_num} with {len(body_rows)} rows"
                        )

                    try:
                        page.flush_cache()
                    except Exception:
                        pass

            logger.info(f"pdfplumber extracted {len(all_tables)} tables total")
            return all_tables

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return []

    # ---------- HELPERS ----------

    def _is_same_table(self, headers1: List, headers2: List) -> bool:
        """Treat each page/table as independent to avoid misalignment."""
        return False

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

    def _flatten_multirow_header(self, df: pd.DataFrame, max_header_rows: int = 3) -> pd.DataFrame:
        """
        If the first 2–3 rows look like header fragments, merge them into one header row.
        Example: stacked 'Gen', 'der', '(M/F/O)' → 'Gen der (M/F/O)'.
        """
        if df.empty or len(df) < 2:
            return df

        header_block = df.head(max_header_rows)

        # If top row is mostly empty/NaN (title band), drop it
        if header_block.iloc[0].isna().mean() > 0.7:
            df = df.iloc[1:].reset_index(drop=True)
            header_block = df.head(max_header_rows)
            if df.empty:
                return df

        header_rows = header_block.index.tolist()
        if len(header_rows) <= 1:
            return df

        parts = df.iloc[header_rows].astype(str).fillna("").applymap(str.strip)
        new_cols = []
        for col_idx in range(parts.shape[1]):
            tokens = [t for t in parts.iloc[:, col_idx].tolist() if t and t.lower() != "nan"]
            col_name = " ".join(tokens).strip()
            new_cols.append(col_name or f"col_{col_idx}")

        df = df.iloc[len(header_rows):].reset_index(drop=True)
        df.columns = new_cols
        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleanup + fix stacked headers."""
        df = self._flatten_multirow_header(df)
        df = self.remove_duplicate_headers(df)

        df = df.dropna(how="all").dropna(axis=1, how="all")

        df = df[
            ~df.apply(lambda row: all(str(val).strip() == "" for val in row), axis=1)
        ]

        for col in df.columns:
            df[col] = df[col].apply(lambda v: str(v).strip())
            df[col] = df[col].replace("nan", "")

        return df.reset_index(drop=True)

    def merge_similar_tables(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Merge tables that have the same logical columns.
        Align columns by name and fill missing ones with empty strings
        so page breaks do not shift cells or drop values.
        """
        if len(dfs) <= 1:
            return dfs

        def norm_cols(df: pd.DataFrame):
            return tuple(str(c).strip().lower() for c in df.columns)

        groups = {}
        for df in dfs:
            key = norm_cols(df)
            groups.setdefault(key, []).append(df)

        merged: List[pd.DataFrame] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Union of all columns across the group
            all_cols = []
            for df in group:
                for c in df.columns:
                    if c not in all_cols:
                        all_cols.append(c)

            aligned = []
            for df in group:
                tmp = df.copy()
                for c in all_cols:
                    if c not in tmp.columns:
                        tmp[c] = ""
                tmp = tmp[all_cols]
                aligned.append(tmp)

            merged_df = pd.concat(aligned, ignore_index=True)
            merged.append(merged_df)

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

        # ensure unique column names in each df before any concat
        safe_dfs: List[pd.DataFrame] = []
        for df in dfs:
            df = df.copy()
            counts = {}
            new_cols = []
            for col in df.columns:
                col_str = str(col)
                if col_str not in counts:
                    counts[col_str] = 0
                    new_cols.append(col_str)
                else:
                    counts[col_str] += 1
                    new_cols.append(f"{col_str}.{counts[col_str]}")
            df.columns = new_cols
            safe_dfs.append(df)
        dfs = safe_dfs

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
