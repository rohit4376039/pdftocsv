import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFtoCSVConverter:
    """
    A robust PDF to CSV converter supporting multiple extraction methods.
    Falls back to alternative methods if primary extraction fails.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the converter.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_with_camelot(self, pdf_path: str, pages: str = 'all') -> List[pd.DataFrame]:
        """Extract tables using Camelot (best for simple tables)."""
        try:
            import camelot
            
            logger.info(f"Extracting with Camelot from {pdf_path}")
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice')
            
            if len(tables) == 0:
                logger.info("Lattice method found no tables, trying stream method")
                tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream')
            
            dataframes = [table.df for table in tables]
            logger.info(f"Camelot extracted {len(dataframes)} tables")
            return dataframes
            
        except Exception as e:
            logger.error(f"Camelot extraction failed: {str(e)}")
            return []
    
    def extract_with_tabula(self, pdf_path: str, pages: str = 'all') -> List[pd.DataFrame]:
        """Extract tables using Tabula (good for multi-page PDFs)."""
        try:
            import tabula
            
            logger.info(f"Extracting with Tabula from {pdf_path}")
            tables = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
            logger.info(f"Tabula extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Tabula extraction failed: {str(e)}")
            return []
    
    def extract_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using Pdfplumber with improved multi-page handling."""
        try:
            import pdfplumber
            
            logger.info(f"Extracting with Pdfplumber from {pdf_path}")
            all_tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                current_table_data = []
                current_headers = None
                
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) == 0:
                            continue
                        
                        # Check if this is a continuation of previous table
                        potential_headers = [str(cell).strip().lower() for cell in table[0]]
                        
                        if current_headers is None:
                            # First table - set headers
                            current_headers = table[0]
                            current_table_data = table[1:]
                            logger.info(f"Started new table on page {page_num} with {len(table[0])} columns")
                        
                        elif self._is_same_table(current_headers, table[0]):
                            # Same table continuing - skip header row and append data
                            current_table_data.extend(table[1:])
                            logger.info(f"Continued table on page {page_num}, added {len(table)-1} rows")
                        
                        else:
                            # Different table - save previous and start new one
                            if current_table_data:
                                df = pd.DataFrame(current_table_data, columns=current_headers)
                                all_tables.append(df)
                                logger.info(f"Completed table with {len(current_table_data)} rows")
                            
                            current_headers = table[0]
                            current_table_data = table[1:]
                            logger.info(f"Started new table on page {page_num}")
                
                # Don't forget the last table
                if current_table_data and current_headers:
                    df = pd.DataFrame(current_table_data, columns=current_headers)
                    all_tables.append(df)
                    logger.info(f"Completed final table with {len(current_table_data)} rows")
            
            logger.info(f"Pdfplumber extracted {len(all_tables)} tables total")
            return all_tables
            
        except Exception as e:
            logger.error(f"Pdfplumber extraction failed: {str(e)}")
            return []
    
    def _is_same_table(self, headers1: List, headers2: List) -> bool:
        """
        Check if two header rows belong to the same table.
        Returns True if they match (indicating table continuation across pages).
        """
        # Normalize headers for comparison
        h1 = [str(h).strip().lower() for h in headers1]
        h2 = [str(h).strip().lower() for h in headers2]
        
        # Must have same number of columns
        if len(h1) != len(h2):
            return False
        
        # Check if headers match (allowing for minor variations)
        matches = sum(1 for a, b in zip(h1, h2) if a == b or 
                     (a in b or b in a) and len(a) > 2 and len(b) > 2)
        
        # Consider it the same table if 70% or more headers match
        similarity = matches / len(h1)
        return similarity >= 0.7
    
    def remove_duplicate_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are duplicate headers within the dataframe.
        This handles cases where headers appear in the middle of data.
        """
        if df.empty:
            return df
        
        # Get the actual header
        header_values = [str(col).strip().lower() for col in df.columns]
        
        # Find rows that match the header
        rows_to_drop = []
        for idx, row in df.iterrows():
            row_values = [str(val).strip().lower() for val in row]
            
            # Check if this row matches the header
            matches = sum(1 for a, b in zip(header_values, row_values) if a == b)
            if matches >= len(header_values) * 0.7:  # 70% match
                rows_to_drop.append(idx)
        
        if rows_to_drop:
            logger.info(f"Removing {len(rows_to_drop)} duplicate header rows")
            df = df.drop(rows_to_drop)
        
        return df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted DataFrame with improved multi-page handling."""
        # Remove duplicate header rows first
        df = self.remove_duplicate_headers(df)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Remove rows where all values are empty strings
        df = df[~df.apply(lambda row: all(str(val).strip() == '' for val in row), axis=1)]
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' strings with empty strings
                df[col] = df[col].replace('nan', '')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def merge_similar_tables(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Merge tables that have the same structure (same columns).
        This helps when a single table is incorrectly split into multiple tables.
        """
        if len(dataframes) <= 1:
            return dataframes
        
        merged = []
        current_group = [dataframes[0]]
        current_columns = set(str(col).strip().lower() for col in dataframes[0].columns)
        
        for df in dataframes[1:]:
            df_columns = set(str(col).strip().lower() for col in df.columns)
            
            # Check if columns match
            if df_columns == current_columns:
                current_group.append(df)
                logger.info(f"Grouping table with matching columns")
            else:
                # Different structure - merge current group and start new one
                if len(current_group) > 1:
                    logger.info(f"Merging {len(current_group)} tables with same structure")
                    merged_df = pd.concat(current_group, ignore_index=True)
                    merged.append(merged_df)
                else:
                    merged.append(current_group[0])
                
                current_group = [df]
                current_columns = df_columns
        
        # Don't forget the last group
        if len(current_group) > 1:
            logger.info(f"Merging final {len(current_group)} tables")
            merged_df = pd.concat(current_group, ignore_index=True)
            merged.append(merged_df)
        else:
            merged.append(current_group[0])
        
        return merged
    
    def convert(self, pdf_path: str, method: str = 'auto', pages: str = 'all',
                clean: bool = True, merge: bool = False) -> List[pd.DataFrame]:
        """Convert PDF to CSV using specified method."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        dataframes = []
        
        if method == 'auto':
            logger.info("Using auto mode: trying multiple extraction methods")
            dataframes = self.extract_with_pdfplumber(str(pdf_path))
            
            if not dataframes:
                dataframes = self.extract_with_camelot(str(pdf_path), pages)
            
            if not dataframes:
                dataframes = self.extract_with_tabula(str(pdf_path), pages)
                
        elif method == 'camelot':
            dataframes = self.extract_with_camelot(str(pdf_path), pages)
        elif method == 'tabula':
            dataframes = self.extract_with_tabula(str(pdf_path), pages)
        elif method == 'pdfplumber':
            dataframes = self.extract_with_pdfplumber(str(pdf_path))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not dataframes:
            logger.warning("No tables extracted from PDF")
            return []
        
        # Try to merge tables with same structure before cleaning
        if len(dataframes) > 1:
            dataframes = self.merge_similar_tables(dataframes)
        
        if clean:
            dataframes = [self.clean_dataframe(df) for df in dataframes if not df.empty]
            # Remove any empty dataframes after cleaning
            dataframes = [df for df in dataframes if not df.empty]
        
        base_name = pdf_path.stem
        
        if merge and len(dataframes) > 1:
            merged_df = pd.concat(dataframes, ignore_index=True)
            output_path = self.output_dir / f"{base_name}_merged.csv"
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved merged CSV to {output_path}")
        else:
            for idx, df in enumerate(dataframes, 1):
                output_path = self.output_dir / f"{base_name}_table_{idx}.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"Saved table {idx} to {output_path}")
        
        return dataframes
