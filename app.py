import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
from io import BytesIO
import zipfile

from pdf_converter import PDFtoCSVConverter

# Page configuration
st.set_page_config(
    page_title="PDF to CSV Converter",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for clean design
st.markdown("""
    <style>
    .main {
        padding: 3rem 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.75rem;
        font-size: 16px;
        font-weight: 600;
        margin-top: 1rem;
    }
    h1 {
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📄 PDF to CSV Converter")

# File uploader
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=['pdf'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        temp_pdf_path = Path(temp_dir) / uploaded_file.name
        with open(temp_pdf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Auto-convert on upload
        with st.spinner("Converting..."):
            try:
                # Initialize converter
                converter = PDFtoCSVConverter(output_dir=temp_dir)
                
                # Perform conversion with auto method and merge enabled
                dataframes = converter.convert(
                    pdf_path=str(temp_pdf_path),
                    method='auto',
                    clean=True,
                    merge=True
                )
                
                if dataframes:
                    st.success(f"✅ Extracted {len(dataframes)} table(s)")
                    
                    # Single merged CSV
                    merged_df = pd.concat(dataframes, ignore_index=True)
                    csv_buffer = BytesIO()
                    merged_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_buffer.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_buffer,
                        file_name=f"{Path(uploaded_file.name).stem}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.error("❌ No tables found in the PDF")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(traceback.format_exc())
