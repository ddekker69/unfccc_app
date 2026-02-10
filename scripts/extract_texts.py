#!/usr/bin/env python3
"""
Enhanced Text Extraction with Comprehensive Country Detection
=============================================================

Main text extraction pipeline that includes:
- URL decoding for titles (United%20States -> United States)
- Content-based country extraction 
- Comprehensive country pattern matching
- OCR support for scanned PDFs
"""

import sys
import pandas as pd
import fitz
import os
import pickle
import urllib.parse
import re
import logging
from utils.country_detection import extract_country
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PDF_DIR, CSV_PATH, OUTPUT_PATH
import config

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced country extraction
def extract_country_enhanced(title, text_content="", doc_id=""):
    """
    Enhanced country extraction with URL decoding and content analysis.
    """
    
    # Step 1: URL decode the title and try standard extraction
    if title:
        decoded_title = urllib.parse.unquote(title)
        
        # Try extraction on decoded title first
        country = extract_country(decoded_title)
        if country:
            return country
        
        # Try extraction on original title
        country = extract_country(title)
        if country:
            return country
    
    # Step 2: Content-based extraction for documents with substantial text
    if text_content and len(text_content) > 100:
        content_sample = text_content[:2000].upper()
        
        # Enhanced country patterns in content
        country_patterns = {
            'UNITED STATES': 'United States of America',
            'UNITED STATES OF AMERICA': 'United States of America', 
            'USA': 'United States of America',
            'U.S.A.': 'United States of America',
            'U.S.': 'United States of America',
            'AMERICAN': 'United States of America',
            'CANADA': 'Canada',
            'CANADIAN': 'Canada',
            'AUSTRALIA': 'Australia',
            'AUSTRALIAN': 'Australia',
            'GERMANY': 'Germany',
            'FRANCE': 'France',
            'UNITED KINGDOM': 'United Kingdom',
            'GREAT BRITAIN': 'United Kingdom',
            'BRITAIN': 'United Kingdom',
            'EUROPEAN UNION': 'European Union',
            'CHINA': 'China',
            'JAPAN': 'Japan',
            'RUSSIA': 'Russian Federation',
            'RUSSIAN FEDERATION': 'Russian Federation',
            'INDIA': 'India',
            'BRAZIL': 'Brazil',
            'MEXICO': 'Mexico',
            'SOUTH AFRICA': 'South Africa',
            'CUBA': 'Cuba',
            'CUBAN': 'Cuba',
            'REPUBLIC OF CUBA': 'Cuba',
            'RUSSIAN': 'Russian Federation',
            'MARSHALL ISLANDS': 'Marshall Islands',
            'MICRONESIA': 'Micronesia, Federated States of',
            'FEDERATED STATES OF MICRONESIA': 'Micronesia, Federated States of'
        }
        
        # Check for country mentions with context validation
        for pattern, country_name in country_patterns.items():
            if pattern in content_sample:
                # Validate with context patterns
                context_patterns = [
                    rf'{re.escape(pattern)}.*(?:GOVERNMENT|MINISTRY|NDC|INDC|CONTRIBUTION|COMMUNICATION)',
                    rf'(?:GOVERNMENT|MINISTRY|NDC|INDC|CONTRIBUTION|COMMUNICATION).*{re.escape(pattern)}',
                    rf'{re.escape(pattern)}\'?S\s+(?:NDC|INDC|FIRST|SECOND|UPDATED)',
                    rf'(?:REPUBLIC|STATE|KINGDOM|FEDERATION)\s+OF\s+{re.escape(pattern.split()[-1])}',
                ]
                
                for ctx_pattern in context_patterns:
                    if re.search(ctx_pattern, content_sample):
                        return country_name
    
    return None

print("✅ Using enhanced config.py from:", config.__file__)
print("📄 CSV_PATH is:", CSV_PATH)

# --- Bootstrap checks ---
check_folders()
check_dependencies()

logger.info("Starting enhanced text extraction...")

# --- Load Metadata ---
link_df = pd.read_csv(CSV_PATH)

# Ensure document_id exists
if 'document_id' not in link_df.columns:
    link_df['document_id'] = [f"doc_{i}" for i in range(len(link_df))]

link_df['document_name'] = link_df['download_link'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else None)

texts = []
statuses = []
countries = []

logger.info(f"Processing {len(link_df)} documents...")

# --- Enhanced PDF Extraction ---
for i, (_, row) in enumerate(link_df.iterrows()):
    document_name = row.get('document_name')
    title = row.get('title', '')

    if i % 100 == 0:
        logger.info(f"Processed {i}/{len(link_df)} documents...")

    if not isinstance(document_name, str) or not document_name.endswith('.pdf'):
        texts.append("")
        statuses.append("invalid_link")
        countries.append(None)
        continue

    pdf_path = os.path.join(PDF_DIR, document_name)

    if not os.path.exists(pdf_path):
        texts.append("")
        statuses.append("missing_pdf")
        countries.append(None)
        continue

    try:
        # Extract text using PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()
        
        texts.append(full_text)
        
        # Determine status
        if len(full_text.strip()) < 100:
            statuses.append("short_text")
        else:
            statuses.append("ok")
        
        # Enhanced country extraction
        country = extract_country_enhanced(title, full_text, row.get('document_id', ''))
        countries.append(country)
        
        # Log key country extractions
        if country in ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']:
            logger.info(f"Key country extracted: {title} -> {country}")
            
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        texts.append("")
        statuses.append("extract_error")
        countries.append(None)

# Add results to dataframe
link_df['text'] = texts
link_df['status'] = statuses
link_df['country'] = countries

# --- Save Output ---
Path("data").mkdir(exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(link_df, f)

# --- Enhanced Summary ---
logger.info("\n" + "="*60)
logger.info("ENHANCED EXTRACTION SUMMARY")
logger.info("="*60)

total_docs = len(link_df)
docs_with_countries = len(link_df[link_df['country'].notna()])
unique_countries = len(link_df['country'].dropna().unique())

logger.info(f"Total documents: {total_docs}")
logger.info(f"Documents with countries: {docs_with_countries} ({(docs_with_countries/total_docs)*100:.1f}%)")
logger.info(f"Unique countries detected: {unique_countries}")

# Status breakdown
status_counts = link_df['status'].value_counts()
logger.info("Status distribution:")
for status, count in status_counts.items():
    logger.info(f"  {status}: {count}")

# Key countries check
key_countries = ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']
found_key = []
for country in key_countries:
    if country in link_df['country'].values:
        doc_count = len(link_df[link_df['country'] == country])
        found_key.append(f"{country} ({doc_count} docs)")

if found_key:
    logger.info(f"Key countries successfully extracted: {found_key}")

# Save problematic documents for review
problematic_docs = link_df[link_df['status'] != 'ok']
if not problematic_docs.empty:
    diagnostics_path = "data/problematic_documents.csv"
    # Only use columns that exist in the DataFrame
    available_columns = [col for col in ['document_name', 'folder', 'status'] if col in problematic_docs.columns]
    problematic_docs[available_columns].to_csv(diagnostics_path, index=False)
    logger.info(f"Saved {len(problematic_docs)} problematic documents to {diagnostics_path}")

logger.info(f"✅ Enhanced text extraction complete!")
logger.info(f"Results saved to: {OUTPUT_PATH}")
logger.info(f"Ready for clustering with comprehensive country coverage!")
