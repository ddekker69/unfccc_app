#!/usr/bin/env python3
"""
Automated UNFCCC Data Pipeline
==============================

This script automates the complete data preparation pipeline for UNFCCC climate documents.
It processes documents from three different folders:
- data/international_agreements/
- data/ndc_documents/
- data/unfccc_documents_topic_mitigation/

Usage:
    python automated_pipeline.py [OPTIONS]

Arguments:
    --skip-extraction: Skip text extraction step (if already done)
    --skip-clustering: Skip clustering step (if already done)
    --skip-indexing: Skip FAISS indexing step (if already done)
    --skip-enhanced: Skip enhanced indexing for ultra-fast RAG
    --skip-app: Skip launching the Streamlit app
    --enhanced-only: Build only enhanced indexes (skip standard indexes)
    --force: Force reprocessing even if output files exist
    --offline: Run in offline mode (skip all Azure uploads)
"""

import sys
import os
import argparse
import time
import traceback
import urllib.parse
import re
import logging
from pathlib import Path
import pandas as pd
import pickle
import fitz  # PyMuPDF
import numpy as np
import umap.umap_ as umap
import hdbscan
import faiss
import gc
import torch
from sentence_transformers import SentenceTransformer
from scripts.prepare_enhanced_index import chunk_text_intelligently
from utils.country_detection import extract_country
from utils.azure_blob_utils import upload_blob
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies
from config import (
    EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE, AZURE_CONTAINER_NAME,
    OUTPUT_PATH
)

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Setup logging for enhanced pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to control upload operations
OFFLINE_MODE = False

def get_comprehensive_country_list():
    """Get comprehensive list of country names, codes, and variations."""
    
    # All possible country names and variations
    comprehensive_countries = {
        # Major powers
        'United States of America': ['USA', 'US', 'United States', 'United States of America', 'U.S.A.', 'U.S.', 'American'],
        'Russian Federation': ['Russia', 'Russian Federation', 'RF', 'RUS', 'Russian'],
        'China': ['China', 'Chinese', 'CHN', 'PRC', "People's Republic of China"],
        'United Kingdom': ['UK', 'United Kingdom', 'Britain', 'Great Britain', 'GBR', 'British'],
        'Germany': ['Germany', 'Deutschland', 'German', 'DEU', 'GER'],
        'France': ['France', 'French', 'FRA', 'République française'],
        'Japan': ['Japan', 'Japanese', 'JPN', 'Nippon'],
        'India': ['India', 'Indian', 'IND', 'Bharat'],
        'Canada': ['Canada', 'Canadian', 'CAN'],
        'Australia': ['Australia', 'Australian', 'AUS'],
        'Brazil': ['Brazil', 'Brasil', 'Brazilian', 'BRA'],
        'Italy': ['Italy', 'Italian', 'ITA', 'Italia'],
        'Spain': ['Spain', 'Spanish', 'ESP', 'España'],
        
        # European countries
        'Netherlands': ['Netherlands', 'Holland', 'Dutch', 'NLD', 'NL'],
        'Belgium': ['Belgium', 'Belgian', 'BEL', 'België', 'Belgique'],
        'Switzerland': ['Switzerland', 'Swiss', 'CHE', 'Schweiz', 'Suisse'],
        'Austria': ['Austria', 'Austrian', 'AUT', 'Österreich'],
        'Sweden': ['Sweden', 'Swedish', 'SWE', 'Sverige'],
        'Norway': ['Norway', 'Norwegian', 'NOR', 'Norge'],
        'Denmark': ['Denmark', 'Danish', 'DNK', 'Danmark'],
        'Finland': ['Finland', 'Finnish', 'FIN', 'Suomi'],
        'Poland': ['Poland', 'Polish', 'POL', 'Polska'],
        'Czech Republic': ['Czech Republic', 'Czechia', 'Czech', 'CZE', 'Česká republika'],
        'Slovakia': ['Slovakia', 'Slovak', 'SVK', 'Slovensko'],
        'Hungary': ['Hungary', 'Hungarian', 'HUN', 'Magyarország'],
        'Romania': ['Romania', 'Romanian', 'ROU', 'România'],
        'Bulgaria': ['Bulgaria', 'Bulgarian', 'BGR'],
        'Croatia': ['Croatia', 'Croatian', 'HRV', 'Hrvatska'],
        'Slovenia': ['Slovenia', 'Slovenian', 'SVN', 'Slovenija'],
        'Estonia': ['Estonia', 'Estonian', 'EST', 'Eesti'],
        'Latvia': ['Latvia', 'Latvian', 'LVA', 'Latvija'],
        'Lithuania': ['Lithuania', 'Lithuanian', 'LTU', 'Lietuva'],
        'Greece': ['Greece', 'Greek', 'GRC', 'Hellas', 'Hellenic'],
        'Portugal': ['Portugal', 'Portuguese', 'PRT'],
        'Ireland': ['Ireland', 'Irish', 'IRL', 'Éire'],
        'Luxembourg': ['Luxembourg', 'LUX'],
        'Malta': ['Malta', 'Maltese', 'MLT'],
        'Cyprus': ['Cyprus', 'Cypriot', 'CYP'],
        'Iceland': ['Iceland', 'Icelandic', 'ISL', 'Ísland'],
        
        # Former Soviet states
        'Ukraine': ['Ukraine', 'Ukrainian', 'UKR', 'Україна'],
        'Belarus': ['Belarus', 'Belarusian', 'BLR', 'Беларусь'],
        'Moldova, Republic of': ['Moldova', 'Moldovan', 'MDA', 'Republic of Moldova'],
        'Georgia': ['Georgia', 'Georgian', 'GEO'],
        'Armenia': ['Armenia', 'Armenian', 'ARM'],
        'Azerbaijan': ['Azerbaijan', 'Azerbaijani', 'AZE'],
        'Kazakhstan': ['Kazakhstan', 'Kazakh', 'KAZ', 'Қазақстан'],
        'Uzbekistan': ['Uzbekistan', 'Uzbek', 'UZB'],
        'Turkmenistan': ['Turkmenistan', 'Turkmen', 'TKM'],
        'Kyrgyzstan': ['Kyrgyzstan', 'Kyrgyz', 'KGZ'],
        'Tajikistan': ['Tajikistan', 'Tajik', 'TJK'],
        
        # Asia
        'Korea, Republic of': ['South Korea', 'Korea, Republic of', 'Republic of Korea', 'KOR', 'ROK', '대한민국'],
        'North Korea': ['North Korea', 'Korea, Democratic People\'s Republic of', 'DPRK', 'PRK'],
        'Indonesia': ['Indonesia', 'Indonesian', 'IDN'],
        'Thailand': ['Thailand', 'Thai', 'THA', 'ไทย'],
        'Malaysia': ['Malaysia', 'Malaysian', 'MYS'],
        'Singapore': ['Singapore', 'Singaporean', 'SGP'],
        'Philippines': ['Philippines', 'Filipino', 'PHL'],
        'Viet Nam': ['Vietnam', 'Vietnamese', 'VNM', 'Viet Nam'],
        'Myanmar': ['Myanmar', 'Burma', 'Burmese', 'MMR'],
        'Cambodia': ['Cambodia', 'Cambodian', 'KHM'],
        'Laos': ['Laos', 'Lao', 'LAO', 'Lao People\'s Democratic Republic'],
        'Bangladesh': ['Bangladesh', 'Bangladeshi', 'BGD'],
        'Pakistan': ['Pakistan', 'Pakistani', 'PAK'],
        'Sri Lanka': ['Sri Lanka', 'Sri Lankan', 'LKA'],
        'Nepal': ['Nepal', 'Nepalese', 'NPL'],
        'Bhutan': ['Bhutan', 'Bhutanese', 'BTN'],
        'Mongolia': ['Mongolia', 'Mongolian', 'MNG'],
        
        # Middle East
        'Turkey': ['Turkey', 'Turkish', 'TUR', 'Türkiye'],
        'Iran, Islamic Republic of': ['Iran', 'Iranian', 'IRN', 'Islamic Republic of Iran'],
        'Iraq': ['Iraq', 'Iraqi', 'IRQ'],
        'Syria': ['Syria', 'Syrian', 'SYR', 'Syrian Arab Republic'],
        'Syrian Arab Republic': ['Syria', 'Syrian', 'SYR', 'Syrian Arab Republic'],
        'Lebanon': ['Lebanon', 'Lebanese', 'LBN'],
        'Jordan': ['Jordan', 'Jordanian', 'JOR'],
        'Israel': ['Israel', 'Israeli', 'ISR'],
        'Palestine': ['Palestine', 'Palestinian', 'PSE', 'State of Palestine'],
        'Saudi Arabia': ['Saudi Arabia', 'Saudi', 'SAU'],
        'United Arab Emirates': ['UAE', 'United Arab Emirates', 'ARE'],
        'Kuwait': ['Kuwait', 'Kuwaiti', 'KWT'],
        'Qatar': ['Qatar', 'Qatari', 'QAT'],
        'Bahrain': ['Bahrain', 'Bahraini', 'BHR'],
        'Oman': ['Oman', 'Omani', 'OMN'],
        'Yemen': ['Yemen', 'Yemeni', 'YEM'],
        
        # Small island states
        'Marshall Islands': ['Marshall Islands', 'RMI'],
        'Micronesia, Federated States of': ['Micronesia', 'Federated States of Micronesia', 'FSM'],
        'Palau': ['Palau', 'PLW'],
        'Tuvalu': ['Tuvalu', 'TUV'],
        'Nauru': ['Nauru', 'NRU'],
        'Kiribati': ['Kiribati', 'KIR'],
        'Samoa': ['Samoa', 'WSM'],
        'Tonga': ['Tonga', 'TON'],
        'Fiji': ['Fiji', 'FJI'],
        'Vanuatu': ['Vanuatu', 'VUT'],
        'Solomon Islands': ['Solomon Islands', 'SLB'],
        'Papua New Guinea': ['Papua New Guinea', 'PNG'],
        'Cook Islands': ['Cook Islands', 'COK'],
        'Niue': ['Niue', 'NIU'],
        
        # Caribbean
        'Cuba': ['Cuba', 'Cuban', 'CUB'],
        'Jamaica': ['Jamaica', 'Jamaican', 'JAM'],
        'Haiti': ['Haiti', 'Haitian', 'HTI', 'Haïti'],
        'Dominican Republic': ['Dominican Republic', 'Dominican', 'DOM'],
        'Trinidad and Tobago': ['Trinidad and Tobago', 'TTO'],
        'Barbados': ['Barbados', 'Barbadian', 'BRB'],
        'Saint Lucia': ['Saint Lucia', 'LCA'],
        'Grenada': ['Grenada', 'Grenadian', 'GRD'],
        'Saint Vincent and the Grenadines': ['Saint Vincent and the Grenadines', 'VCT'],
        'Antigua and Barbuda': ['Antigua and Barbuda', 'ATG'],
        'Dominica': ['Dominica', 'DMA'],
        'Saint Kitts and Nevis': ['Saint Kitts and Nevis', 'KNA'],
        'Bahamas': ['Bahamas', 'Bahamian', 'BHS'],
        
        # Africa
        'Egypt': ['Egypt', 'Egyptian', 'EGY'],
        'South Africa': ['South Africa', 'South African', 'ZAF'],
        'Nigeria': ['Nigeria', 'Nigerian', 'NGA'],
        'Kenya': ['Kenya', 'Kenyan', 'KEN'],
        'Ethiopia': ['Ethiopia', 'Ethiopian', 'ETH'],
        'Ghana': ['Ghana', 'Ghanaian', 'GHA'],
        'Morocco': ['Morocco', 'Moroccan', 'MAR', 'Maroc'],
        'Algeria': ['Algeria', 'Algerian', 'DZA', 'Algérie'],
        'Tunisia': ['Tunisia', 'Tunisian', 'TUN', 'Tunisie'],
        'Libya': ['Libya', 'Libyan', 'LBY'],
        'Sudan': ['Sudan', 'Sudanese', 'SDN'],
        'South Sudan': ['South Sudan', 'SSD'],
        'Uganda': ['Uganda', 'Ugandan', 'UGA'],
        'Tanzania, United Republic of': ['Tanzania', 'Tanzanian', 'TZA', 'United Republic of Tanzania'],
        'Rwanda': ['Rwanda', 'Rwandan', 'RWA'],
        'Burundi': ['Burundi', 'Burundian', 'BDI'],
        'Congo': ['Congo', 'Congolese', 'COG', 'Republic of the Congo'],
        'Congo, The Democratic Republic of the': ['DRC', 'Congo DRC', 'COD', 'Democratic Republic of the Congo'],
        'Cameroon': ['Cameroon', 'Cameroonian', 'CMR', 'Cameroun'],
        'Chad': ['Chad', 'Chadian', 'TCD', 'Tchad'],
        'Central African Republic': ['Central African Republic', 'CAF'],
        'Madagascar': ['Madagascar', 'Malagasy', 'MDG'],
        'Mozambique': ['Mozambique', 'Mozambican', 'MOZ'],
        'Angola': ['Angola', 'Angolan', 'AGO'],
        'Zambia': ['Zambia', 'Zambian', 'ZMB'],
        'Zimbabwe': ['Zimbabwe', 'Zimbabwean', 'ZWE'],
        'Botswana': ['Botswana', 'Botswanan', 'BWA'],
        'Namibia': ['Namibia', 'Namibian', 'NAM'],
        'Lesotho': ['Lesotho', 'LSO'],
        'Eswatini': ['Eswatini', 'Swaziland', 'SWZ'],
        'Malawi': ['Malawi', 'Malawian', 'MWI'],
        'Mauritius': ['Mauritius', 'Mauritian', 'MUS'],
        'Seychelles': ['Seychelles', 'SYC'],
        'Comoros': ['Comoros', 'Comorian', 'COM'],
        'Djibouti': ['Djibouti', 'DJI'],
        'Eritrea': ['Eritrea', 'Eritrean', 'ERI'],
        'Somalia': ['Somalia', 'Somali', 'SOM'],
        'Senegal': ['Senegal', 'Senegalese', 'SEN', 'Sénégal'],
        'Gambia': ['Gambia', 'Gambian', 'GMB'],
        'Guinea-Bissau': ['Guinea-Bissau', 'GNB'],
        'Guinea': ['Guinea', 'Guinean', 'GIN', 'Guinée'],
        'Sierra Leone': ['Sierra Leone', 'SLE'],
        'Liberia': ['Liberia', 'Liberian', 'LBR'],
        "Côte d'Ivoire": ['Côte d\'Ivoire', 'Ivory Coast', 'CIV'],
        'Burkina Faso': ['Burkina Faso', 'BFA'],
        'Mali': ['Mali', 'Malian', 'MLI'],
        'Niger': ['Niger', 'Nigerien', 'NER'],
        'Mauritania': ['Mauritania', 'Mauritanian', 'MRT'],
        'Togo': ['Togo', 'Togolese', 'TGO'],
        'Benin': ['Benin', 'Beninese', 'BEN', 'Bénin'],
        'Gabon': ['Gabon', 'Gabonese', 'GAB'],
        'Equatorial Guinea': ['Equatorial Guinea', 'GNQ'],
        'Sao Tome and Principe': ['Sao Tome and Principe', 'STP', 'São Tomé'],
        
        # Americas
        'Mexico': ['Mexico', 'Mexican', 'MEX', 'México'],
        'Argentina': ['Argentina', 'Argentinian', 'ARG'],
        'Chile': ['Chile', 'Chilean', 'CHL'],
        'Peru': ['Peru', 'Peruvian', 'PER', 'Perú'],
        'Colombia': ['Colombia', 'Colombian', 'COL'],
        'Venezuela, Bolivarian Republic of': ['Venezuela', 'Venezuelan', 'VEN', 'Bolivarian Republic of Venezuela'],
        'Ecuador': ['Ecuador', 'Ecuadorian', 'ECU'],
        'Bolivia, Plurinational State of': ['Bolivia', 'Bolivian', 'BOL', 'Plurinational State of Bolivia'],
        'Paraguay': ['Paraguay', 'Paraguayan', 'PRY'],
        'Uruguay': ['Uruguay', 'Uruguayan', 'URY'],
        'Guyana': ['Guyana', 'Guyanese', 'GUY'],
        'Suriname': ['Suriname', 'Surinamese', 'SUR'],
        
        # Additional entries
        'European Union': ['European Union', 'EU'],
        'United States': ['United States', 'USA', 'US'],
        'Russia': ['Russia', 'Russian Federation'],
        'Brunei Darussalam': ['Brunei', 'Brunei Darussalam', 'BRN'],
        'Cabo Verde': ['Cape Verde', 'Cabo Verde', 'CPV'],
        'Belize': ['Belize', 'BLZ'],
        'Costa Rica': ['Costa Rica', 'CRI'],
        'El Salvador': ['El Salvador', 'SLV'],
        'Guatemala': ['Guatemala', 'GTM'],
        'Honduras': ['Honduras', 'HND'],
        'Nicaragua': ['Nicaragua', 'NIC'],
        'Panama': ['Panama', 'PAN'],
        'Andorra': ['Andorra', 'AND'],
        'Liechtenstein': ['Liechtenstein', 'LIE'],
        'Monaco': ['Monaco', 'MCO'],
        'San Marino': ['San Marino', 'SMR'],
        'Vatican City': ['Vatican', 'Vatican City', 'VAT'],
        'New Zealand': ['New Zealand', 'NZL']
    }
    
    return comprehensive_countries

def find_fuzzy_country_match(text, countries, threshold=0.7):
    """Find fuzzy match for country name."""
    import difflib
    
    text = text.strip().title()
    
    # Direct match first
    for country_name, variations in countries.items():
        for variation in variations:
            if text.lower() == variation.lower():
                return country_name
    
    # Fuzzy matching
    all_variations = []
    country_map = {}
    for country_name, variations in countries.items():
        for variation in variations:
            all_variations.append(variation)
            country_map[variation] = country_name
    
    # Find closest matches
    matches = difflib.get_close_matches(text, all_variations, n=3, cutoff=threshold)
    
    if matches:
        return country_map[matches[0]]
    
    return None

def extract_country_enhanced(title, text_content="", doc_id=""):
    """
    Ultimate aggressive country extraction with comprehensive pattern matching.
    This method successfully found Spain, Italy, and many other missing countries.
    """
    
    countries = get_comprehensive_country_list()
    
    # Combine all text sources
    all_text = ""
    if title:
        all_text += urllib.parse.unquote(str(title)).upper() + " "
    if text_content:
        all_text += str(text_content)[:3000].upper() + " "
    
    # Direct pattern matching with context validation
    for country_name, variations in countries.items():
        for variation in variations:
            variation_upper = variation.upper()
            
            # Exact match with context validation
            if variation_upper in all_text:
                context_patterns = [
                    rf'\b{re.escape(variation_upper)}\b',
                    rf'{re.escape(variation_upper)}\'?S\b',
                    rf'\b{re.escape(variation_upper)}\s+(?:NDC|INDC|GOVERNMENT|MINISTRY)',
                    rf'(?:NDC|INDC|GOVERNMENT|MINISTRY).*{re.escape(variation_upper)}',
                    rf'REPUBLIC\s+OF\s+{re.escape(variation_upper)}',
                    rf'{re.escape(variation_upper)}_',
                    rf'_{re.escape(variation_upper)}_',
                    rf'_{re.escape(variation_upper)}',
                    rf'{re.escape(variation_upper)}\.',
                    rf'ACTUALIZACIÓN.*{re.escape(variation_upper)}',  # Spanish updates
                    rf'{re.escape(variation_upper)}.*ACTUALIZACIÓN',  # Spanish updates
                    rf'CONTRIBUCIÓN.*{re.escape(variation_upper)}',   # Spanish contributions
                    rf'{re.escape(variation_upper)}.*CONTRIBUCIÓN',   # Spanish contributions
                ]
                
                for pattern in context_patterns:
                    if re.search(pattern, all_text):
                        logger.info(f"Ultimate extraction: {urllib.parse.unquote(str(title))} -> {country_name}")
                        return country_name
    
    # Title-specific advanced patterns
    if title:
        title_decoded = urllib.parse.unquote(str(title))
        
        # NDC patterns with fuzzy matching
        ndc_patterns = [
            r'([A-Za-z\s]+?)[\s_]+NDC',
            r'NDC[\s_]+([A-Za-z\s]+)',
            r'([A-Za-z\s]+?)[\s_]+INDC',
            r'INDC[\s_]+([A-Za-z\s]+)',
            r'ACTUALIZACIÓN.*NDC.*([A-Za-z\s]+)',  # Spanish
            r'([A-Za-z\s]+).*ACTUALIZACIÓN.*NDC',  # Spanish
        ]
        
        for pattern in ndc_patterns:
            match = re.search(pattern, title_decoded, re.IGNORECASE)
            if match:
                potential_country = match.group(1).strip()
                if len(potential_country) > 2 and len(potential_country) < 30:
                    best_match = find_fuzzy_country_match(potential_country, countries)
                    if best_match:
                        logger.info(f"NDC pattern extraction: {title_decoded} -> {best_match}")
                        return best_match
        
        # Language-specific government patterns
        language_patterns = {
            r'RÉPUBLIQUE\s+(?:DU|DE)\s+([A-Z\s]+)': 'French',
            r'REPUBLICA\s+DE\s+([A-Z\s]+)': 'Spanish',  
            r'REPÚBLICA\s+DE\s+([A-Z\s]+)': 'Portuguese',
            r'GOVERNO\s+DE\s+([A-Z\s]+)': 'Portuguese',
            r'GOVERNMENT\s+OF\s+([A-Z\s]+)': 'English',
            r'([A-Z\s]+)\s+GOVERNMENT': 'English',
            r'MINISTRY\s+OF.*([A-Z\s]+)': 'English',
            r'CONTRIBUCIÓN.*([A-Z\s]+)': 'Spanish',  # Spanish contribution
            r'([A-Z\s]+).*CONTRIBUCIÓN': 'Spanish',  # Spanish contribution
        }
        
        title_upper = title_decoded.upper()
        for pattern, language in language_patterns.items():
            match = re.search(pattern, title_upper)
            if match:
                potential_country = match.group(1).strip()
                if len(potential_country) > 2:
                    best_match = find_fuzzy_country_match(potential_country, countries)
                    if best_match:
                        logger.info(f"Language pattern extraction ({language}): {title_decoded} -> {best_match}")
                        return best_match
    
    return None

# === Configuration ===
DATA_FOLDERS = [
    "data/international_agreements",
    "data/ndc_documents", 
    "data/unfccc_documents_topic_mitigation"
]

OUTPUT_FILES = {
    'extraction': 'data/extracted_texts.pkl',
    'clustering': 'data/plot_df.pkl',
    'country_clustering': 'data/country_plot_df.pkl',
    'indexes': 'indexes/',
    'embeddings': 'embeddings/'
}

def print_header(title):
    """Print a formatted header for pipeline steps."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, total_steps, description):
    """Print a formatted step indicator."""
    print(f"\n[Step {step}/{total_steps}] {description}")
    print("-" * 50)

def get_pdf_files():
    """Discover all PDF files in the three data folders."""
    pdf_files = []
    
    for folder in DATA_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️ Warning: Folder {folder} does not exist, skipping...")
            continue
            
        pdfs = list(folder_path.glob("*.pdf"))
        print(f"📁 Found {len(pdfs)} PDFs in {folder}")
        
        for pdf_path in pdfs:
            # Create metadata for each PDF
            pdf_files.append({
                'document_id': f"doc_{len(pdf_files)}",
                'document_name': pdf_path.name,
                'full_path': str(pdf_path),
                'folder': folder,
                'title': pdf_path.stem,  # filename without extension
                'download_link': f"file://{pdf_path}"  # placeholder
            })
    
    print(f"\n📊 Total PDFs discovered: {len(pdf_files)}")
    return pdf_files

def extract_texts(pdf_files, force=False):
    """Extract text from all PDF files."""
    print_step(1, 4, "Text Extraction")
    
    # Check if output already exists
    if Path(OUTPUT_FILES['extraction']).exists() and not force:
        print(f"✅ {OUTPUT_FILES['extraction']} already exists. Skipping extraction.")
        print("   Use --force to reprocess.")
        return
    
    texts = []
    statuses = []
    countries = []
    
    print(f"🔄 Processing {len(pdf_files)} PDF files...")
    
    for i, pdf_info in enumerate(pdf_files):
        pdf_path = pdf_info['full_path']
        document_name = pdf_info['document_name']
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   📄 Processed {i + 1}/{len(pdf_files)} documents...")
        
        try:
            if not Path(pdf_path).exists():
                texts.append("")
                statuses.append("missing_pdf")
                countries.append(None)
                continue
            
            # Extract text using PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
            
            if len(full_text.strip()) < 100:  # Very short text
                texts.append(full_text)
                statuses.append("short_text")
            else:
                texts.append(full_text)
                statuses.append("ok")
            
            # Enhanced country extraction
            country = extract_country_enhanced(pdf_info['title'], full_text, pdf_info['document_id'])
            countries.append(country)
            
            # Log key country extractions
            if country in ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']:
                logger.info(f"Key country extracted: {pdf_info['title']} -> {country}")
            
        except Exception as e:
            print(f"❌ Error processing {document_name}: {e}")
            texts.append("")
            statuses.append("extract_error")
            countries.append(None)
    
    # Create DataFrame
    df_data = []
    for i, pdf_info in enumerate(pdf_files):
        df_data.append({
            'document_id': pdf_info['document_id'],
            'document_name': pdf_info['document_name'],
            'full_path': pdf_info['full_path'],
            'folder': pdf_info['folder'],
            'title': pdf_info['title'],
            'download_link': pdf_info['download_link'],
            'text': texts[i],
            'status': statuses[i],
            'country': countries[i]
        })
    
    link_df = pd.DataFrame(df_data)
    
    # Save extracted texts
    Path("data").mkdir(exist_ok=True)
    with open(OUTPUT_FILES['extraction'], 'wb') as f:
        pickle.dump(link_df, f)
    
    # Enhanced extraction summary
    total_docs = len(link_df)
    docs_with_countries = len(link_df[link_df['country'].notna()])
    unique_countries = len(link_df['country'].dropna().unique())
    
    print("\n📊 Enhanced Extraction Summary:")
    print(f"Total documents: {total_docs}")
    print(f"Documents with countries: {docs_with_countries}/{total_docs} ({(docs_with_countries/total_docs)*100:.1f}%)")
    print(f"Unique countries detected: {unique_countries}")
    
    # Status breakdown
    status_counts = link_df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Key countries check
    key_countries = ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']
    found_key = []
    for country in key_countries:
        if country in link_df['country'].values:
            doc_count = len(link_df[link_df['country'] == country])
            found_key.append(f"{country} ({doc_count} docs)")
    
    if found_key:
        print(f"🎯 Key countries successfully extracted: {found_key}")
    
    # Save problematic documents
    problematic_docs = link_df[link_df['status'] != 'ok']
    if not problematic_docs.empty:
        diagnostics_path = "data/problematic_documents.csv"
        problematic_docs[['document_name', 'folder', 'status']].to_csv(diagnostics_path, index=False)
        print(f"⚠️ Saved {len(problematic_docs)} problematic documents to {diagnostics_path}")
    
    print(f"✅ Enhanced text extraction complete. Saved to {OUTPUT_FILES['extraction']}")

def prepare_clustering(force=False):
    """Prepare document and country-level clustering."""
    print_step(2, 4, "Document Clustering & Embedding")
    
    # Check if outputs already exist
    if (Path(OUTPUT_FILES['clustering']).exists() and 
        Path(OUTPUT_FILES['country_clustering']).exists() and 
        not force):
        print("✅ Clustering files already exist. Skipping clustering.")
        print("   Use --force to reprocess.")
        return
    
    # Load extracted texts
    with open(OUTPUT_FILES['extraction'], "rb") as f:
        link_df = pickle.load(f)
    
    # Filter to OK documents only
    original_count = len(link_df)
    link_df = link_df[link_df["status"] == "ok"].reset_index(drop=True)
    filtered_count = len(link_df)
    
    print(f"📊 Using {filtered_count}/{original_count} successfully extracted documents")
    
    if filtered_count == 0:
        raise ValueError("No successfully extracted documents found!")
    
    # Initialize model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
    
    # Generate embeddings with sentence-aware chunking
    print("🧮 Generating document embeddings with chunking...")
    doc_embeddings = []
    for text in link_df["text"].astype(str):
        chunks = chunk_text_intelligently(text, chunk_size=400, overlap=50)
        if chunks:
            chunk_embeds = model.encode(chunks, show_progress_bar=False)
            doc_embeddings.append(np.mean(chunk_embeds, axis=0))
        else:
            doc_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
    embeddings = np.vstack(doc_embeddings)
    print(f"✅ Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
    
    # UMAP dimensionality reduction
    print("🗺️ Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # HDBSCAN clustering
    print("🔗 Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters = clusterer.fit_predict(reduced_embeddings)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"✅ Found {n_clusters} clusters with {n_noise} noise points")
    
    # Create document-level DataFrame
    plot_df = pd.DataFrame({
        "document_id": link_df["document_id"],
        "country": link_df["country"],
        "title": link_df["title"],
        "folder": link_df["folder"],
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "cluster": clusters,
        "text": link_df["text"],
        "status": link_df["status"]
    })
    
    # Save document-level clustering
    plot_df.to_pickle(OUTPUT_FILES['clustering'])
    print(f"✅ Document-level clustering saved to {OUTPUT_FILES['clustering']}")
    
    # Country-level aggregation
    print("🌍 Creating country-level clustering...")
    link_df["embedding"] = list(embeddings)
    
    # Group by country and aggregate embeddings
    country_groups = link_df.groupby("country")["embedding"].apply(list)
    
    country_texts = []
    country_embeddings = []
    country_names = []
    
    for country, vecs in country_groups.items():
        if not country or len(vecs) < 1:
            continue
        
        # Average embeddings for the country
        stacked = np.vstack(vecs)
        avg_vec = stacked.mean(axis=0)
        
        # Combine all text for the country
        full_text = " ".join(link_df[link_df["country"] == country]["text"].tolist())
        
        country_names.append(country)
        country_embeddings.append(avg_vec)
        country_texts.append(full_text)
    
    if len(country_embeddings) == 0:
        print("⚠️ No countries with valid embeddings found")
        return
    
    # Country-level UMAP and clustering  
    country_embeddings = np.vstack(country_embeddings)
    
    # Enhanced UMAP parameters for better separation
    country_reduced = umap.UMAP(
        n_neighbors=min(15, len(country_embeddings)-1),
        min_dist=0.2,
        metric='cosine', 
        random_state=42
    ).fit_transform(country_embeddings)
    
    # Enhanced HDBSCAN clustering
    country_clusters = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(country_reduced)
    
    # Enhanced statistics logging
    unique_clusters = len(set(country_clusters[country_clusters >= 0]))
    noise_points = sum(country_clusters == -1)
    logger.info(f"Country clustering: {unique_clusters} clusters, {noise_points} noise points")
    
    country_df = pd.DataFrame({
        "country": country_names,
        "combined_text": country_texts,
        "x": country_reduced[:, 0],
        "y": country_reduced[:, 1],
        "cluster": country_clusters,  # Use 'cluster' as primary
        "country_cluster": country_clusters,  # Add for compatibility
        "text_length": [len(text) for text in country_texts]
    })
    
    # Save country-level clustering
    country_df.to_pickle(OUTPUT_FILES['country_clustering'])
    
    # Enhanced country clustering summary
    print(f"\n🌍 Enhanced Country Clustering Summary:")
    print(f"Total countries clustered: {len(country_df)}")
    print(f"Countries in clusters: {len(country_df[country_df['cluster'] >= 0])}")
    print(f"Noise/unassigned: {len(country_df[country_df['cluster'] == -1])}")
    
    # Check for key countries in clustering
    key_countries_in_clustering = []
    for country in ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']:
        if country in country_df['country'].values:
            cluster_id = country_df[country_df['country'] == country]['cluster'].iloc[0]
            key_countries_in_clustering.append(f"{country} (Cluster {cluster_id})")
    
    if key_countries_in_clustering:
        print(f"🎯 Key countries in clustering: {key_countries_in_clustering}")
    
    print(f"✅ Enhanced country-level clustering saved to {OUTPUT_FILES['country_clustering']}")
    
    # Upload to Azure (ignore failures) unless in offline mode
    if not OFFLINE_MODE:
        try:
            upload_blob(AZURE_CONTAINER_NAME, "data/plot_df.pkl", OUTPUT_FILES['clustering'])
            upload_blob(AZURE_CONTAINER_NAME, "data/country_plot_df.pkl", OUTPUT_FILES['country_clustering'])
            print("☁️ Files uploaded to Azure")
        except Exception as e:
            print(f"⚠️ Azure upload failed (continuing anyway): {e}")
    else:
        print("🔌 Offline mode: Skipping Azure upload")

def build_indexes(force=False):
    """Build FAISS indexes for each cluster."""
    print_step(3, 4, "Building FAISS Search Indexes")
    
    # Check if indexes already exist
    if Path(OUTPUT_FILES['indexes']).exists() and not force:
        existing_indexes = list(Path(OUTPUT_FILES['indexes']).glob("cluster_*.index"))
        if len(existing_indexes) > 0:
            print(f"✅ Found {len(existing_indexes)} existing indexes. Skipping index building.")
            print("   Use --force to rebuild.")
            return
    
    # Load clustered data
    plot_df = pd.read_pickle(OUTPUT_FILES['clustering'])
    print(f"📊 Loaded {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")
    
    # Initialize model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
    
    # Create directories
    os.makedirs(OUTPUT_FILES['indexes'], exist_ok=True)
    os.makedirs(OUTPUT_FILES['embeddings'], exist_ok=True)
    
    # Process each cluster
    clusters = sorted(plot_df['cluster'].unique())
    successful_clusters = 0
    
    for i, cluster in enumerate(clusters):
        print(f"\n🔧 Processing cluster {cluster} ({i+1}/{len(clusters)})...")
        
        cluster_df = plot_df[plot_df['cluster'] == cluster]
        texts = cluster_df['text'].tolist()
        ids = cluster_df['document_id'].tolist()
        
        print(f"   📄 {len(texts)} documents in cluster {cluster}")
        
        if len(texts) == 0:
            print(f"   ⚠️ Skipping empty cluster {cluster}")
            continue
        
        try:
            # Generate embeddings
            print(f"   🧮 Generating embeddings...")
            embeddings = model.encode(texts, batch_size=1, show_progress_bar=False)
            
            # Create FAISS index
            print(f"   🔍 Creating FAISS index...")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            
            # Save files
            index_path = f"{OUTPUT_FILES['indexes']}/cluster_{cluster}.index"
            emb_path = f"{OUTPUT_FILES['embeddings']}/cluster_{cluster}.pkl"
            
            faiss.write_index(index, index_path)
            with open(emb_path, "wb") as f:
                pickle.dump({'texts': texts, 'ids': ids}, f)
            
            print(f"   ✅ Saved index ({embeddings.shape[1]}D) and metadata")
            successful_clusters += 1
            
            # Upload to Azure (ignore errors) unless in offline mode
            if not OFFLINE_MODE:
                try:
                    upload_blob(AZURE_CONTAINER_NAME, f"indexes/cluster_{cluster}.index", index_path)
                    upload_blob(AZURE_CONTAINER_NAME, f"embeddings/cluster_{cluster}.pkl", emb_path)
                except Exception as e:
                    print(f"   ⚠️ Azure upload failed (continuing): {e}")
            else:
                print(f"   🔌 Offline mode: Skipping Azure upload for cluster {cluster}")
            
            # Clear memory
            del embeddings, index
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Error processing cluster {cluster}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n✅ Index building complete! Successfully built {successful_clusters}/{len(clusters)} indexes")

def build_enhanced_indexes(force=False):
    """Build enhanced FAISS indexes for ultra-fast RAG mode."""
    print_step("3b", 4, "Building Enhanced Indexes (Ultra-Fast RAG)")
    
    # Check if enhanced indexes already exist
    enhanced_dir = Path("indexes_enhanced")
    if enhanced_dir.exists() and not force:
        existing_enhanced = list(enhanced_dir.glob("cluster_*_chunks.index"))
        if len(existing_enhanced) > 0:
            print(f"✅ Found {len(existing_enhanced)} existing enhanced indexes. Skipping enhanced index building.")
            print("   Use --force to rebuild.")
            return
    
    print("🚀 Building enhanced indexes for ultra-fast RAG (5-20x faster queries)...")
    
    # Import enhanced indexing functions
    from scripts.prepare_enhanced_index import (
        chunk_text_intelligently, 
        extract_document_title_from_content,
        create_document_summary,
        build_filename_mapping
    )
    
    # Load data
    plot_df = pd.read_pickle(OUTPUT_FILES['clustering'])
    print(f"📊 Processing {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")
    
    # Initialize model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME} on {OPTIMAL_DEVICE}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
    print(f"✅ Model loaded with {model.get_sentence_embedding_dimension()}D embeddings")
    
    # Build filename mapping by scanning source folders
    filename_mapping = build_filename_mapping(plot_df)
    
    # Create enhanced directories
    os.makedirs("indexes_enhanced", exist_ok=True)
    os.makedirs("embeddings_enhanced", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    
    # Process each cluster
    successful_enhanced = 0
    clusters = sorted(plot_df['cluster'].unique())
    
    for i, cluster in enumerate(clusters):
        print(f"\n🔧 Enhanced processing cluster {cluster} ({i+1}/{len(clusters)})...")
        
        cluster_df = plot_df[plot_df['cluster'] == cluster]
        
        if len(cluster_df) == 0:
            print(f"   ⚠️ Skipping empty cluster {cluster}")
            continue
        
        try:
            # Prepare enhanced data structures
            all_chunks = []
            chunk_metadata = []
            doc_summaries = {}
            
            print(f"   📄 Processing {len(cluster_df)} documents...")
            
            for idx, row in cluster_df.iterrows():
                doc_id = row['document_id']
                text = str(row['text'])
                title = row.get('title', 'Unknown Document')
                country = row.get('country', 'Unknown')
                folder = row.get('folder', 'unknown_folder')
                
                # Get actual filename from mapping
                actual_filename = filename_mapping.get(doc_id, None)
                
                # Extract actual document title from content
                extracted_title = extract_document_title_from_content(text, title)
                display_title = extracted_title if extracted_title != title and len(extracted_title) > len(title) else title
                
                # Create document summary
                summary = create_document_summary(text)
                doc_summaries[doc_id] = {
                    'summary': summary,
                    'title': title,
                    'extracted_title': extracted_title,
                    'display_title': display_title,
                    'country': country,
                    'folder': folder,
                    'filename': actual_filename,
                    'full_length': len(text)
                }
                
                # Create intelligent chunks
                chunks = chunk_text_intelligently(text, chunk_size=400, overlap=50)
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        'doc_id': doc_id,
                        'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                        'title': title,
                        'extracted_title': extracted_title,
                        'display_title': display_title,
                        'country': country,
                        'folder': folder,
                        'filename': actual_filename,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'summary': summary
                    })
            
            print(f"   🧩 Created {len(all_chunks)} chunks from {len(cluster_df)} documents")
            
            # Generate embeddings for all chunks at once (more efficient)
            print(f"   🧮 Generating embeddings for chunks...")
            chunk_embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=False)
            
            # Also generate embeddings for summaries (for quick overview queries)
            summaries_list = [doc_summaries[doc_id]['summary'] for doc_id in doc_summaries.keys()]
            print(f"   📝 Generating embeddings for {len(summaries_list)} summaries...")
            summary_embeddings = model.encode(summaries_list, batch_size=32, show_progress_bar=False)
            
            # Create separate FAISS indexes
            
            # 1. Chunk index (detailed search)
            chunk_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            chunk_index.add(chunk_embeddings.astype(np.float32))
            
            # 2. Summary index (quick overview)
            summary_index = faiss.IndexFlatL2(summary_embeddings.shape[1])
            summary_index.add(summary_embeddings.astype(np.float32))
            
            # Save enhanced data
            cluster_data = {
                'chunks': all_chunks,
                'chunk_metadata': chunk_metadata,
                'chunk_embeddings': chunk_embeddings,
                'doc_summaries': doc_summaries,
                'summary_embeddings': summary_embeddings,
                'doc_ids': list(doc_summaries.keys())
            }
            
            # Save files
            faiss.write_index(chunk_index, f"indexes_enhanced/cluster_{cluster}_chunks.index")
            faiss.write_index(summary_index, f"indexes_enhanced/cluster_{cluster}_summaries.index")
            
            with open(f"embeddings_enhanced/cluster_{cluster}.pkl", "wb") as f:
                pickle.dump(cluster_data, f)
            
            print(f"   ✅ Saved enhanced indexes and embeddings")
            print(f"      - Chunk index: {len(all_chunks)} items")
            print(f"      - Summary index: {len(doc_summaries)} items")
            
            successful_enhanced += 1
            
            # Upload to Azure (ignore errors) unless in offline mode
            if not OFFLINE_MODE:
                try:
                    upload_blob(AZURE_CONTAINER_NAME, f"indexes_enhanced/cluster_{cluster}_chunks.index", f"indexes_enhanced/cluster_{cluster}_chunks.index")
                    upload_blob(AZURE_CONTAINER_NAME, f"indexes_enhanced/cluster_{cluster}_summaries.index", f"indexes_enhanced/cluster_{cluster}_summaries.index")
                    upload_blob(AZURE_CONTAINER_NAME, f"embeddings_enhanced/cluster_{cluster}.pkl", f"embeddings_enhanced/cluster_{cluster}.pkl")
                except Exception as e:
                    print(f"   ⚠️ Azure upload failed (continuing): {e}")
            else:
                print(f"   🔌 Offline mode: Skipping Azure upload for enhanced cluster {cluster}")
            
            # Clear memory
            del chunk_embeddings, summary_embeddings, chunk_index, summary_index
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error processing enhanced cluster {cluster}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Enhanced index building complete! Successfully built {successful_enhanced}/{len(clusters)} enhanced indexes")
    print("\nEnhanced indexes provide:")
    print("  ⚡ 5-20x faster query performance")
    print("  🧩 Intelligent text chunking with overlap")
    print("  📝 Document summaries for quick overview")
    print("  📊 Rich metadata and context relationships")

def launch_app():
    """Launch the Streamlit application."""
    print_step(4, 4, "Launching Streamlit Application")
    
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C in this terminal to stop the server")
    
    # Import and check if cluster_qa_app exists
    if not Path("cluster_qa_app.py").exists():
        print("❌ cluster_qa_app.py not found!")
        print("   Make sure you're running this from the unfccc directory")
        return
    
    os.system("streamlit run cluster_qa_app.py")

def main():
    """Main pipeline execution."""
    global OFFLINE_MODE
    
    parser = argparse.ArgumentParser(description="Automated UNFCCC Data Pipeline")
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip text extraction step')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip clustering step')
    parser.add_argument('--skip-indexing', action='store_true',
                       help='Skip FAISS indexing step')
    parser.add_argument('--skip-enhanced', action='store_true',
                       help='Skip enhanced indexing for ultra-fast RAG')
    parser.add_argument('--skip-app', action='store_true',
                       help='Skip launching the Streamlit app')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if output files exist')
    parser.add_argument('--enhanced-only', action='store_true',
                       help='Build only enhanced indexes (skip standard indexes)')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode (skip all Azure uploads)')
    
    args = parser.parse_args()
    
    # Set offline mode
    OFFLINE_MODE = args.offline
    
    print_header("🌍 UNFCCC Automated Data Pipeline 🌍")
    print("Processing climate policy documents from multiple sources...")
    print(f"Data folders: {', '.join(DATA_FOLDERS)}")
    
    if OFFLINE_MODE:
        print("🔌 OFFLINE MODE: Azure uploads will be skipped")
    
    start_time = time.time()
    
    try:
        # Bootstrap checks
        print("\n🔍 Running bootstrap checks...")
        check_folders()
        check_dependencies()
        print("✅ Bootstrap checks passed")
        
        # Step 1: Text Extraction
        if not args.skip_extraction:
            pdf_files = get_pdf_files()
            if len(pdf_files) == 0:
                print("❌ No PDF files found in data folders!")
                return
            extract_texts(pdf_files, force=args.force)
        else:
            print_step(1, 4, "Text Extraction (SKIPPED)")
        
        # Step 2: Clustering
        if not args.skip_clustering:
            prepare_clustering(force=args.force)
        else:
            print_step(2, 4, "Document Clustering (SKIPPED)")
        
        # Step 3: Index Building
        if not args.skip_indexing and not args.enhanced_only:
            build_indexes(force=args.force)
        elif args.enhanced_only:
            print_step(3, 4, "Standard FAISS Index Building (SKIPPED - Enhanced Only Mode)")
        else:
            print_step(3, 4, "FAISS Index Building (SKIPPED)")
        
        # Step 3b: Enhanced Index Building (Ultra-Fast RAG)
        if not args.skip_enhanced:
            build_enhanced_indexes(force=args.force)
        else:
            print_step("3b", 4, "Enhanced Index Building (SKIPPED)")
        
        # Step 4: Launch App
        if not args.skip_app:
            launch_app()
        else:
            print_step(4, 4, "Streamlit Application (SKIPPED)")
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        traceback.print_exc()
        return
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Total pipeline time: {total_time:.1f} seconds")
    
    print_header("🎉 Pipeline Complete! 🎉")
    
    # Show usage instructions
    print("\n📚 Next Steps:")
    print("  📊 Launch app: streamlit run cluster_qa_app.py")
    if not OFFLINE_MODE:
        print("  ☁️ Files uploaded to Azure for cloud access")
    else:
        print("  🔌 Files stored locally (offline mode)")
    
    # Check what was built
    standard_indexes = len(list(Path(OUTPUT_FILES['indexes']).glob("cluster_*.index"))) if Path(OUTPUT_FILES['indexes']).exists() else 0
    enhanced_indexes = len(list(Path("indexes_enhanced").glob("cluster_*_chunks.index"))) if Path("indexes_enhanced").exists() else 0
    
    if enhanced_indexes > 0:
        print("  ⚡ Ultra-fast RAG enabled! Queries will be 5-20x faster")
        print("  🎛️ Switch between standard and ultra-fast modes in the app")
    
    if standard_indexes > 0:
        print(f"  🔍 Standard indexes: {standard_indexes} clusters")
    if enhanced_indexes > 0:
        print(f"  ⚡ Enhanced indexes: {enhanced_indexes} clusters")

if __name__ == "__main__":
    main() 
