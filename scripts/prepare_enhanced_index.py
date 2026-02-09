# prepare_enhanced_index.py
# Enhanced preprocessing for ultra-fast context generation

import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import gc
import numpy as np
import torch
from pathlib import Path
from config import EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies
import glob
from core.pipeline.embedding_store import EmbeddingStore
import streamlit as st

def chunk_text_intelligently(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks with sentence boundaries."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Finalize current chunk
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append(chunk_text.strip())
    
    return chunks

def extract_document_title_from_content(text, fallback_title="Unknown Document"):
    """
    Extract the actual document title from PDF content by looking for common patterns.
    """
    if not text or len(text) < 100:
        return fallback_title
    
    # Take first 2000 characters where titles are usually found
    header_text = text[:2000].upper()
    
    # Common UNFCCC document title patterns
    title_patterns = [
        # National Communications patterns
        r'NATIONAL COMMUNICATION[S]?\s+OF\s+([A-Z\s]+?)(?:\s+TO|\s+UNDER|\s+\d|\n)',
        r'([A-Z\s]+?)\s+NATIONAL COMMUNICATION',
        r'INITIAL NATIONAL COMMUNICATION\s+OF\s+([A-Z\s]+)',
        r'SECOND NATIONAL COMMUNICATION\s+OF\s+([A-Z\s]+)',
        r'THIRD NATIONAL COMMUNICATION\s+OF\s+([A-Z\s]+)',
        r'FOURTH NATIONAL COMMUNICATION\s+OF\s+([A-Z\s]+)',
        
        # NDC patterns  
        r'NATIONALLY DETERMINED CONTRIBUTION[S]?\s+OF\s+([A-Z\s]+)',
        r'([A-Z\s]+?)\s+NATIONALLY DETERMINED CONTRIBUTION',
        r'UPDATED?\s+NDC\s+OF\s+([A-Z\s]+)',
        r'([A-Z\s]+?)\s+UPDATED?\s+NDC',
        
        # Biennial reports
        r'BIENNIAL REPORT\s+OF\s+([A-Z\s]+)',
        r'([A-Z\s]+?)\s+BIENNIAL REPORT',
        r'BIENNIAL UPDATE REPORT\s+OF\s+([A-Z\s]+)',
        
        # Technical reports
        r'TECHNICAL REVIEW\s+OF\s+([A-Z\s]+)',
        r'IN-DEPTH REVIEW\s+OF\s+([A-Z\s]+)',
        r'REPORT ON THE IN-DEPTH REVIEW\s+OF[^A-Z]*([A-Z\s]+)',
        
        # General country report patterns
        r'^([A-Z]{3,}(?:\s+[A-Z]{3,})*)\s*(?:REPORT|COMMUNICATION|SUBMISSION)',
        r'REPORT\s+(?:FROM|OF)\s+([A-Z\s]+?)(?:\s+TO|\s+UNDER|\s+\d|\n)',
    ]
    
    # Try each pattern
    import re
    for pattern in title_patterns:
        match = re.search(pattern, header_text)
        if match:
            country_or_title = match.group(1).strip()
            if len(country_or_title) > 3 and len(country_or_title) < 50:
                # Clean up the extracted title
                cleaned = country_or_title.title().replace('  ', ' ')
                # Add context based on pattern type
                if 'NATIONAL COMMUNICATION' in header_text:
                    return f"{cleaned} - National Communication"
                elif 'NDC' in header_text or 'NATIONALLY DETERMINED' in header_text:
                    return f"{cleaned} - Nationally Determined Contribution"
                elif 'BIENNIAL' in header_text:
                    return f"{cleaned} - Biennial Report"
                elif 'TECHNICAL REVIEW' in header_text or 'IN-DEPTH REVIEW' in header_text:
                    return f"{cleaned} - Technical Review Report"
                else:
                    return cleaned
    
    # Look for any document title in first few lines
    lines = text.split('\n')[:20]  # First 20 lines
    for line in lines:
        line = line.strip()
        if len(line) > 20 and len(line) < 200:
            # Check if this looks like a title (mostly uppercase, contains key words)
            upper_line = line.upper()
            if any(keyword in upper_line for keyword in [
                'REPORT', 'COMMUNICATION', 'SUBMISSION', 'CONTRIBUTION', 
                'REVIEW', 'ASSESSMENT', 'STRATEGY', 'PLAN', 'POLICY'
            ]):
                # This might be a title
                cleaned_title = line.title().strip()
                if len(cleaned_title) > 10:
                    return cleaned_title
    
    # Fallback: look for country names in first paragraph
    country_names = [
        'AFGHANISTAN', 'ALBANIA', 'ANDORRA', 'ARGENTINA', 'AUSTRALIA', 'AUSTRIA',
        'BANGLADESH', 'BELGIUM', 'BOLIVIA', 'BRAZIL', 'BULGARIA', 'CANADA',
        'CHILE', 'CHINA', 'COLOMBIA', 'CROATIA', 'CUBA', 'CZECH REPUBLIC',
        'DENMARK', 'EGYPT', 'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY',
        'GREECE', 'GUATEMALA', 'HUNGARY', 'ICELAND', 'INDIA', 'INDONESIA',
        'IRAN', 'IRELAND', 'ISRAEL', 'ITALY', 'JAPAN', 'KAZAKHSTAN',
        'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MEXICO', 'MONGOLIA', 'NETHERLANDS',
        'NEW ZEALAND', 'NORWAY', 'PAKISTAN', 'PERU', 'POLAND', 'PORTUGAL',
        'ROMANIA', 'RUSSIA', 'SLOVAKIA', 'SLOVENIA', 'SOUTH AFRICA', 'SPAIN',
        'SWEDEN', 'SWITZERLAND', 'TURKEY', 'UKRAINE', 'UNITED KINGDOM', 'UNITED STATES'
    ]
    
    for country in country_names:
        if country in header_text:
            return f"{country.title()} - Climate Report"
    
    return fallback_title

def create_document_summary(text, max_length=200):
    """Create a brief summary of the document (first meaningful paragraph)."""
    # Simple extraction of first substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    
    if paragraphs:
        summary = paragraphs[0]
        if len(summary) > max_length:
            # Truncate at sentence boundary
            sentences = summary.split('. ')
            summary = '. '.join(sentences[:2]) + '.'
        return summary
    
    # Fallback: first 200 chars
    return text[:max_length] + "..." if len(text) > max_length else text

def discover_actual_filenames(folder_path, doc_id, title):
    """
    Discover the actual filename by scanning the source folder for PDF files.
    Try to match based on document ID, title, or content patterns.
    """
    if not folder_path or folder_path == 'unknown_folder':
        return None
    
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            return None
        
        # Get all PDF files in the folder
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if not pdf_files:
            return None
        
        # Extract just the filenames for matching
        filenames = [os.path.basename(pdf) for pdf in pdf_files]
        
        # Try different matching strategies
        
        # Strategy 1: Direct title match (e.g., title="can02" -> "can02.pdf")
        if title and title != 'Unknown Document':
            for filename in filenames:
                if filename.lower() == f"{title.lower()}.pdf":
                    return filename
                # Also try without extension matching
                base_name = os.path.splitext(filename)[0]
                if base_name.lower() == title.lower():
                    return filename
        
        # Strategy 2: Look for doc_id pattern in filename (e.g., doc_259 might match something)
        if doc_id and doc_id.startswith('doc_'):
            doc_number = doc_id.replace('doc_', '')
            for filename in filenames:
                if doc_number in filename:
                    return filename
        
        # Strategy 3: If we have a small number of files, we might need manual mapping
        # For now, return None if no clear match
        return None
        
    except Exception as e:
        print(f"   ⚠️ Error discovering filename for {doc_id}: {e}")
        return None

def build_filename_mapping(plot_df):
    """
    Build a comprehensive mapping of document IDs to actual filenames.
    """
    print("🔍 Building filename mapping from source folders...")
    
    filename_mapping = {}
    unique_folders = plot_df['folder'].dropna().unique()
    
    for folder in unique_folders:
        if folder == 'unknown_folder':
            continue
            
        print(f"   📁 Scanning folder: {folder}")
        
        if os.path.exists(folder):
            pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
            print(f"      Found {len(pdf_files)} PDF files")
            
            # Get documents from this folder
            folder_docs = plot_df[plot_df['folder'] == folder]
            
            for _, row in folder_docs.iterrows():
                doc_id = row['document_id']
                title = row.get('title', 'Unknown Document')
                
                actual_filename = discover_actual_filenames(folder, doc_id, title)
                if actual_filename:
                    filename_mapping[doc_id] = actual_filename
                    print(f"      ✅ Mapped {doc_id} ({title}) -> {actual_filename}")
        else:
            print(f"      ❌ Folder not found: {folder}")
    
    print(f"📋 Filename mapping complete: {len(filename_mapping)} files mapped")
    return filename_mapping

# def prepare_enhanced_indexes():
#     """Create enhanced indexes with pre-computed embeddings and metadata."""
#
#     check_folders()
#     check_dependencies()
#     store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)
#     print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME} on {OPTIMAL_DEVICE}")
#     model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
#     print(f"✅ Model loaded with {model.get_sentence_embedding_dimension()}D embeddings")
#
#     # Load data
#     plot_df = pd.read_pickle("data/plot_df.pkl")
#     print(f"📊 Processing {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")
#
#     # Build filename mapping by scanning source folders
#     filename_mapping = build_filename_mapping(plot_df)
#
#     # Create enhanced directories
#     os.makedirs("indexes_enhanced", exist_ok=True)
#     os.makedirs("embeddings_enhanced", exist_ok=True)
#     os.makedirs("summaries", exist_ok=True)
#
#     # Process each cluster
#     for cluster in sorted(plot_df['cluster'].unique()):
#         print(f"\n🔧 Processing cluster {cluster}...")
#
#         cluster_df = plot_df[plot_df['cluster'] == cluster]
#
#         if len(cluster_df) == 0:
#             print(f"   ⚠️ Skipping empty cluster {cluster}")
#             continue
#
#         # Prepare enhanced data structures
#         all_chunks = []
#         all_embeddings = []
#         chunk_metadata = []
#         doc_summaries = {}
#
#         print(f"   📄 Processing {len(cluster_df)} documents...")
#
#         # ---- inside the “for cluster …” loop ----
#         for _, row in cluster_df.iterrows():
#             doc_id = row['document_id']
#             text = str(row['text'])
#             title = row.get('title', 'Unknown Document')
#             country = row.get('country', 'Unknown')
#             folder = row.get('folder', 'unknown_folder')
#
#             # filename from earlier mapping
#             actual_filename = filename_mapping.get(doc_id)
#
#             # screen output + smarter title
#             print(f"      📄 Processing {doc_id} ({title})…")
#             extracted_title = extract_document_title_from_content(text, title)
#             display_title = extracted_title if extracted_title != title and len(extracted_title) > len(title) else title
#             if extracted_title != title:
#                 print(f"         📝 Extracted title: '{extracted_title}'")
#
#             # one‑paragraph summary
#             # ─── summaries: load from cache ─────────────────────────────
#             summary = store.summary(doc_id)  # ✨ 1‑liner, no recompute
#             summary_vector = store.summary_vec(doc_id)  # for the FAISS overview index
#             doc_summaries[doc_id] = {
#                 'summary': summary,
#                 'title': title,
#                 'extracted_title': extracted_title,
#                 'display_title': display_title,
#                 'country': country,
#                 'folder': folder,
#                 'filename': actual_filename,
#                 'full_length': len(text)
#             }
#
#             # -------- cached chunks + embeddings --------
#             chunks = store.chunks(doc_id)  # list[str]
#             chunk_vectors = store.chunk_vecs(doc_id)  # np.ndarray, shape (n_chunks, dim)
#
#             for idx_c, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
#                 all_chunks.append(chunk)
#                 all_embeddings.append(vec)  # <- 1‑D array per chunk
#                 chunk_metadata.append({
#                     'doc_id': doc_id,
#                     'chunk_id': f"{doc_id}_chunk_{idx_c}",
#                     'title': title,
#                     'extracted_title': extracted_title,
#                     'display_title': display_title,
#                     'country': country,
#                     'folder': folder,
#                     'filename': actual_filename,
#                     'chunk_index': idx_c,
#                     'total_chunks': len(chunks),
#                     'summary': summary
#                 })
#
#         print(f"   🧩 Created {len(all_chunks)} chunks from {len(cluster_df)} documents")
#
#         # -------- assemble embeddings (no re‑encoding!) --------
#         chunk_embeddings = np.vstack(all_embeddings).astype(np.float32)
#
#         # Also generate embeddings for summaries (for quick overview queries)
#         summaries_list = [doc_summaries[doc_id]['summary'] for doc_id in doc_summaries.keys()]
#         print(f"   📝 Generating embeddings for {len(summaries_list)} summaries...")
#         # ----- after collecting all per‑doc data -----
#         summary_embeddings = np.vstack([store.summary_vec(d) for d in doc_summaries]).astype(np.float32)
#
#         # Create separate FAISS indexes
#
#         # 1. Chunk index (detailed search)
#         chunk_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
#         chunk_index.add(chunk_embeddings.astype(np.float32))
#
#         # 2. Summary index (quick overview)
#         summary_index = faiss.IndexFlatL2(summary_embeddings.shape[1])
#         summary_index.add(summary_embeddings.astype(np.float32))
#
#         # Save enhanced data
#         cluster_data = {
#             'chunks': all_chunks,
#             'chunk_metadata': chunk_metadata,
#             'chunk_embeddings': chunk_embeddings,
#             'doc_summaries': doc_summaries,
#             'summary_embeddings': summary_embeddings,
#             '': list(doc_summaries.keys())
#         }
#
#         # Save files
#         faiss.write_index(chunk_index, f"indexes_enhanced/cluster_{cluster}_chunks.index")
#         faiss.write_index(summary_index, f"indexes_enhanced/cluster_{cluster}_summaries.index")
#
#         with open(f"embeddings_enhanced/cluster_{cluster}.pkl", "wb") as f:
#             pickle.dump(cluster_data, f)
#
#         print(f"   ✅ Saved enhanced indexes and embeddings")
#         print(f"      - Chunk index: {len(all_chunks)} items")
#         print(f"      - Summary index: {len(doc_summaries)} items")
#
#         # Clear memory
#         del chunk_embeddings, summary_embeddings, chunk_index, summary_index
#         gc.collect()
#
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#     print("\n🎉 Enhanced index preparation complete!")
#     print("\nNow you can use ultra-fast context generation with:")
#     print("  - Pre-computed embeddings (no real-time encoding)")
#     print("  - Intelligent text chunks (better context boundaries)")
#     print("  - Document summaries (quick overview mode)")
#     print("  - Rich metadata (titles, countries, chunk relationships)")
def prepare_enhanced_indexes():
    """Create enhanced indexes with pre-computed embeddings and metadata."""

    # Check folders and dependencies
    check_folders()
    check_dependencies()

    # Initialize Embedding Store
    store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)
    st.write(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME} on {OPTIMAL_DEVICE}")

    # Initialize the model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
    st.write(f"✅ Model loaded with {model.get_sentence_embedding_dimension()}D embeddings")

    # Load the plot_df data
    plot_df = pd.read_pickle("data/plot_df.pkl")
    st.write(f"📊 Processing {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")

    # Build filename mapping
    filename_mapping = build_filename_mapping(plot_df)

    # Create enhanced directories
    os.makedirs("indexes_enhanced", exist_ok=True)
    os.makedirs("embeddings_enhanced", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

    # Process each cluster
    for cluster in sorted(plot_df['cluster'].unique()):
        st.write(f"\n🔧 Processing cluster {cluster}...")

        cluster_df = plot_df[plot_df['cluster'] == cluster]

        if len(cluster_df) == 0:
            st.warning(f"⚠️ Skipping empty cluster {cluster}")
            continue

        # Prepare enhanced data structures
        all_chunks = []
        all_embeddings = []
        chunk_metadata = []
        doc_summaries = {}

        st.write(f"   📄 Processing {len(cluster_df)} documents...")

        # Process documents in the cluster
        for _, row in cluster_df.iterrows():
            doc_id = row['document_id']
            text = str(row['text'])
            title = row.get('title', 'Unknown Document')
            country = row.get('country', 'Unknown')
            folder = row.get('folder', 'unknown_folder')

            # Filename from earlier mapping
            actual_filename = filename_mapping.get(doc_id)

            # Display information about the document
            # st.write(f"      📄 Processing {doc_id} ({title})…")
            extracted_title = extract_document_title_from_content(text, title)
            display_title = extracted_title if extracted_title != title and len(extracted_title) > len(title) else title
            # if extracted_title != title:
            #     st.write(f"         📝 Extracted title: '{extracted_title}'")

            # One-paragraph summary (from cache)
            summary = store.summary(doc_id)  # 1-liner, no recompute
            summary_vector = store.summary_vec(doc_id)  # For the FAISS overview index
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

            # Cached chunks + embeddings
            chunks = store.chunks(doc_id)  # list[str]
            chunk_vectors = store.chunk_vecs(doc_id)  # np.ndarray, shape (n_chunks, dim)

            for idx_c, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
                all_chunks.append(chunk)
                all_embeddings.append(vec)  # 1-D array per chunk
                chunk_metadata.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_chunk_{idx_c}",
                    'title': title,
                    'extracted_title': extracted_title,
                    'display_title': display_title,
                    'country': country,
                    'folder': folder,
                    'filename': actual_filename,
                    'chunk_index': idx_c,
                    'total_chunks': len(chunks),
                    'summary': summary
                })

        st.write(f"   🧩 Created {len(all_chunks)} chunks from {len(cluster_df)} documents")

        # Assemble embeddings (no re-encoding)
        chunk_embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Generate embeddings for summaries
        summaries_list = [doc_summaries[doc_id]['summary'] for doc_id in doc_summaries.keys()]
        st.write(f"   📝 Generating embeddings for {len(summaries_list)} summaries...")

        summary_embeddings = np.vstack([store.summary_vec(d) for d in doc_summaries]).astype(np.float32)

        # Create FAISS indexes
        chunk_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        chunk_index.add(chunk_embeddings.astype(np.float32))

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

        faiss.write_index(chunk_index, f"indexes_enhanced/cluster_{cluster}_chunks.index")
        faiss.write_index(summary_index, f"indexes_enhanced/cluster_{cluster}_summaries.index")

        with open(f"embeddings_enhanced/cluster_{cluster}.pkl", "wb") as f:
            pickle.dump(cluster_data, f)

        st.success(f"   ✅ Saved enhanced indexes and embeddings")
        st.write(f"      - Chunk index: {len(all_chunks)} items")
        st.write(f"      - Summary index: {len(doc_summaries)} items")

        # Clear memory
        del chunk_embeddings, summary_embeddings, chunk_index, summary_index
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    st.write("\n🎉 Enhanced index preparation complete!")
    st.write("\nNow you can use ultra-fast context generation with:")
    st.write("  - Pre-computed embeddings (no real-time encoding)")
    st.write("  - Intelligent text chunks (better context boundaries)")
    st.write("  - Document summaries (quick overview mode)")
    st.write("  - Rich metadata (titles, countries, chunk relationships)")


if __name__ == "__main__":
    prepare_enhanced_indexes() 