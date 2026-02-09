# ultra_fast_rag.py
# Ultra-fast context generation using preprocessed data

import faiss
import pickle
import numpy as np
from sentence_transformers import util
import streamlit as st
from pathlib import Path
import time
import re

class UltraFastRAG:
    """Ultra-fast RAG engine using preprocessed embeddings and intelligent chunks."""
    
    def __init__(self, model):
        self.model = model
        self.cache = {}  # Cache loaded cluster data
    
    def clear_cache(self):
        """Clear the cached cluster data to force reload of enhanced indexes."""
        st.info(f"🗑️ **DEBUG**: Clearing cache for {len(self.cache)} clusters")
        self.cache.clear()
        st.success(f"✅ **DEBUG**: Cache cleared - will reload enhanced indexes")
    
    def load_enhanced_cluster(self, cluster_id):
        """Load enhanced cluster data with caching."""
        st.info(f"🔍 **DEBUG**: Checking for enhanced indexes for cluster {cluster_id}")
        
        if cluster_id in self.cache:
            st.success(f"📦 **DEBUG**: Using cached enhanced data for cluster {cluster_id}")
            return self.cache[cluster_id]
        
        try:
            # Check if enhanced indexes exist
            chunk_index_path = f"indexes_enhanced/cluster_{cluster_id}_chunks.index"
            summary_index_path = f"indexes_enhanced/cluster_{cluster_id}_summaries.index"
            data_path = f"embeddings_enhanced/cluster_{cluster_id}.pkl"
            
            st.info(f"🔍 **DEBUG**: Looking for enhanced files:")
            st.info(f"   📁 Chunk index: {chunk_index_path}")
            st.info(f"   📁 Summary index: {summary_index_path}")
            st.info(f"   📁 Data file: {data_path}")
            
            files_exist = [Path(p).exists() for p in [chunk_index_path, summary_index_path, data_path]]
            st.info(f"🔍 **DEBUG**: File existence: {files_exist}")
            
            if not all(files_exist):
                st.warning(f"❌ **DEBUG**: Enhanced indexes not found for cluster {cluster_id}")
                return None
            
            # Load indexes and data
            st.info(f"📦 **DEBUG**: Loading enhanced indexes for cluster {cluster_id}...")
            start_time = time.time()
            
            chunk_index = faiss.read_index(chunk_index_path)
            summary_index = faiss.read_index(summary_index_path)
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            load_time = time.time() - start_time
            st.success(f"✅ **DEBUG**: Enhanced data loaded in {load_time:.2f}s")
            st.info(f"📊 **DEBUG**: Loaded {chunk_index.ntotal} chunks, {summary_index.ntotal} summaries")
            
            cluster_data = {
                'chunk_index': chunk_index,
                'summary_index': summary_index,
                'data': data
            }
            
            # Cache for future use
            self.cache[cluster_id] = cluster_data
            st.success(f"💾 **DEBUG**: Cached enhanced data for cluster {cluster_id}")
            return cluster_data
            
        except Exception as e:
            st.error(f"❌ **DEBUG**: Failed to load enhanced data for cluster {cluster_id}: {e}")
            return None
    
    def ultra_fast_retrieve(self, question, cluster_id, mode="detailed", top_k=5):
        """
        Ultra-fast retrieval using pre-computed embeddings.
        
        Args:
            question: User question
            cluster_id: Cluster to search
            mode: "detailed" (chunks) or "overview" (summaries)
            top_k: Number of results
        """
        st.info(f"🚀 **DEBUG**: Starting ultra-fast retrieval")
        st.info(f"   🔍 Mode: {mode}")
        st.info(f"   📊 Top-k: {top_k}")
        
        cluster_data = self.load_enhanced_cluster(cluster_id)
        if not cluster_data:
            st.error(f"❌ **DEBUG**: No enhanced cluster data available")
            return [], []
        
        # Encode question only (much faster than encoding all passages)
        st.info(f"🧮 **DEBUG**: Encoding question: '{question[:50]}...'")
        encode_start = time.time()
        question_embedding = self.model.encode([question])
        encode_time = time.time() - encode_start
        st.info(f"⚡ **DEBUG**: Question encoded in {encode_time:.3f}s")
        
        search_start = time.time()
        
        if mode == "overview":
            st.info(f"📋 **DEBUG**: Using OVERVIEW mode (document summaries)")
            # Quick overview using document summaries
            index = cluster_data['summary_index']
            doc_summaries = cluster_data['data']['doc_summaries']
            
            # FAISS search
            scores, indices = index.search(question_embedding.astype(np.float32), min(top_k, index.ntotal))
            
            # Build results
            results = []
            metadata = []
            doc_ids = list(doc_summaries.keys())
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    doc_info = doc_summaries[doc_id]
                    results.append(doc_info['summary'])
                    metadata.append({
                        'doc_id': doc_id,
                        'title': doc_info['title'],
                        'country': doc_info['country'],
                        'score': 1.0 - (score / 2.0),  # Convert L2 distance to similarity
                        'type': 'summary'
                    })
                    
        else:  # detailed mode
            st.info(f"🔍 **DEBUG**: Using DETAILED mode (text chunks)")
            # Detailed search using text chunks
            index = cluster_data['chunk_index']
            chunks = cluster_data['data']['chunks']
            chunk_metadata = cluster_data['data']['chunk_metadata']
            
            # FAISS search
            scores, indices = index.search(question_embedding.astype(np.float32), min(top_k, index.ntotal))
            
            # Build results
            results = []
            metadata = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    chunk_meta = chunk_metadata[idx]
                    results.append(chunk)
                    metadata.append({
                        **chunk_meta,
                        'score': 1.0 - (score / 2.0),  # Convert L2 distance to similarity
                        'type': 'chunk'
                    })
        
        search_time = time.time() - search_start
        st.success(f"🎯 **DEBUG**: Retrieved {len(results)} items in {search_time:.3f}s")
        
        # Show top results
        if metadata:
            st.info(f"🏆 **DEBUG**: Top results:")
            for i, meta in enumerate(metadata[:3]):
                score = meta['score']
                title = meta.get('title', 'Unknown')[:30]
                st.info(f"   {i+1}. {title}... (score: {score:.3f})")
        
        return results, metadata
    
    def build_ultra_fast_context(self, question, cluster_id, max_chars=12000, mode="auto", top_k=5):
        """
        Build context using ultra-fast preprocessing.
        
        Args:
            mode: "auto", "detailed", or "overview"
        """
        st.info(f"⚡ **DEBUG**: Starting ultra-fast context generation")
        st.info(f"   📏 Max chars: {max_chars}")
        st.info(f"   🎛️ Mode: {mode}")
        st.info(f"   📊 Top-k documents: {top_k}")
        
        if mode == "auto":
            # Smart mode selection based on question type
            overview_keywords = ["summary", "overview", "what is", "general", "broadly"]
            detected_mode = "overview" if any(kw in question.lower() for kw in overview_keywords) else "detailed"
            st.info(f"🤖 **DEBUG**: Auto-detected mode: {detected_mode}")
            mode = detected_mode
        
        # Retrieve relevant content
        st.info(f"📊 **DEBUG**: Retrieving top-{top_k} results for {mode} mode")
        
        context_start = time.time()
        results, metadata = self.ultra_fast_retrieve(question, cluster_id, mode=mode, top_k=top_k)
        retrieval_time = time.time() - context_start
        
        if not results:
            st.warning(f"❌ **DEBUG**: No results retrieved")
            return "", []
        
        st.success(f"📦 **DEBUG**: Retrieved {len(results)} items in {retrieval_time:.3f}s")
        
        # Build structured context
        build_start = time.time()
        context_parts = []
        total_chars = 0
        included_metadata = []
        
        st.info(f"🔧 **DEBUG**: Building context from {len(results)} items...")
        
        for i, (result, meta) in enumerate(zip(results, metadata)):
            if mode == "overview":
                header = f"\n--- DOCUMENT SUMMARY {i+1}: {meta['title']} ({meta['country']}) ---\n"
            else:
                header = f"\n--- EXCERPT {i+1}: {meta['title']} (chunk {meta['chunk_index']+1}/{meta['total_chunks']}) ---\n"
            
            formatted = header + result.strip()
            
            if total_chars + len(formatted) <= max_chars:
                context_parts.append(formatted)
                total_chars += len(formatted)
                included_metadata.append(meta)
                st.info(f"   ✅ Added item {i+1}: {len(formatted)} chars")
            else:
                # Try to fit truncated version
                remaining = max_chars - total_chars - len(header)
                if remaining > 200:
                    truncated = header + result[:remaining-50] + "... [truncated]"
                    context_parts.append(truncated)
                    included_metadata.append({**meta, 'truncated': True})
                    st.info(f"   ✂️ Added truncated item {i+1}: {len(truncated)} chars")
                else:
                    st.info(f"   ❌ Skipped item {i+1}: would exceed limit")
                break
        
        context = "".join(context_parts)
        build_time = time.time() - build_start
        
        st.success(f"✅ **DEBUG**: Context built in {build_time:.3f}s")
        st.info(f"📏 **DEBUG**: Final context: {len(context)} chars, {len(included_metadata)} items")
        
        return context, included_metadata
    
    def get_cluster_stats(self, cluster_id):
        """Get statistics about the enhanced cluster data."""
        cluster_data = self.load_enhanced_cluster(cluster_id)
        if not cluster_data:
            return None
        
        data = cluster_data['data']
        stats = {
            'total_documents': len(data['doc_summaries']),
            'total_chunks': len(data['chunks']),
            'avg_chunks_per_doc': len(data['chunks']) / len(data['doc_summaries']),
            'has_enhanced_indexes': True
        }
        
        st.info(f"📊 **DEBUG**: Cluster {cluster_id} stats: {stats}")
        return stats

def clean_deepseek_response_ultra(generated_text):
    """
    Ultra-aggressive DeepSeek response cleaner for the ultra-fast pipeline.
    Removes ALL thinking content and extracts only the final answer.
    """
    st.info(f"🧹 **DEBUG**: Starting aggressive DeepSeek response cleaning")
    st.info(f"   📏 Raw response: {len(generated_text)} characters")
    
    # Original text for debugging
    original_text = generated_text
    
    # Remove end tokens first
    text = generated_text.replace("<|im_end|>", "").strip()
    
    # Remove assistant prefix if present
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1].strip()
        st.info(f"🧹 **DEBUG**: Removed assistant prefix")
    
    # AGGRESSIVE: Remove everything before and including </think>
    if "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            text = parts[1].strip()
            st.info(f"🧹 **DEBUG**: Removed thinking process (everything before </think>)")
        else:
            st.warning(f"⚠️ **DEBUG**: Found </think> but couldn't split properly")
    
    # Also remove any remaining <think> tags (in case of incomplete tags)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)  # Remove unclosed thinking
    
    # Remove common thinking patterns that might remain
    thinking_patterns = [
        r'^.*?(?:Let me analyze|I need to|Looking at|First, let me|To answer this)',
        r'^.*?(?:Based on the|According to|From the documents)',
        r'^.*?(?:The question asks|The user is asking)',
        r'^.*?(?:I should|I will|I can see)',
        r'^.*?(?:Here\'s what|Here is what)',
    ]
    
    for pattern in thinking_patterns:
        if re.match(pattern, text, re.IGNORECASE | re.DOTALL):
            # Find the actual answer start
            answer_markers = [
                r'(?:In summary|Therefore|Thus|Hence|To conclude|The answer is|Overall)',
                r'(?:Based on this analysis|From this information)',
                r'(?:The key findings|The main points)',
                r'(?:\n\n|\n[A-Z])',  # New paragraph or sentence
            ]
            
            for marker_pattern in answer_markers:
                match = re.search(marker_pattern, text, re.IGNORECASE)
                if match:
                    text = text[match.start():].strip()
                    st.info(f"🧹 **DEBUG**: Found answer marker, extracted from: '{marker_pattern}'")
                    break
            break
    
    # Clean up excessive whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading spaces
    text = text.strip()
    
    # Final validation
    if len(text) < 20:
        st.warning(f"⚠️ **DEBUG**: Cleaned text too short ({len(text)} chars), trying extraction")
        # Try to extract any substantial content from original
        sentences = re.split(r'[.!?]+', original_text)
        substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 30 and 
                               not any(think_word in s.lower() for think_word in 
                                     ['let me', 'i need to', 'i should', 'looking at', 'analyzing'])]
        
        if substantial_sentences:
            text = '. '.join(substantial_sentences[-3:]).strip()  # Take last few substantial sentences
            if not text.endswith('.'):
                text += '.'
            st.info(f"🧹 **DEBUG**: Extracted substantial sentences: {len(text)} chars")
        else:
            text = "The model response contained mostly thinking process. Please try rephrasing your question for a more direct answer."
    
    # Final check for quality
    if "think>" in text.lower() or len([c for c in text if c.isalpha()]) < 10:
        text = "Response contained excessive thinking content. Please try a simpler question or rephrase for a direct answer."
        st.warning(f"⚠️ **DEBUG**: Response still contained thinking content, using fallback message")
    
    st.success(f"✅ **DEBUG**: Cleaning complete: {len(original_text)} → {len(text)} chars")
    st.info(f"💬 **DEBUG**: Cleaned preview: '{text[:100]}...'")
    
    return text

def extract_country_from_content(doc_id, title, content_sample=""):
    """Extract country information from document ID, title, or content."""
    
    # Common country code mappings
    country_codes = {
        'can': 'Canada', 'gbr': 'United Kingdom', 'usa': 'United States', 'aus': 'Australia',
        'bul': 'Bulgaria', 'cze': 'Czech Republic', 'slo': 'Slovakia', 'pol': 'Poland',
        'deu': 'Germany', 'fra': 'France', 'ita': 'Italy', 'esp': 'Spain', 'nld': 'Netherlands',
        'bel': 'Belgium', 'dnk': 'Denmark', 'swe': 'Sweden', 'nor': 'Norway', 'fin': 'Finland',
        'jpn': 'Japan', 'kor': 'South Korea', 'chn': 'China', 'ind': 'India', 'bra': 'Brazil',
        'mex': 'Mexico', 'arg': 'Argentina', 'chl': 'Chile', 'per': 'Peru', 'col': 'Colombia',
        'rus': 'Russia', 'ukr': 'Ukraine', 'tur': 'Turkey', 'irn': 'Iran', 'isr': 'Israel',
        'egy': 'Egypt', 'zaf': 'South Africa', 'nga': 'Nigeria', 'eth': 'Ethiopia', 'ken': 'Kenya'
    }
    
    # Try to extract from document ID first
    if title and len(title) >= 3:
        title_lower = title.lower()
        for code, country in country_codes.items():
            if title_lower.startswith(code):
                return country
    
    # Try from document ID
    if doc_id and len(doc_id) >= 6:  # e.g., "doc_259" -> check original data
        doc_lower = doc_id.lower()
        for code, country in country_codes.items():
            if code in doc_lower:
                return country
    
    # Try from content if available
    if content_sample:
        content_lower = content_sample.lower()
        # Look for country names in content
        for code, country in country_codes.items():
            if country.lower() in content_lower:
                return country
        
        # Look for "CANADA", "UNITED KINGDOM" etc. in caps
        if 'canada' in content_lower:
            return 'Canada'
        elif 'united kingdom' in content_lower or 'great britain' in content_lower:
            return 'United Kingdom'
        elif 'united states' in content_lower:
            return 'United States'
        elif 'australia' in content_lower:
            return 'Australia'
    
    return None

# Integration function for existing RAG engine
def ultra_fast_answer_question(question, cluster_id, model, model_name=None, max_tokens=8000, top_k=5):
    """Drop-in replacement for the existing answer_question function."""
    
    st.info(f"🚀 **DEBUG**: ULTRA-FAST RAG PIPELINE STARTING")
    st.info(f"   🔍 Question: '{question[:50]}...'")
    st.info(f"   📂 Cluster: {cluster_id}")
    st.info(f"   🎯 Max tokens: {max_tokens}")
    st.info(f"   📊 Top-k documents: {top_k}")
    
    # Try to use enhanced indexes first
    fast_rag = UltraFastRAG(model)
    
    # Check if enhanced data exists
    if fast_rag.load_enhanced_cluster(cluster_id):
        st.success(f"⚡ **DEBUG**: ENHANCED INDEXES FOUND - Using ultra-fast mode!")
        
        # Get cluster stats
        stats = fast_rag.get_cluster_stats(cluster_id)
        st.info(f"📊 **DEBUG**: {stats['total_documents']} documents, {stats['total_chunks']} chunks")
        
        # Build context ultra-fast with slider-controlled parameters
        st.info(f"🔧 **DEBUG**: Building ultra-fast context with top_k={top_k}...")
        total_start = time.time()
        
        context, metadata = fast_rag.build_ultra_fast_context(
            question, cluster_id, max_chars=max_tokens*3, top_k=top_k
        )
        
        context_time = time.time() - total_start
        
        if context:
            st.success(f"✅ **DEBUG**: Context ready in {context_time:.2f}s - {len(metadata)} items, {len(context)} chars")
            
            # Show context preview
            st.info(f"📄 **DEBUG**: Context preview (first 200 chars):")
            st.code(context[:200] + "..." if len(context) > 200 else context)
            
            # Generate answer with ENHANCED token limits for ultra-fast pipeline
            st.info(f"🧠 **DEBUG**: Passing context to model for answer generation...")
            answer_start = time.time()
            
            # ULTRA-FAST PIPELINE: Use custom generation for better token limits
            if model_name and any(keyword in model_name.lower() for keyword in ["deepseek", "r1"]):
                # Get display name for the message
                model_name_mapping = {
                    "DeepSeek-R1-Distill-Qwen-14B (Recommended)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    "DeepSeek-R1-Distill-Qwen-7B (Faster)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "DeepSeek-R1-Distill-Llama-8B (Alternative)": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "DeepSeek-LLM-7B-Chat (Fallback)": "deepseek-ai/deepseek-llm-7b-chat",
                }
                actual_model_name = model_name_mapping.get(model_name, model_name)
                model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
                st.info(f"🤖 **DEBUG**: Using ULTRA-FAST {model_display_name} generation with enhanced token limits")
                raw_answer = generate_answer_ultra_fast_deepseek(question, context, model_name)
            else:
                from rag_engine import generate_answer
                raw_answer = generate_answer(question, context, model_name)
            
            # ENHANCED: Apply ultra-aggressive cleaning specifically for DeepSeek responses
            if model_name and any(keyword in model_name.lower() for keyword in ["deepseek", "r1"]):
                st.info(f"🧹 **DEBUG**: Applying ULTRA-FAST pipeline DeepSeek cleaning")
                answer = clean_deepseek_response_ultra(raw_answer)
            else:
                answer = raw_answer
            
            answer_time = time.time() - answer_start
            
            st.success(f"✅ **DEBUG**: Answer generated in {answer_time:.2f}s")
            st.info(f"💬 **DEBUG**: Final answer preview (first 100 chars):")
            st.code(answer[:100] + "..." if len(answer) > 100 else answer)
            
            from utils.token_counting import count_tokens
            from config import OPENAI_MODEL
            token_count = count_tokens(context, model_name=model_name or OPENAI_MODEL)
            
            # Convert metadata back to text for display - ENHANCED for better document matching
            passages = []
            
            # Get the original retrieved content for proper passage display
            # Re-retrieve to get the actual chunks for display
            retrieval_data = fast_rag.ultra_fast_retrieve(question, cluster_id, mode="detailed", top_k=top_k)
            retrieved_chunks, retrieved_metadata = retrieval_data
            
            for i, meta in enumerate(metadata):
                # Create well-formatted passages that match the expected format
                doc_title = meta.get('title', f'Document {i+1}')
                extracted_title = meta.get('extracted_title', '')
                display_title = meta.get('display_title', '')
                doc_country = meta.get('country', 'Unknown')
                content_type = meta.get('type', 'chunk')
                doc_id = meta.get('doc_id', 'unknown')
                
                # ENHANCED: Intelligent country extraction when country is None/missing
                if doc_country is None or doc_country == 'None' or str(doc_country).lower() == 'none':
                    # Try to extract from document information
                    sample_content = ""
                    if i < len(metadata) and 'summary' in meta:
                        sample_content = meta['summary'][:500]  # Use summary for country detection
                    
                    extracted_country = extract_country_from_content(doc_id, doc_title, sample_content)
                    if extracted_country:
                        doc_country = extracted_country
                        st.info(f"🔍 **DEBUG**: Extracted country '{extracted_country}' from {doc_title}")
                    else:
                        doc_country = 'Unspecified'
                
                # Use the best available title for display
                if display_title and display_title != doc_title and len(display_title) > len(doc_title):
                    # We have a good extracted title
                    clean_title = display_title
                    st.info(f"📝 **DEBUG**: Using extracted title for {doc_id}: '{display_title}'")
                elif extracted_title and extracted_title != doc_title and len(extracted_title) > len(doc_title):
                    # We have an extracted title
                    clean_title = extracted_title 
                    st.info(f"📝 **DEBUG**: Using extracted title for {doc_id}: '{extracted_title}'")
                elif doc_title and doc_title != 'Unknown Document':
                    # Fall back to cleaning up the original title
                    if len(doc_title) <= 10 and not ' ' in doc_title:
                        # This looks like a document ID, enhance it
                        if doc_country != 'Unspecified':
                            clean_title = f"{doc_country} Report ({doc_title.upper()})"
                        else:
                            clean_title = f"Document {doc_title.upper()}"
                    else:
                        clean_title = doc_title[:60] + "..." if len(doc_title) > 60 else doc_title
                else:
                    clean_title = f"Document {i+1}"
                
                if content_type == 'summary':
                    passage_header = f"📋 SUMMARY: {clean_title}"
                else:
                    chunk_info = f"chunk {meta.get('chunk_index', i)+1}/{meta.get('total_chunks', '?')}"
                    passage_header = f"📄 EXCERPT: {clean_title} - {chunk_info}"
                
                # Get the ACTUAL content from the retrieved chunks
                if i < len(retrieved_chunks):
                    # Use the actual retrieved chunk content
                    passage_content = retrieved_chunks[i].strip()
                    st.info(f"📄 **DEBUG**: Using actual chunk content ({len(passage_content)} chars) for passage {i+1}")
                else:
                    # Fallback: try to extract from context (less reliable)
                    st.warning(f"⚠️ **DEBUG**: No retrieved chunk for passage {i+1}, extracting from context")
                    context_parts = context.split(f'--- EXCERPT {i+1}:')
                    if len(context_parts) > 1:
                        # Get content between this excerpt and the next
                        content_part = context_parts[1]
                        if f'--- EXCERPT {i+2}:' in content_part:
                            content_part = content_part.split(f'--- EXCERPT {i+2}:')[0]
                        # Remove header line and get actual content
                        content_lines = content_part.split('\n')[1:]  # Skip header line
                        passage_content = '\n'.join(content_lines).strip()
                    else:
                        passage_content = f"Could not extract content for {clean_title}"
                
                # Truncate if too long for display
                if len(passage_content) > 800:
                    passage_content = passage_content[:800] + "... [truncated for display]"
                
                # Additional context for better understanding
                if doc_id and doc_id != 'unknown':
                    # Check if we have actual filename information
                    actual_filename = meta.get('filename', None)
                    
                    if actual_filename:
                        # We have a real filename
                        if doc_country != 'Unspecified':
                            metadata_line = f"Document ID: {doc_id} | Filename: {actual_filename} | Country: {doc_country}"
                        else:
                            metadata_line = f"Document ID: {doc_id} | Filename: {actual_filename}"
                    else:
                        # No real filename, just show what we have
                        if doc_country != 'Unspecified':
                            metadata_line = f"Document ID: {doc_id} | Title: {doc_title} | Country: {doc_country}"
                        else:
                            metadata_line = f"Document ID: {doc_id} | Title: {doc_title}"
                    
                    passage_content = f"{metadata_line}\n\n{passage_content}"
                
                # Combine header and content
                formatted_passage = f"{passage_header}\n\n{passage_content}"
                passages.append(formatted_passage)
            
            st.info(f"📦 **DEBUG**: Formatted {len(passages)} enhanced passages with actual content")
            # Show sample of formatted passages with better info
            if passages:
                sample_preview = passages[0][:150].replace('\n', ' ')
                st.info(f"📄 **DEBUG**: Sample passage: '{sample_preview}...'")
            
            total_time = time.time() - total_start
            st.success(f"🎉 **DEBUG**: ULTRA-FAST PIPELINE COMPLETE in {total_time:.2f}s")
            st.info(f"   ⚡ Context: {context_time:.2f}s")
            st.info(f"   🧠 Answer: {answer_time:.2f}s")
            
            return answer, passages, token_count
        else:
            st.error(f"❌ **DEBUG**: No context generated")
            return "No relevant information found.", [], 0
    else:
        # Fallback to regular RAG
        st.warning(f"⚠️ **DEBUG**: ENHANCED INDEXES NOT FOUND - Using standard RAG")
        st.info("💡 **DEBUG**: Run `python prepare_enhanced_index.py` to enable ultra-fast mode")
        
        fallback_start = time.time()
        from rag_engine import answer_question
        result = answer_question(question, cluster_id, model_name, top_k=5, max_tokens=max_tokens)
        fallback_time = time.time() - fallback_start
        
        st.info(f"🐌 **DEBUG**: Standard RAG completed in {fallback_time:.2f}s")
        return result

def generate_answer_ultra_fast_deepseek(question, context, model_name):
    """
    Enhanced generation specifically for ultra-fast pipeline with proper token limits.
    """
    try:
        from rag_engine import get_cached_local_model
        import warnings
        import time as time_module
        import threading
        
        # Get the cached model - FIXED: Pass model_name for proper selection
        qa_model, actual_model_name = get_cached_local_model(preferred_model_name=model_name)
        
        # Create display name for messages
        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
        
        st.info(f"🚀 **DEBUG**: ULTRA-FAST {model_display_name} generation starting")
        st.info(f"   🎯 Model: {model_name}")
        st.info(f"   📏 Context: {len(context)} chars")
        
        # Handle OpenAI models (return None from get_cached_local_model)
        if qa_model is None:
            st.info(f"☁️ **DEBUG**: Model {model_name} is handled by OpenAI API, falling back to standard generation")
            from rag_engine import generate_answer
            return generate_answer(question, context, model_name)
        
        st.success(f"🧠 **DEBUG**: Using cached model: {actual_model_name}")
        
        # Show which model is actually being used vs requested
        if model_name and model_name != actual_model_name:
            # Check if it's a display name that was mapped
            model_name_mapping = {
                "DeepSeek-R1-Distill-Qwen-14B (Recommended)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "DeepSeek-R1-Distill-Qwen-7B (Faster)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "DeepSeek-R1-Distill-Llama-8B (Alternative)": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "DeepSeek-LLM-7B-Chat (Fallback)": "deepseek-ai/deepseek-llm-7b-chat",
                "TinyLlama-1B (8GB RAM)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "DistilGPT2 (Low Memory)": "distilgpt2",
                "FLAN-T5-Small (Ultra Light)": "google/flan-t5-small",
            }
            
            mapped_name = model_name_mapping.get(model_name, model_name)
            if mapped_name == actual_model_name:
                st.success(f"✅ **DEBUG**: Using requested model: {model_name}")
            else:
                st.info(f"📝 **DEBUG**: ULTRA-FAST: Requested: {model_name}, Using: {actual_model_name}")
        
        # Enhanced prompt for reasoning models
        prompt = f"""<|im_start|>system
You are a specialized AI assistant for climate policy analysis. Your role is to provide accurate, well-structured, and actionable insights from UNFCCC climate documents.

CORE RESPONSIBILITIES:
• Analyze climate policy documents with precision and depth
• Synthesize information from multiple sources coherently
• Provide evidence-based answers with proper context
• Maintain objectivity and scientific accuracy
• Focus on practical implications and policy relevance

RESPONSE FORMAT - DETAILED ANALYSIS:
1. **Executive Summary** (2-3 sentences)
   - Key finding or direct answer to the question
   - Most critical insight or recommendation

2. **Main Analysis** (structured paragraphs)
   - Detailed explanation with supporting evidence
   - Reference specific documents, countries, or policies
   - Include relevant context and background

3. **Supporting Evidence**
   - Quote or reference specific document excerpts
   - Mention countries, dates, or specific commitments
   - Highlight patterns or trends across documents

4. **Key Takeaways** (bullet points)
   - 3-5 actionable insights
   - Clear, concise statements
   - Prioritized by importance

IMPORTANT FORMATTING GUIDELINES:
• Start responses immediately with substantive content
• Use markdown formatting for structure (##, **, •)
• Keep explanations clear and focused
• Cite specific countries or document details when relevant
• End with actionable insights
<|im_end|>
<|im_start|>user
Based on the following excerpts from UNFCCC climate policy documents, please answer the question with the specified format and quality standards.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Please provide a comprehensive, well-structured response following the format guidelines above.
<|im_end|>
<|im_start|>assistant
"""
        
        # ULTRA-FAST PIPELINE: Enhanced token limits specifically for this pipeline
        # Model-specific token limits to prevent truncation
        if "tinyllama" in actual_model_name.lower():
            # TinyLlama needs more tokens for complete responses in ultra-fast mode
            enhanced_params = {
                'max_new_tokens': 800,  # MATCHED to rag_engine.py fix
                'num_return_sequences': 1,
                'temperature': 0.8,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.15,
                'pad_token_id': qa_model.tokenizer.eos_token_id,
                'early_stopping': False,  # Let TinyLlama complete responses
                'num_beams': 1,
            }
            st.info(f"🐣 **DEBUG**: ULTRA-FAST TinyLlama generation (max_tokens={enhanced_params['max_new_tokens']})")
        elif "distilgpt2" in actual_model_name.lower() or "flan-t5" in actual_model_name.lower():
            # Lightweight models need moderate token limits
            enhanced_params = {
                'max_new_tokens': 600,  # MATCHED to rag_engine.py fix
                'num_return_sequences': 1,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'pad_token_id': qa_model.tokenizer.eos_token_id,
                'early_stopping': False,
                'num_beams': 1,
            }
            st.info(f"⚡ **DEBUG**: ULTRA-FAST lightweight model generation (max_tokens={enhanced_params['max_new_tokens']})")
        else:
            # DeepSeek R1 models need more tokens for thinking + answer
            enhanced_params = {
                'max_new_tokens': 1200,  # INCREASED: Allow for thinking + substantial answer
                'num_return_sequences': 1,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'pad_token_id': qa_model.tokenizer.eos_token_id,
                'early_stopping': False,  # Let it complete the thinking process
                'num_beams': 1,  # Keep beam search disabled for speed
            }
            st.info(f"🧠 **DEBUG**: ULTRA-FAST {model_display_name} generation (max_tokens={enhanced_params['max_new_tokens']})")
        
        st.info(f"📤 **DEBUG**: ULTRA-FAST {model_display_name} generation with enhanced limits")
        st.info(f"   📏 Prompt: {len(prompt)} chars")
        
        # Progress indicator for long generation with model-specific time estimates
        progress_placeholder = st.empty()
        
        # Import the helper function for consistent time estimates
        from rag_engine import get_model_time_estimate
        time_estimate = get_model_time_estimate(actual_model_name, is_ultra_fast=True)
        
        progress_placeholder.info(f"🔄 **Ultra-Fast {model_display_name} thinking... This should take {time_estimate}**")
        
        # Enhanced generation with timeout monitoring
        def progress_updater():
            elapsed = 0
            while elapsed < 180:  # 3 minute timeout for enhanced generation
                time_module.sleep(10)
                elapsed += 10
                try:
                    # Only update progress if we're still in Streamlit context
                    if elapsed <= 60:
                        progress_placeholder.info(f"🔄 **Ultra-Fast {model_display_name} thinking... {elapsed}s**")
                    elif elapsed <= 120:
                        progress_placeholder.warning(f"⏰ **Still thinking... {elapsed}s (enhanced token limits)**")
                    else:
                        progress_placeholder.error(f"❌ **Generation taking too long: {elapsed}s**")
                except Exception:
                    # Ignore Streamlit context errors from background thread
                    break
        
        # Start progress thread
        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()
        
        # Generate with enhanced limits
        generation_start = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = qa_model(prompt, **enhanced_params)
        
        generation_time = time.time() - generation_start
        progress_placeholder.empty()
        
        st.success(f"⚡ **DEBUG**: ULTRA-FAST {model_display_name} completed in {generation_time:.2f}s")
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0]['generated_text']
            st.info(f"📤 **DEBUG**: Raw output: {len(generated_text)} chars")
            
            # Quick check if generation was complete
            if len(generated_text) > 1000:
                st.success(f"✅ **DEBUG**: Generated substantial response ({len(generated_text)} chars)")
            else:
                st.warning(f"⚠️ **DEBUG**: Generated short response ({len(generated_text)} chars)")
            
            return generated_text
        else:
            st.error(f"❌ **DEBUG**: No valid result from ULTRA-FAST generation")
            return "Ultra-fast generation failed to produce a valid response."
            
    except Exception as e:
        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
        st.error(f"❌ **DEBUG**: ULTRA-FAST {model_display_name} generation error: {e}")
        # Fallback to standard generation
        from rag_engine import generate_answer
        return generate_answer(question, context, model_name) 