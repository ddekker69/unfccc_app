# 🌍 UNFCCC Climate QA System

A comprehensive AI-powered system for analyzing UNFCCC climate policy documents with semantic search, intelligent clustering, enhanced country extraction, and automated publication year processing capabilities.

## 🏗️ Architecture Layout

The codebase is now organized into clear boundaries:

```text
unfccc/
├── apps/streamlit/      # Streamlit UI + headless app workflows
├── core/config/         # Shared runtime settings (UI-optional)
├── core/rag/            # Retrieval and answer generation engine
├── docs/                # Architecture and technical documentation
├── examples/            # Ad-hoc demos and tutorial scripts
└── *.py                 # Backward-compatible wrapper entrypoints
```

Existing commands and imports still work through wrappers at the original file paths.

## 🚀 Quick Start

### 1. Setup Environment

**Windows:**
```batch
git clone <repository-url>
cd unfccc
setup_environment.bat  # Automated setup with desktop shortcuts
```

**Unix/Mac/Linux:**
```bash
git clone <repository-url>
cd unfccc
conda env create -f environment.yml
conda activate unfccc_env
```

### 2. Configure API Keys (Required for OpenAI)

**⚠️ IMPORTANT**: For Q&A functionality, you need to configure API keys in **two places**:

**Option 1: Quick Setup (Recommended)**
```bash
# Edit Streamlit secrets file
nano .streamlit/secrets.toml
```

Add your keys:
```toml
OPENAI_API_KEY = "sk-proj-YOUR_ACTUAL_KEY_HERE"
AZURE_STORAGE_ACCOUNT_NAME = "your_account_name"  # Optional
AZURE_STORAGE_ACCOUNT_KEY = "your_account_key"    # Optional
```

**Option 2: Environment Variables**
```bash
# For bash/zsh
export OPENAI_API_KEY="sk-proj-YOUR_ACTUAL_KEY_HERE"

# For Windows
set OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
```

**🔗 Get API keys:**
- **OpenAI**: https://platform.openai.com/api-keys (Free $5 credits)
- **Azure**: https://portal.azure.com (Optional, for cloud model storage)

**📖 Detailed setup guide**: [`SECURITY_SETUP.md`](SECURITY_SETUP.md)

**✅ Test your setup**: `python test_api_setup.py`

**🆕 Note for Windows users:** All command line scripts automatically
reconfigure the console encoding to UTF-8 so emoji output works without
`UnicodeEncodeError`.

### 3. Run Pipeline
```bash
# Process documents and build indexes (includes enhanced country extraction)
python automated_pipeline.py

# Enable ultra-fast mode (5-20x faster queries, now integrated!)
python automated_pipeline.py  # Ultra-fast indexes built automatically

# Launch application
streamlit run cluster_qa_app.py
# (canonical UI implementation lives at apps/streamlit/cluster_qa_app.py)
```

### 4. Start Exploring
- 📊 **Browse clusters** of 1,531 climate documents across 120+ countries
- 🔍 **Ask questions** about climate policies and strategies  
- ⚡ **Generate reports** with automated summaries and visualizations
- 🌍 **Analyze countries** with comprehensive country-level clustering
- ⚡ **Ultra-fast queries** with 5-20x performance boost (when enhanced indexes are built)
- 📅 **Publication year filtering** for temporal analysis
- 🤖 **Batch processing** with headless workflow capabilities

## 🎯 What You Get

- **1,531 Documents**: International agreements, NDCs, UNFCCC reports
- **120+ Countries**: Enhanced extraction covering major economies and developing nations
- **Smart Clustering**: Semantic grouping using UMAP + HDBSCAN with country-level analysis
- **Advanced Country Detection**: Multi-language pattern matching with 97%+ assignment accuracy
- **Dual Q&A Modes**: Standard and ultra-fast (5-20x speedup) pipelines
- **Enhanced RAG Engine**: Intelligent chunking, document summaries, and pre-computed embeddings
- **Cross-Platform**: Windows, Mac, Linux with automatic setup
- **8GB+ RAM Support**: Auto-detects hardware and optimizes performance
- **🆕 Enhanced Clustering**: Configurable clustering with multiple algorithms (KMeans, HDBSCAN, Agglomerative)
- **🆕 Automated Publication Year Workflow**: Comprehensive year extraction and management
- **🆕 Headless Processing**: Batch processing capabilities without web interface

## 📊 Enhanced Country Coverage

The system includes advanced country extraction capabilities:

### **Coverage Statistics**
- **Documents with Countries**: 97.6% assignment rate (1,495/1,531 documents)
- **Countries Detected**: 120+ countries including all major economies
- **Key Countries Included**: USA (192 docs), Russia (26 docs), China (19 docs), Spain (19 docs), Italy, Germany, France, UK, and more

### **Technical Features**
- **Multi-language Support**: English, Spanish, French, Portuguese patterns
- **URL Decoding**: Handles encoded filenames (`United%20States` → `United States`)
- **Context Validation**: NDC/INDC pattern matching for accuracy
- **Fuzzy Matching**: Handles country name variations and abbreviations
- **Comprehensive Database**: 200+ country variations, codes, and translations

## 🆕 New Features

### **Enhanced Clustering System**
```bash
# Configurable clustering with multiple algorithms
python enhanced_country_clustering.py --clusters 15 --method kmeans
python enhanced_country_clustering.py --clusters 20 --method hdbscan
python enhanced_country_clustering.py --clusters 10 --method agglomerative
```

**Features:**
- **Multiple Algorithms**: KMeans, HDBSCAN, Agglomerative clustering
- **Configurable Clusters**: Specify exact number of clusters
- **Enhanced Performance**: Improved UMAP parameters and intelligent text chunking
- **Better Visualization**: Enhanced cluster assignments and statistics

### **Automated Publication Year Workflow**
```bash
# Scan for new documents
python auto_publication_year_workflow.py --scan-only

# Process new documents automatically
python auto_publication_year_workflow.py

# Run scheduled workflow with notifications
python run_scheduled_workflow.py
```

**Features:**
- **Automatic Detection**: Scans document directories for new files
- **Comprehensive Extraction**: PDF metadata, text patterns, headers/footers
- **Multi-language Support**: English, Spanish, French, Chinese patterns
- **Confidence Scoring**: Ranked results with confidence levels
- **Automated Updates**: Updates CSV file with new results
- **Backup System**: Automatic backups before updates

### **Headless Batch Processing**
```bash
# Create questions file
echo "What are the main climate policy challenges?" > questions.txt

# Process with headless workflow (legacy wrapper path)
python working_headless_processor.py questions.txt output.txt \
  --format bullet_points \
  --cluster-id 5 \
  --debug
```

**Features:**
- **Full Configuration**: All Streamlit controls as command-line arguments
- **Multiple Formats**: Detailed, summary, bullet points, comparative, technical
- **Model Selection**: DeepSeek, TinyLlama, GPT-4o, and more
- **Batch Processing**: Process multiple questions efficiently
- **Debug Mode**: Detailed logging and progress information

## 📚 Complete Documentation

**📖 For comprehensive system documentation, see [`docs/`](docs/) folder:**

- **[📋 CORE FEATURES](docs/CORE_FEATURES.md)** - **Complete guide to ALL system capabilities**
- **[🚀 Getting Started](docs/GETTING_STARTED.md)** - Detailed setup instructions
- **[🪟 Windows Setup](docs/WINDOWS_SETUP.md)** - Windows-specific installation guide
- **[🔧 Debug Guide](docs/DEBUG_GUIDE.md)** - Troubleshooting and debugging
- **[📝 Documentation Index](docs/README.md)** - Navigation to all guides
- **[🆕 Automated Workflow](docs/AUTOMATED_WORKFLOW_README.md)** - Publication year automation
- **[🆕 Publication Year Filtering](docs/PUBLICATION_YEAR_FILTERING_GUIDE.md)** - Temporal analysis guide
- **[🆕 Headless Workflow](docs/HEADLESS_STREAMLIT_WORKFLOW.md)** - Batch processing guide

## 🏷️ Key Features

### **Advanced Analysis**
- **Interactive clustering** with UMAP projections and choropleth maps
- **Country-level aggregation** for national policy analysis with comprehensive coverage
- **Cross-cluster analysis** comparing policies across document groups  
- **Signed similarity graphs** showing relationships between clusters
- **🆕 Enhanced clustering** with configurable algorithms and parameters

### **Enhanced Country Processing**
- **Automated Pipeline**: Built-in country extraction in `automated_pipeline.py`
- **High Accuracy**: 97.6% document assignment rate
- **Global Coverage**: Includes developing nations, small island states, and major economies
- **Verification Tools**: Built-in validation with `final_verification.py`

### **AI Integration**
- **Multiple models**: DeepSeek R1 (14B/7B), TinyLlama, GPT-4o
- **Automatic optimization**: RAM detection with model recommendations
- **Local + cloud options**: Offline operation or OpenAI API integration

### **Export & Reporting**
- **PDF/Markdown reports** with embedded visualizations
- **Cross-cluster comparison** reports with network analysis
- **Data export** for external analysis (CSV, network graphs)

### **🆕 Publication Year Management**
- **Comprehensive Extraction**: Multiple methods with confidence scoring
- **Automated Workflow**: Detects and processes new documents
- **Temporal Analysis**: Filter documents by publication year
- **Multi-language Support**: Handles various date formats and languages

### **🆕 Batch Processing**
- **Headless Operation**: Full functionality without web interface
- **Multiple Formats**: Detailed, summary, bullet points, comparative, technical
- **Efficient Processing**: Model loading once for entire batch
- **Debug Support**: Detailed logging and progress tracking

## 🔧 Pipeline Commands

### **Full Pipeline** (Recommended)
```bash
# Complete pipeline with standard + enhanced indexes
python automated_pipeline.py

# Enhanced indexes only (ultra-fast RAG mode)
python automated_pipeline.py --enhanced-only

# Skip enhanced indexes (standard mode only)
python automated_pipeline.py --skip-enhanced
```

### **🆕 Enhanced Clustering**
```bash
# Run enhanced clustering with KMeans
python enhanced_country_clustering.py --clusters 15 --method kmeans

# Run with HDBSCAN for automatic cluster detection
python enhanced_country_clustering.py --method hdbscan --min-cluster-size 2

# Run with Agglomerative clustering
python enhanced_country_clustering.py --clusters 10 --method agglomerative
```

### **🆕 Publication Year Workflow**
```bash
# Scan for new documents
python auto_publication_year_workflow.py --scan-only

# Process new documents
python auto_publication_year_workflow.py

# Run scheduled workflow
python run_scheduled_workflow.py
```

### **🆕 Headless Processing**
```bash
# Basic batch processing
python working_headless_processor.py questions.txt output.txt

# Advanced configuration
python working_headless_processor.py questions.txt output.txt \
  --format detailed \
  --model deepseek-r1-distill-qwen-14b \
  --pipeline ultra_fast \
  --top-k 10 \
  --max-tokens 6000 \
  --debug
```

### **Individual Steps**
```bash
# Text extraction with enhanced country detection
python automated_pipeline.py --skip-clustering --skip-indexing --skip-enhanced --skip-app

# Clustering only
python automated_pipeline.py --skip-extraction --skip-indexing --skip-enhanced --skip-app

# Standard indexes only
python automated_pipeline.py --skip-extraction --skip-clustering --skip-enhanced --skip-app

# Enhanced indexes only (ultra-fast RAG)
python automated_pipeline.py --skip-extraction --skip-clustering --skip-indexing --skip-app

# Verification
python final_verification.py
```

### **Offline Mode**
```bash
# Run completely offline (skip all Azure uploads)
python automated_pipeline.py --offline

# Force offline for Hugging Face models
export HF_HUB_OFFLINE=1
python automated_pipeline.py --offline
```

### **Ultra-Fast RAG Features**
When enhanced indexes are built, the system provides:
- **5-20x faster query performance** through pre-computed embeddings
- **Intelligent text chunking** with sentence boundary awareness and overlap
- **Document summaries** for quick overview queries
- **Rich metadata** including extracted titles, countries, and relationships
- **Dual search modes**: Detailed (chunks) and Overview (summaries)

## 🔧 Offline Mode

The system works completely offline once set up:

```bash
export HF_HUB_OFFLINE=1  # Optional: Force offline mode
```

See [`OFFLINE_MODE.md`](OFFLINE_MODE.md) for details.

## 🐛 Quick Troubleshooting

### **API Key Issues**
- **"OpenAI API error: Error code: 401"**: Check `.streamlit/secrets.toml` has your real API key (not template)
- **"your-ope************here" in errors**: Replace template values in `.streamlit/secrets.toml`
- **"OpenAI API key not configured"**: Set API key in both Streamlit secrets AND environment variables

### **Common Issues**
- **Missing data**: Run `python check_data.py` to verify setup
- **Country coverage**: Run `python final_verification.py` to check extraction results
- **Slow performance**: Run `python prepare_enhanced_index.py`
- **Windows issues**: Use `setup_environment.bat` for automated setup
- **Memory issues**: System auto-detects RAM and suggests compatible models

**For detailed troubleshooting**: See [`docs/DEBUG_GUIDE.md`](docs/DEBUG_GUIDE.md)

## 📊 System Requirements

- **RAM**: 8GB minimum (TinyLlama), 16GB+ recommended (DeepSeek)
- **Storage**: ~5GB for models and indexes
- **Platform**: Windows 10+, macOS, Linux
- **GPU**: Optional (CUDA/MPS supported)

---

**📖 Complete documentation**: [`docs/`](docs/) folder  
**❓ Need help?** Run `python final_verification.py` or check [`docs/README.md`](docs/README.md) 
