# Semantics Explorer - Multilingual Embedding Visualization

This repository contains the complete implementation of the Semantics Explorer application described in the arXiv paper "Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings".

## Overview

A Streamlit-based interactive application for exploring the geometric structure of multilingual word embeddings across multiple embedding models and multiple dimensionality reduction methods. Features advanced 3D interactive visualization with robust Chinese character support.

## Quick Start

```bash
# Create virtual environment
conda create -n zinets python=3.11
conda activate zinets

# Clone source code
git clone git@github.com:digital-duck/semantics-explorer.git
cd semantics-explorer/src

# Setup dependencies
pip install -r requirements.txt

# Download LASER models (optional)
python -m laserembeddings download-models

# Install Ollama models (optional)
ollama pull snowflake-arctic-embed2
ollama pull bge-m3

# Run application
streamlit run Welcome.py
```

## Features

### Core Functionality
- **Interactive 2D/3D Visualization**: Real-time embedding exploration with zoom, rotate, and pan
- **Multiple Embedding Models**: 15+ models including Sentence-BERT, E5-Base-v2, BGE-M3, Ollama models
- **Advanced Dimensionality Reduction**: PHATE, t-SNE, UMAP, Isomap, PCA, and more
- **Multi-lingual Support**: Optimized for Chinese-English language pairs with proper tokenization
- **Intelligent Clustering**: K-means clustering with quality metrics and boundary visualization

### User Experience
- **Enhanced Image Management**: Search, filter, and organize generated visualizations
- **Session Persistence**: Save and load text inputs and visualization settings
- **Publication-Quality Export**: High-DPI PNG, SVG, PDF export with customizable styling
- **Responsive Design**: Works on desktop and tablet devices

### Technical Improvements (v2.8)
- **Streamlit 1.50.x Compatibility**: Fixed widget key/session state consistency issues
- **Robust Chinese Character Processing**: Enhanced tokenization and NaN error handling
- **3D Plotting Stability**: Fixed Plotly griddash property errors for 3D scenes
- **Memory Optimization**: Efficient caching and resource management

## Supported Models

### Hugging Face Models
- **Sentence-BERT Multilingual** (Default) - Reliable cross-lingual performance
- **E5-Base-v2** - Balanced accuracy-speed (⚠️ May have issues with Chinese text)
- **BGE-Large-EN-v1.5** - High-accuracy English embeddings
- **Multilingual-E5-Large** - Advanced multilingual capabilities
- **XLM-RoBERTa-Large** - Strong cross-lingual representations

### Ollama Models (Local)
- **BGE-M3** - Excellent for Chinese-English semantic alignment
- **Snowflake-Arctic-Embed2** - Optimized multilingual performance
- **EmbeddingGemma** - Google's 300M parameter embedding model
- **Nomic-Embed-Text** - Open-source multimodal embeddings

## Dimensionality Reduction Methods

- **PHATE** (Default) - Preserves local and global structure, ideal for manifold learning
- **t-SNE** - Non-linear, good for cluster identification
- **UMAP** - Balanced global/local structure preservation
- **Isomap** - Manifold learning with geodesic distances
- **PCA** - Linear, interpretable components
- **MDS** - Preserves pairwise distances
- **Spectral Embedding** - Graph-based dimensionality reduction

## Architecture

```
semantics-explorer/
├── src/                        # Main application code
│   ├── Welcome.py             # Streamlit entry point
│   ├── config.py              # Model definitions and settings
│   ├── pages/                 # Multi-page interface
│   │   ├── 1_🧭_Semantics_Explorer.py     # Main visualization
│   │   ├── 2_🔍_Semantics_Explorer-Dual_View.py  # Dual-panel view
│   │   ├── 3_🖼️_Review_Images.py         # Image management
│   │   └── 9_🌐_Translator.py            # Translation tools
│   ├── models/                # Embedding model management
│   │   └── model_manager.py   # Factory pattern for models
│   ├── components/            # Reusable UI components
│   │   ├── embedding_viz.py   # Core visualization logic
│   │   ├── plotting.py        # Plotly-based charts
│   │   └── clustering.py      # Clustering algorithms
│   ├── services/              # External integrations
│   │   ├── google_translate.py
│   │   └── tts_service.py
│   ├── utils/                 # Utility functions
│   │   └── error_handling.py  # Robust error management
│   └── data/                  # Curated datasets
│       └── input/             # Text input files
└── requirements.txt           # Python dependencies
```

## Usage Examples

### Basic Semantic Exploration
```python
# Load Chinese-English word pairs
chinese_text = "你好\n再见\n朋友"
english_text = "hello\ngoodbye\nfriend"

# Select model: Sentence-BERT Multilingual
# Choose method: PHATE
# Set dimensions: 2D or 3D
# Enable clustering for pattern discovery
```

### Advanced Analysis
```python
# Use specialized datasets
- Chinese character families (子-family, 木-family)
- Semantic categories (colors, animals, emotions)
- Cross-lingual concept pairs

# Publication-quality visualization
- Enable publication mode
- Set high DPI (300+)
- Export as SVG/PDF
- Customize fonts and styling
```

## Known Issues & Solutions

### Model-Specific Notes
- **E5-Base-v2**: May produce NaN errors with Chinese text → Use Sentence-BERT Multilingual
- **Ollama Models**: Require local Ollama server running on port 11434
- **Large Models**: May need >8GB RAM for optimal performance

### Troubleshooting
```bash
# If embeddings fail
pip install --upgrade transformers torch

# If Ollama models unavailable
ollama serve  # Start Ollama server

# If Chinese characters display incorrectly
# Ensure UTF-8 encoding in input files
```

## Performance Optimization

- **Caching**: Models and embeddings cached across sessions
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Automatic cleanup of large objects
- **Progressive Loading**: Lazy model initialization

## Dataset Information

The application includes curated datasets used in the research:
- **Chinese character families**: Radical-based groupings (子, 木, 水, etc.)
- **Semantic categories**: Colors, animals, emotions, time concepts
- **Cross-lingual pairs**: Chinese-English translation equivalents
- **ASCII control sets**: For baseline comparisons

See `src/data/README-DATASET.md` for detailed descriptions.

## Citation

```bibtex
@misc{gong_wg_2025,
  title={Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings},
  author={Gong, Wen G.},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## License

MIT License - see LICENSE file for details.

---

**Version 2.8** - Updated for Streamlit 1.50.x compatibility with enhanced Chinese character support and 3D visualization improvements.