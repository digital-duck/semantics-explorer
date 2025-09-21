# Semantics Explorer - Multilingual Embedding Visualization

This repository contains the complete implementation of the Semantics Explorer application described in the arXiv paper "Geometric Structures and Patterns of Meaning in Chinese Character Embeddings: A PHATE Manifold Analysis".

## Overview

A Streamlit-based interactive application for exploring the geometric structure of Chinese character embeddings across multiple embedding models and dimensionality reduction techniques.

## Quick Start

```bash
# Initialize the project
./init_project.sh

# Run the application
cd src && streamlit run Welcome.py
```

## Paper Reference

This codebase implements the analysis described in:
**"Geometric Structures and Patterns of Meaning in Chinese Character Embeddings: A PHATE Manifold Analysis"**

The application enables interactive exploration of:
- 6 embedding models (BGE-Base-ZH-v1.5, mBERT, XLM-R, DistilBERT, Jina-v2, Snowflake-Arctic)
- 8 dimensionality reduction methods (t-SNE, UMAP, MDS, PCA, Isomap, Spectral Embedding, LLE, PHATE)
- Structural vs. meaningful Chinese character filtering
- Cross-model and cross-method validation of geometric patterns

## Repository Structure

```
semantics-explorer/
├── src/                    # Main application code
│   ├── Welcome.py         # Streamlit entry point
│   ├── config.py          # Configuration and model definitions
│   ├── pages/             # Streamlit pages
│   ├── models/            # Embedding model management
│   ├── components/        # Reusable UI components
│   ├── services/          # External service integrations
│   ├── utils/             # Utility functions
│   └── data/              # Dataset collection
├── paper-datasets/        # Datasets referenced in the paper
├── requirements.txt       # Python dependencies
└── init_project.sh       # Project initialization script
```

## Features

- **Interactive Visualization**: Real-time 2D/3D embedding exploration
- **Multiple Models**: Support for both local (Ollama) and cloud-based embedding models
- **Advanced Dimensionality Reduction**: PHATE, t-SNE, UMAP, and more
- **Multilingual Support**: Optimized for Chinese-English language pairs
- **Clustering Analysis**: Automated pattern detection in semantic space
- **Session Caching**: Improved performance for repeated operations

## Dependencies

See `requirements.txt` for complete dependencies. Key packages:
- Streamlit for the web interface
- Transformers for embedding models
- PHATE for manifold learning
- Plotly for interactive visualization
- scikit-learn for clustering and dimensionality reduction

## Data

The application includes curated datasets of Chinese characters and text samples used in the paper analysis. See `src/data/README-DATASET.md` for detailed descriptions.

## Setup

```bash
conda activate zinets
cd ~/projects/digital-duck/semantics-explorer
```

## License

MIT License - see LICENSE file for details.
