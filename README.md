# Semantics Explorer - Multilingual Embedding Visualization

This repository contains the complete implementation of the Semantics Explorer application described in the arXiv paper "Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings".

## Overview

A Streamlit-based interactive application for exploring the geometric structure of multilingual word embeddings across multiple embedding models and multiple dimensionality reduction methods.

## Quick Start

```bash
# create virtual env
conda create -n zinets
conda activate zinets

# clone source code
git clone git@github.com:digital-duck/semantics-explorer.git
cd semantics-explorer/src 

# setup
pip install -r requirements.txt

# run app
cd src
streamlit run Welcome.py
```

## Features

- **Interactive Visualization**: Real-time 2D/3D embedding exploration
- **Multiple Models**: Support for both local (Ollama) and cloud-based embedding models
- **Multiple Dimensionality Reduction Methods**: PHATE, t-SNE, UMAP, and more
- **Multi-lingual Support**: Optimized for Chinese-English language pairs
- **Clustering Analysis**: Automated pattern detection in semantic space
- **Session Caching**: Improved performance for interactive session

## Folder Structure

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
└── requirements.txt       # Python dependencies
```


## Dependencies

See `requirements.txt` for complete dependencies. Key packages:
- Streamlit for the web interface
- Transformers for embedding models
- PHATE for manifold learning
- Plotly for interactive visualization
- scikit-learn for clustering and dimensionality reduction

## Data

The application includes curated datasets of Chinese characters and text samples used in the paper analysis. See `src/data/README-DATASET.md` for detailed descriptions.


## Citation

@misc{gong_wg_2025,
title={Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings},
author={Gong, Wen G.},
year={2025},
eprint={XXXX.XXXXX},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

## License

MIT License - see LICENSE file for details.

