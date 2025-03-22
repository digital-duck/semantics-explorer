"""
PROMPT (2024-12-23 Gemini 2.0 Flash Experimental):

can you write me an streamlit app in python called semantic analyzer. 
It should have the following features: 
1) given any 2 phrases or words in chinese or English, show their embedding similarity
2) one can choose from a list of common multilingual embedding models
3) visualize the semantic similarity analysis in vector space, if Gemini Flash model is not the best for coding, please suggest alternative model to build this multilingual semantics analyzer tool

using a smaller LLM and the transformers library

"""
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Model options
model_options = {
    "Multilingual MPNet": "all-mpnet-base-v2",
    "Multilingual Paraphrase MPNet": "paraphrase-multilingual-mpnet-base-v2",
    "Multilingual XLM-R": "stsb-xlm-r-multilingual",
    "English MiniLM": "all-MiniLM-L6-v2"
}

st.title("Semantic Analyzer")

# Model selection
selected_model = st.selectbox("Select Embedding Model", list(model_options.keys()))
model = SentenceTransformer(model_options[selected_model])

# Input phrases
phrase1 = st.text_input("Enter Phrase/Word 1", "")
phrase2 = st.text_input("Enter Phrase/Word 2", "")

if phrase1 and phrase2:
    try:
        # Generate embeddings
        embeddings = model.encode([phrase1, phrase2])

        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        st.write(f"Cosine Similarity: {similarity:.4f}")

        # Visualization (using Plotly)
        df = pd.DataFrame(embeddings, index=[phrase1, phrase2])
        fig = go.Figure(data=go.Scatter3d(
            x=df[0], y=df[1], z=df[2],
            mode='markers+text',
            text=df.index,
            marker=dict(size=8)
        ))
        fig.update_layout(title="Embedding Visualization (First 3 Dimensions)",
                          scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3'))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check your input and model selection. Some models might not be suitable for single words.")