# models/model_manager.py
import streamlit as st
from abc import ABC, abstractmethod
import numpy as np
import requests
from typing import List, Optional
from utils.error_handling import (
    handle_errors, ModelNotFoundError, EmbeddingError,
)
from config import (
    OLLAMA_MODELS, MODEL_INFO,
)
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import transformers: {e}")
    # Fallback - only support Ollama models
    AutoTokenizer = None
    AutoModel = None

@st.cache_resource
def get_ollama_session():
    """Create a cached session for Ollama requests to improve performance"""
    session = requests.Session()
    return session

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en") -> np.ndarray:
        pass

class OllamaModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = get_ollama_session()
        
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en") -> Optional[np.ndarray]:
        embeddings = []
        for idx, text in enumerate(texts):
            try:
                response = self.session.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    if embedding:
                        embeddings.append(embedding)
                else:
                    st.warning(f"Failed to get embedding for text: {text}")
            except Exception as e:
                st.error(str(e))
                    
        return np.array(embeddings) if embeddings else None

class HuggingFaceModel(EmbeddingModel):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def _lazy_load(self):
        if not self.tokenizer:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("Transformers library not available due to compatibility issues")
            
            # T5 models disabled due to torch compatibility issues
            # if self.model_name == "mT5":
            #     # Load the mT5 tokenizer and encoder model
            #     self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
            #     self.model = T5EncoderModel.from_pretrained("google/mt5-small")
            # else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en") -> Optional[np.ndarray]:
        # LASER support disabled due to torch compatibility issues
        # if self.model_name == "LASER":
        #     try:
        #         from laserembeddings import Laser
        #         laser = Laser()
        #         return laser.embed_sentences(texts, lang=lang)
        #     except Exception as e:
        #         st.error(f"Unsupported model: {self.model_name}")
        #         return None

        self._lazy_load()
        embeddings = []
        for text in texts:
            # Skip empty texts that could cause NaN issues
            if not text or not text.strip():
                st.warning(f"Empty text detected, using zero embedding")
                # Create a zero embedding with the model's hidden size
                dummy_inputs = self.tokenizer("dummy", return_tensors="pt")
                dummy_outputs = self.model(**dummy_inputs)
                zero_embedding = np.zeros_like(dummy_outputs.last_hidden_state.mean(dim=1).detach().numpy())
                embeddings.append(zero_embedding)
                continue

            # Chinese character preprocessing for E5 models
            if self.model_name in ["E5-Base-v2"] and any('\u4e00' <= char <= '\u9fff' for char in text):
                # Add a space before Chinese text for better E5 tokenization
                text = f" {text}"

            # Tokenize the input text with proper attention masks
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Debug tokenization issues with Chinese text
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                token_count = inputs['input_ids'].size(1)
                if token_count > 100:  # Very long tokenization might indicate issues
                    st.warning(f"Chinese text '{text[:20]}...' tokenized to {token_count} tokens")

            # Get the encoder outputs
            outputs = self.model(**inputs)

            # Proper mean pooling using attention masks to avoid NaN from padding tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Mask out padding tokens and compute mean
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)

            # Avoid division by zero
            sum_mask = sum_mask.clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # Check for NaN values and extreme values before adding to embeddings
            embedding_array = mean_pooled.detach().numpy()
            if np.isnan(embedding_array).any():
                st.warning(f"NaN detected in embedding for text: '{text[:50]}...', using zero embedding")
                embedding_array = np.zeros_like(embedding_array)
            elif np.isinf(embedding_array).any():
                st.warning(f"Infinite values detected in embedding for text: '{text[:50]}...', clipping values")
                embedding_array = np.clip(embedding_array, -10.0, 10.0)
            elif np.abs(embedding_array).max() > 100:
                st.warning(f"Extreme values detected in embedding for text: '{text[:50]}...', normalizing")
                # L2 normalize to prevent extreme values
                norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
                embedding_array = embedding_array / (norm + 1e-8)

            embeddings.append(embedding_array)

        return np.vstack(embeddings)

def get_model(model_name: str) -> EmbeddingModel:
    """Factory function for creating embedding models"""
    if model_name in OLLAMA_MODELS:
        return OllamaModel(OLLAMA_MODELS[model_name]["path"])
    elif model_name in MODEL_INFO:
        return HuggingFaceModel(model_name, MODEL_INFO[model_name]["path"])
    else:
        raise ModelNotFoundError(f"Model {model_name} not found")