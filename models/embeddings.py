import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import logging
from config.config import config

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding.model_name
        self.dimension = config.embedding.dimension
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            if "nomic" in self.model_name.lower():
                self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

class FAISSVectorStore:
    def __init__(self, dimension: Optional[int] = None, persist_directory: Optional[str] = None):
        self.dimension = dimension or config.embedding.dimension
        self.persist_directory = persist_directory or config.vector_db.persist_directory
        self.index = None
        self.texts = []
        self.metadatas = []
        self.embedding_model = EmbeddingModel()
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._initialize_index()
    
    def _initialize_index(self):
        try:
            if self._index_exists():
                self.load_index()
            else:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
                logger.info(f"Created new FAISS index with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _index_exists(self) -> bool:
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        return os.path.exists(index_path) and os.path.exists(metadata_path)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        if metadatas is None:
            metadatas = [{"id": i} for i in range(len(texts))]
        
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have same length")
        
        try:
            logger.info(f"Adding {len(texts)} documents to vector store")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Normalize for cosine similarity (required for IndexFlatIP)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings.astype(np.float32))
            
            # Store texts and metadata
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)
            
            logger.info(f"Successfully added {len(texts)} documents. Total: {len(self.texts)}")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= threshold:  # -1 means not found
                    results.append({
                        "text": self.texts[idx],
                        "metadata": self.metadatas[idx],
                        "score": float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise
    
    def save_index(self):
        try:
            # Save FAISS index
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                    "dimension": self.dimension
                }, f)
            
            logger.info(f"Saved index to {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self):
        try:
            # Load FAISS index
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.metadatas = data["metadatas"]
                self.dimension = data["dimension"]
            
            logger.info(f"Loaded index with {len(self.texts)} documents from {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.texts),
            "dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "persist_directory": self.persist_directory
        }
    
    def clear(self):
        self.texts = []
        self.metadatas = []
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info("Cleared vector store")

# Create global instance
vector_store = FAISSVectorStore()