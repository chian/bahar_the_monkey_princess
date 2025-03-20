import torch
import torch.nn.functional as F
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Dict, Any

# Configuration for the embedding model
EMBEDDINGS_CONFIG = {
    "model_id": "nvidia/NV-Embed-v2",
    "batch_size": 100,  # Default batch size for processing multiple texts
    "max_length": 32768  # Maximum sequence length supported by the model
}

class VectorStore:
    """Manages vector storage and retrieval for embedded chunks"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def build_from_chunks(self, chunks: List[Dict[str, Any]], embedder: 'LocalEmbedder') -> None:
        """
        Build vector store from text chunks using the provided embedder
        Args:
            chunks: List of chunk dictionaries containing at least 'text' key
            embedder: LocalEmbedder instance to use for embedding
        """
        if not chunks:
            print("No chunks provided to build vector store")
            return
            
        print(f"Creating vector store with embeddings for {len(chunks)} chunks...")
        
        # Get text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings in batches
        all_embeddings = []
        batch_size = embedder.config["batch_size"]
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embedder.embed_texts(batch)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy for FAISS
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        
        # Store everything
        self.index = index
        self.embeddings = embeddings_np
        self.chunks = chunks
        
        print(f"Vector store created with {len(chunks)} chunks")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for similar chunks
        Args:
            query_embedding: Query embedding to search with
            k: Number of results to return
        Returns:
            List of chunk dictionaries with distances
        """
        if self.index is None:
            return []
            
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['distance'] = float(dist)
                results.append(chunk)
                
        return results

class LocalEmbedder:
    def __init__(self, config: dict):
        """
        Initialize the local embedding model
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        # Initialize using SentenceTransformer for better compatibility
        self.model = SentenceTransformer(config["model_id"], trust_remote_code=True)
        self.model.max_seq_length = config["max_length"]
        self.model.tokenizer.padding_side = "right"
        
    def add_eos(self, input_examples: List[str]) -> List[str]:
        """
        Add EOS token to inputs as required by the model
        Args:
            input_examples: List of text strings
        Returns:
            List of text strings with EOS tokens added
        """
        return [example + self.model.tokenizer.eos_token for example in input_examples]
    
    def embed_texts(self, 
                   texts: List[str], 
                   instruction: Optional[str] = None) -> torch.Tensor:
        """
        Embed a list of texts, optionally with an instruction prefix
        Args:
            texts: List of text strings to embed
            instruction: Optional instruction prefix for queries
        Returns:
            Normalized embeddings as torch tensor
        """
        batch_size = self.config["batch_size"]
        
        if instruction:
            # Format with instruction prefix for queries
            texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        
        # Add EOS tokens and get embeddings
        texts = self.add_eos(texts)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True
        )
        
        return torch.from_numpy(embeddings)

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Embed a single query with appropriate instruction
        Args:
            query: Query string to embed
        Returns:
            Query embedding as torch tensor
        """
        instruction = "Given a question, retrieve passages that answer the question"
        return self.embed_texts([query], instruction=instruction)[0]
    
    def embed_passages(self, passages: List[str]) -> torch.Tensor:
        """
        Embed a list of passages without instruction
        Args:
            passages: List of passage strings to embed
        Returns:
            Passage embeddings as torch tensor
        """
        return self.embed_texts(passages)

    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> VectorStore:
        """
        Create a new vector store from chunks
        Args:
            chunks: List of chunk dictionaries containing at least 'text' key
        Returns:
            VectorStore instance
        """
        vector_store = VectorStore()
        vector_store.build_from_chunks(chunks, self)
        return vector_store

# Global embedder instance
_embedder = None

def get_embedder() -> LocalEmbedder:
    """
    Initialize and return the embedding model (singleton pattern)
    Returns:
        LocalEmbedder instance
    """
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder(EMBEDDINGS_CONFIG)
    return _embedder 