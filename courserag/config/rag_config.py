from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    PREPROCESSED_DATA_DIR: str = "data/processed"
    CHROMA_DB_PATH: str = "./chroma_db"
    CHUNKS_CACHE_PATH: str = "data/cache/cached_chunks.pkl"
    CACHE_METADATA_PATH: str = "data/cache/cache_metadata.json"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    LLM_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0
    
    RETRIEVAL_K: int = 3
    SEARCH_TYPE: str = "mmr"
    MMR_LAMBDA: float = 0.7
    FETCH_K_MULTIPLIER: int = 2
    
    COLLECTION_NAME: str = "ctr_course_documents"
    
    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {
            "k": self.RETRIEVAL_K,
            "lambda_mult": self.MMR_LAMBDA,
            "fetch_k": self.RETRIEVAL_K * self.FETCH_K_MULTIPLIER
        }

config = RAGConfig() 