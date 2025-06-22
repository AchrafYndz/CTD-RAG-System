from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

from courserag.config.rag_config import config
from courserag.utils.utils import ensure_directory_exists

from courserag.config.logging_config import get_logger
logger = get_logger()


class ChromaVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            max_retries=3
        )
        self.vector_store: Optional[Chroma] = None
        self._client: Optional[chromadb.PersistentClient] = None
    
    def initialize(self) -> Chroma:
        if self.vector_store is not None:
            return self.vector_store
        
        ensure_directory_exists(config.CHROMA_DB_PATH)
        
        self._client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        
        self.vector_store = Chroma(
            client=self._client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        
        logger.info(f"Initialized Chroma vector store at {config.CHROMA_DB_PATH}")
        return self.vector_store
    
    def populate_if_empty(self, chunks: List[Document]) -> None:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        try:
            collection = self._client.get_collection(config.COLLECTION_NAME)
            doc_count = collection.count()
            
            if doc_count == 0:
                logger.info("Vector store is empty, populating with documents...")
                self._add_documents_batch(chunks)
                logger.info(f"Added {len(chunks)} chunks to vector store")
            else:
                logger.info(f"Vector store already contains {doc_count} documents")
                
        except Exception as e:
            logger.info(f"Creating new collection: {e}")
            self._add_documents_batch(chunks)
            logger.info(f"Created collection and added {len(chunks)} chunks")
    
    def add_documents(self, chunks: List[Document]) -> List[str]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        logger.info(f"Adding {len(chunks)} new documents to vector store...")
        return self._add_documents_batch(chunks)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        k = k or config.RETRIEVAL_K
        return self.vector_store.similarity_search(query, k=k)
    
    def as_retriever(self):
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        return self.vector_store.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs=config.search_kwargs
        )
    
    def get_collection_info(self) -> dict:
        if self._client is None:
            return {"error": "Client not initialized"}
        
        try:
            collection = self._client.get_collection(config.COLLECTION_NAME)
            return {
                "name": config.COLLECTION_NAME,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self) -> None:
        if self._client is None:
            logger.warning("Client not initialized")
            return
        
        try:
            self._client.delete_collection(config.COLLECTION_NAME)
            logger.info(f"Deleted collection: {config.COLLECTION_NAME}")
            self.vector_store = None
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def _add_documents_batch(self, chunks: List[Document], batch_size: int = 100) -> List[str]:
        all_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                continue
        
        return all_ids


def get_vector_store(chunks: List[Document]) -> Chroma:
    store = ChromaVectorStore()
    vector_store = store.initialize()
    store.populate_if_empty(chunks)
    return vector_store


def create_retriever(chunks: List[Document]):
    store = ChromaVectorStore()
    store.initialize()
    store.populate_if_empty(chunks)
    return store.as_retriever() 