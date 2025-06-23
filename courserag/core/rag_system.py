import os
import sys
from typing import Dict, Any
import dotenv

sys.path.append(os.path.dirname(__file__))

from courserag.config.rag_config import config
from courserag.core.document_loader import DocumentLoader
from courserag.core.vector_store import ChromaVectorStore
from courserag.core.rag_chain import RAGChain, ask_normal_gpt
from courserag.utils.utils import is_cache_valid, load_chunks_from_cache, save_chunks_to_cache

from courserag.config.logging_config import setup_logging
logger = setup_logging()
dotenv.load_dotenv()


class RAGSystem:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.CLEAN_DATA_DIR
        self.document_loader = DocumentLoader()
        self.vector_store = ChromaVectorStore()
        self.rag_chain = RAGChain()
        self._initialized = False
    
    def initialize(self, force_refresh: bool = False) -> 'RAGSystem':
        if self._initialized and not force_refresh:
            logger.info("RAG system already initialized")
            return self
        
        logger.info("Initializing RAG system...")
        
        self._check_openai_key()
        
        chunks = self._get_chunks(force_refresh)
        
        if not chunks:
            raise RuntimeError("No document chunks available")
        
        self.vector_store.initialize()
        self.vector_store.populate_if_empty(chunks)
        
        retriever = self.vector_store.as_retriever()
        self.rag_chain.setup_chain(retriever)
        
        self._initialized = True
        logger.info("RAG system initialization complete")
        
        return self
    
    def query(self, question: str) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        return self.rag_chain.query(question)
    
    def compare_with_normal_gpt(self, question: str) -> Dict[str, str]:
        rag_response = self.query(question)
        rag_answer = rag_response["answer"]
        
        normal_gpt_answer = ask_normal_gpt(question)
        
        return {
            "question": question,
            "rag_answer": rag_answer,
            "normal_gpt_answer": normal_gpt_answer,
            "sources": rag_response.get("sources", [])
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        collection_info = self.vector_store.get_collection_info()
        
        return {
            "data_directory": self.data_dir,
            "collection_info": collection_info,
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL,
                "retrieval_k": config.RETRIEVAL_K
            },
            "initialized": self._initialized
        }
    
    def _check_openai_key(self) -> None:
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
    
    def _get_chunks(self, force_refresh: bool = False):
        if not force_refresh and is_cache_valid(self.data_dir):
            logger.info("Loading chunks from cache...")
            return load_chunks_from_cache()
        
        logger.info("Processing documents...")
        
        documents = self.document_loader.load_documents(self.data_dir)
        if not documents:
            raise RuntimeError(f"No documents found in {self.data_dir}")
        
        chunks = self.document_loader.split_documents(documents)
        
        save_chunks_to_cache(chunks, self.data_dir)
        
        return chunks


def load_documents(data_dir: str):
    loader = DocumentLoader()
    return loader.load_documents(data_dir)


def split_documents(documents):
    loader = DocumentLoader()
    return loader.split_documents(documents)


def get_vector_store(chunks):
    store = ChromaVectorStore()
    vector_store = store.initialize()
    store.populate_if_empty(chunks)
    return vector_store


def setup_rag_chain(vector_store):
    rag = RAGChain()
    retriever = vector_store.as_retriever()
    rag.setup_chain(retriever)
    return rag


def ask_rag(question: str, rag_chain) -> str:
    if hasattr(rag_chain, 'query'):
        response = rag_chain.query(question)
        return response["answer"]
    else:
        response = rag_chain.invoke({"input": question})
        return response["answer"]


def main():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    info = rag_system.get_system_info()
    logger.info(f"System initialized with {info['collection_info']['count']} documents")
    
    questions = [
        "Who was in Martina's group for her Presentation about 'Can LLMS keep a secret'?",
        "What are the key findings regarding LLaMA3 quantization?",
        "What is attention mechanism in transformers?"
    ]
    
    for question in questions:
        print(f"\n--- Question: {question} ---")
        
        comparison = rag_system.compare_with_normal_gpt(question)
        
        print(f"\n📚 CourseGPT (RAG) Answer:")
        print(comparison["rag_answer"])
        
        if comparison["sources"]:
            print(f"\n📖 Sources: {', '.join(comparison['sources'])}")
        
        print(f"\n🤖 Normal GPT Answer:")
        print(comparison["normal_gpt_answer"])
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()