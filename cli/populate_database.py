import os
import sys
import logging
import argparse
from typing import List
from langchain_core.documents import Document
import dotenv

sys.path.append(os.path.dirname(__file__))

from courserag.config.rag_config import config
from courserag.core.document_loader import DocumentLoader
from courserag.core.vector_store import ChromaVectorStore
from courserag.utils.utils import is_cache_valid, load_chunks_from_cache, save_chunks_to_cache, clear_cache

from courserag.config.logging_config import setup_logging
logger = setup_logging()
dotenv.load_dotenv()


def check_openai_key():
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)


def get_chunks(data_dir: str, force_refresh: bool = False) -> List[Document]:
    if not force_refresh and is_cache_valid(data_dir):
        logger.info("Cache is valid, loading chunks from cache...")
        return load_chunks_from_cache()
    
    logger.info("Cache is invalid or force refresh requested, processing documents...")
    
    loader = DocumentLoader()
    documents = loader.load_documents(data_dir)
    
    if not documents:
        logger.warning(f"No documents found in {data_dir}")
        return []
    
    chunks = loader.split_documents(documents)
    
    save_chunks_to_cache(chunks, data_dir)
    
    return chunks


def populate_database(data_dir: str, force_refresh: bool = False, reset_db: bool = False) -> None:
    logger.info("Starting database population...")
    
    check_openai_key()
    
    vector_store = ChromaVectorStore()
    
    if reset_db:
        logger.info("Resetting database...")
        vector_store.delete_collection()
        clear_cache()
    
    vector_store.initialize()
    
    chunks = get_chunks(data_dir, force_refresh)
    
    if not chunks:
        logger.error("No chunks to add to database")
        return
    
    vector_store.populate_if_empty(chunks)
    
    info = vector_store.get_collection_info()
    logger.info(f"Database population complete. Collection info: {info}")


def main():
    parser = argparse.ArgumentParser(description="Populate Chroma database with course documents")
    
    parser.add_argument(
        "--data-dir",
        default=config.CLEAN_DATA_DIR,
        help=f"Directory containing documents (default: {config.CLEAN_DATA_DIR})"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force reprocessing of documents even if cache is valid"
    )
    
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the database before populating (WARNING: This will delete all existing data)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the document cache"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show database information and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        from courserag.config.logging_config import set_log_level
        set_log_level(logging.DEBUG)
    
    if args.clear_cache:
        logger.info("Clearing cache...")
        clear_cache()
        logger.info("Cache cleared successfully")
        return
    
    if args.info:
        try:
            vector_store = ChromaVectorStore()
            vector_store.initialize()
            info = vector_store.get_collection_info()
            print(f"Database Information:")
            print(f"  Collection: {info.get('name', 'Unknown')}")
            print(f"  Document count: {info.get('count', 'Unknown')}")
            print(f"  Metadata: {info.get('metadata', {})}")
        except Exception as e:
            print(f"Error getting database info: {e}")
        return
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if args.reset_db:
        response = input("WARNING: This will delete all existing data in the database. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Operation cancelled")
            return
    
    try:
        populate_database(
            data_dir=args.data_dir,
            force_refresh=args.force_refresh,
            reset_db=args.reset_db
        )
        logger.info("Database population completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during database population: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 