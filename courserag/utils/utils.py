import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from courserag.config.rag_config import config

from courserag.config.logging_config import get_logger
logger = get_logger()


def get_directory_hash(data_dir: str) -> str:
    file_hashes = []
    
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if (file_name.startswith('.') or 
                file_name.endswith('.pkl') or 
                file_name.endswith('.json')):
                continue
                
            file_path = os.path.join(root, file_name)
            try:
                stat = os.stat(file_path)
                file_info = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
                file_hashes.append(hashlib.md5(file_info.encode()).hexdigest())
            except OSError as e:
                logger.warning(f"Could not stat file {file_path}: {e}")
                continue
    
    file_hashes.sort()
    combined_hash = hashlib.md5(''.join(file_hashes).encode()).hexdigest()
    return combined_hash


def is_cache_valid(data_dir: str) -> bool:
    if not os.path.exists(config.CHUNKS_CACHE_PATH) or not os.path.exists(config.CACHE_METADATA_PATH):
        logger.info("Cache files do not exist")
        return False
    
    try:
        with open(config.CACHE_METADATA_PATH, 'r') as f:
            cache_metadata = json.load(f)
        
        current_hash = get_directory_hash(data_dir)
        cached_hash = cache_metadata.get('directory_hash')
        
        if current_hash == cached_hash:
            logger.info("Cache is valid")
            return True
        else:
            logger.info("Directory has changed, cache invalid")
            return False
            
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Cache metadata corrupted: {e}")
        return False


def save_chunks_to_cache(chunks: List[Document], data_dir: str) -> None:
    os.makedirs(os.path.dirname(config.CHUNKS_CACHE_PATH), exist_ok=True)
    
    with open(config.CHUNKS_CACHE_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    
    metadata = {
        'directory_hash': get_directory_hash(data_dir),
        'chunk_count': len(chunks),
        'config_hash': _get_config_hash(),
        'created_at': os.path.getctime(config.CHUNKS_CACHE_PATH) if os.path.exists(config.CHUNKS_CACHE_PATH) else None
    }
    
    with open(config.CACHE_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to cache: {config.CHUNKS_CACHE_PATH}")


def load_chunks_from_cache() -> List[Document]:
    with open(config.CHUNKS_CACHE_PATH, 'rb') as f:
        chunks = pickle.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks from cache: {config.CHUNKS_CACHE_PATH}")
    return chunks


def clear_cache() -> None:
    for cache_file in [config.CHUNKS_CACHE_PATH, config.CACHE_METADATA_PATH]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"Removed cache file: {cache_file}")
    
    logger.info("Cache cleared successfully")


def get_cache_info() -> Optional[Dict[str, Any]]:
    if os.path.exists(config.CACHE_METADATA_PATH):
        with open(config.CACHE_METADATA_PATH, 'r') as f:
            return json.load(f)
    return None


def optimize_chunks(documents: List[Document]) -> List[Document]:
    unique_chunks = []
    seen_hashes = set()
    
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(doc)
    
    removed_count = len(documents) - len(unique_chunks)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate chunks")
    
    return unique_chunks


def _get_config_hash() -> str:
    config_str = f"{config.CHUNK_SIZE}:{config.CHUNK_OVERLAP}:{config.EMBEDDING_MODEL}"
    return hashlib.md5(config_str.encode()).hexdigest()


def ensure_directory_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    try:
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': file_path.split('.')[-1].lower(),
            'name': os.path.basename(file_path)
        }
    except OSError:
        return {} 