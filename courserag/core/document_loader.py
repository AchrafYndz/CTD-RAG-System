import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import CSVLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from courserag.config.rag_config import config
from courserag.utils.utils import get_file_info, optimize_chunks

from courserag.config.logging_config import get_logger
logger = get_logger()


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.loader_map = {
            'txt': self._load_text,
            'pdf': self._load_pdf,
            'md': self._load_markdown,
            'csv': self._load_csv,
        }
    
    def load_documents(self, data_dir: str) -> List[Document]:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        
        logger.info(f"Loading documents from {data_dir}...")
        documents = []
        failed_files = []
        processed_files = 0
        
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if (file_name.startswith('.') or 
                    file_name.endswith('.pkl') or 
                    file_name.endswith('.json')):
                    continue
                
                file_path = os.path.join(root, file_name)
                
                try:
                    docs = self._load_single_file(file_path, data_dir)
                    if docs:
                        documents.extend(docs)
                        processed_files += 1
                        logger.debug(f"Successfully loaded {file_path}")
                    else:
                        logger.warning(f"No documents loaded from {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    failed_files.append(file_path)
        
        logger.info(f"Loaded {len(documents)} documents from {processed_files} files")
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        optimized_chunks = optimize_chunks(chunks)
        
        logger.info(f"Created {len(optimized_chunks)} unique chunks from {len(documents)} documents")
        return optimized_chunks
    
    def _load_single_file(self, file_path: str, data_dir: str) -> Optional[List[Document]]:
        file_info = get_file_info(file_path)
        extension = file_info.get('extension', '').lower()
        
        if extension not in self.loader_map:
            logger.debug(f"Unsupported file type: {extension} for {file_path}")
            return None
        
        loader_func = self.loader_map[extension]
        docs = loader_func(file_path)
        
        if not docs:
            return None
        
        relative_path = os.path.relpath(file_path, data_dir)
        for doc in docs:
            doc.metadata.update({
                'source': relative_path,
                'file_type': extension,
                'file_name': file_info.get('name', ''),
                'file_size': file_info.get('size', 0),
                'modified_time': file_info.get('modified', 0)
            })
        
        return docs
    
    def _load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        for i, page in enumerate(pages):
            page.metadata['page_number'] = i + 1
            page.metadata['total_pages'] = len(pages)
        
        return pages
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    
    def _load_csv(self, file_path: str) -> List[Document]:
        loader = CSVLoader(file_path)
        return loader.load()


def load_documents(data_dir: str) -> List[Document]:
    loader = DocumentLoader()
    return loader.load_documents(data_dir)


def split_documents(documents: List[Document]) -> List[Document]:
    loader = DocumentLoader()
    return loader.split_documents(documents) 