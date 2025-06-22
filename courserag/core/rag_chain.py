from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from courserag.config.rag_config import config

from courserag.config.logging_config import get_logger
logger = get_logger()


class RAGChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            top_p=0.9
        )
        self.rag_chain = None
        self._setup_prompt()
    
    def _setup_prompt(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are CourseGPT, an AI assistant for the 'Current Trends in Data Science and AI' course at University of Antwerp.\n\n"
             "INSTRUCTIONS:\n"
             "1. Answer questions using ONLY the provided context from course materials\n"
             "2. If information isn't in the context, clearly state 'I don't have that information in the course materials'\n"
             "3. Always cite sources using the format: [Source: filename] or [Source: filename, page X] for PDFs\n"
             "4. Provide specific, actionable answers when possible\n"
             "5. If multiple sources support your answer, mention all relevant ones\n"
             "6. For technical concepts, explain them clearly and concisely\n"
             "7. Maintain an academic but approachable tone\n\n"
             "Context from course materials:\n{context}"
            ),
            ("human", "{input}")
        ])
    
    def setup_chain(self, retriever) -> None:
        logger.info("Setting up RAG chain...")
        
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        self.rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG chain setup complete")
    
    def query(self, question: str) -> Dict[str, Any]:
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not setup. Call setup_chain() first.")
        
        logger.info(f"Processing query: {question}")
        
        try:
            response = self.rag_chain.invoke({"input": question})
            
            sources = self._format_sources(response.get("context", []))
            
            return {
                "answer": response["answer"],
                "sources": sources,
                "source_documents": response.get("context", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "source_documents": []
            }
    
    def _format_sources(self, documents) -> list:
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page_number')
            
            if page_number:
                source_str = f"{source} (page {page_number})"
            else:
                source_str = source
            
            if source_str not in seen_sources:
                sources.append(source_str)
                seen_sources.add(source_str)
        
        return sources


def setup_rag_chain(retriever):
    rag = RAGChain()
    rag.setup_chain(retriever)
    return rag


def ask_rag(question: str, rag_chain) -> str:
    if hasattr(rag_chain, 'query'):
        response = rag_chain.query(question)
        return response["answer"]
    else:
        response = rag_chain.invoke({"input": question})
        return response["answer"]


def ask_normal_gpt(question: str) -> str:
    logger.info(f"Asking normal GPT: {question}")
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE
    )
    
    try:
        response = llm.invoke([HumanMessage(content=question)])
        answer = response.content
        logger.info("Normal GPT response generated")
        return answer
        
    except Exception as e:
        logger.error(f"Error with normal GPT: {e}")
        return f"Error generating response: {str(e)}" 