import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import dotenv

dotenv.load_dotenv()

PREPROCESSED_DATA_DIR = "preprocessed-data"
FAISS_INDEX_PATH = "faiss_index"

def load_documents(data_dir):
    print(f"Loading documents from {data_dir}...")
    documents = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(root, file_name)
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

                relative_path = os.path.relpath(file_path, data_dir)
                for doc in docs:
                    doc.metadata["source"] = relative_path
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    else:
        print("Creating new FAISS index (this may take a while and incur OpenAI embedding costs)...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index created and saved to {FAISS_INDEX_PATH}.")
    return vector_store

def setup_rag_chain(vector_store):
    print("Setting up RAG chain...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful assistant for a course titled Current Trends in Data Science and Artificial Intelligence (CTR) at the University of Antwerp. "
            "Answer the user's questions based *only* on the provided context. "
            "If the answer is not in the context, state that you don't know. "
            "Cite the source of your information by referring to the 'source' metadata provided with each document chunk, "
            "e.g., '[Source: papers/attention-is-all-you-need.txt]'. "
            "Be concise and direct.\n\n"
            "Context: {context}"
        ),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain setup complete.")
    return retrieval_chain

def ask_rag(question, rag_chain):
    print(f"\n--- Asking RAG: '{question}' ---")
    response = rag_chain.invoke({"input": question})
    answer = response["answer"]
    print("RAG Answer:")
    print(answer)
    return answer

def ask_normal_gpt(question):
    print(f"\n--- Asking Normal GPT: '{question}' ---")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([HumanMessage(content=question)])
    answer = response.content
    print("Normal GPT Answer:")
    print(answer)
    return answer

if __name__ == "__main__":
    documents = load_documents(PREPROCESSED_DATA_DIR)
    chunks = split_documents(documents)

    vector_store = get_vector_store(chunks)

    rag_chain = setup_rag_chain(vector_store)

    question1 = "Who was in Martina's group for her Presentation about 'Can LLMS keep a secret'?"
    rag_answer1 = ask_rag(question1, rag_chain)
    normal_gpt_answer1 = ask_normal_gpt(question1)
    

    print("\n--- Comparison ---")
    print(f"Question: {question1}")
    print(f"RAG Answer: {rag_answer1}")
    print(f"Normal GPT Answer: {normal_gpt_answer1}\n")