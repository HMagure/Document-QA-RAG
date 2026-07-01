import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# FIXED: Replaced langchain.chains with langchain_classic.chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Load the document
def load_document(file_path):
    """
    Load a PDF or TXT document.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format.")

    documents = loader.load()
    return documents

# Split the doc into chunks
def split_documents(documents):
    """
    Split documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

# Create vector database
def create_vector_store(chunks):
    """
    Create or load a Chroma vector database from document chunks.
    """
    persist_dir = "chroma_db"
    
    # If the database directory already exists, load it instead of reconstructing it from scratch
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
        # Add the new chunks to the existing database safely
        vector_store.add_documents(chunks)
    else:
        # If it doesn't exist, create it completely fresh
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        
    return vector_store
# Create a retriever 
def get_retriever(vector_store):
    """
    Create a retriever from the vector store.
    """
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
    return retriever

# Build the modern RAG chain
def create_qa_chain(retriever):
    """
    Create the modern Retrieval-Augmented QA chain using LCEL.
    """
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer the question ONLY using the context provided.

If the answer is not present in the context, say:
"I couldn't find the answer in the uploaded document."

Context:
{context}

Question:
{input}

Answer:
"""
    )

    # 1. This handles passing the context documents into your prompt template
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 2. This links the retriever and the question answering chain together
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# Build the Main function
def process_document(file_path):
    """
    Complete pipeline:
    Load -> Split -> Embed -> Store -> Retrieve -> QA Chain
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)
    qa_chain = create_qa_chain(retriever)

    return qa_chain