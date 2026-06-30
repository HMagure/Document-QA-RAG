import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_ollama import ChatOllama

from langchain.prompts import ChatPromptTemplate

from langchain.chains import RetrievalQA


#create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


#Create the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


#load the document
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


#split the doc into chunks

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


#create vector database(convert every chunk into vector using Bge-m3 and store it in a database)
def create_vector_store(chunks):
    """
    Create a Chroma vector database from document chunks.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    return vector_store

#create a retreiver 
def get_retriever(vector_store):
    """
    Create a retriever from the vector store.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


# build the rag chain
def create_qa_chain(retriever):
    """
    Create the Retrieval-Augmented QA chain.
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
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )

    return qa_chain


#Build the Main function

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

