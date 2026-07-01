# Document QA - RAG Project

A production-ready Retrieval-Augmented Generation (RAG) application built with **Streamlit** and **LangChain**. This system allows users to upload documents (PDFs/TXTs), processes and chunks the text, embeds them into a local vector database, and enables conversational question-answering using a local Large Language Model (LLM).

---

## 🚀 Features
* **Fully Local & Secure:** Uses local embedding models and local LLMs via Ollama—no data leaves your machine.
* **Smart Document Chunking:** Automatically parses and splits text using advanced LangChain text splitters.
* **Persistent Vector Storage:** Uses ChromaDB locally to cache document embeddings so you don't have to re-index documents on every run.
* **Interactive Chat UI:** Clean and intuitive chat interface built with Streamlit.

---

## 🛠️ Architecture & Tech Stack
* **Frontend:** Streamlit
* **RAG Framework:** LangChain
* **Vector Database:** ChromaDB
* **Local LLM Engine:** Ollama (`llama3.2`)
* **Embedding Model:** HuggingFace Transformers (`BAAI/bge-m3`)

---

## 📦 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Ollama** installed on your system. 
* Download Ollama from [ollama.com](https://ollama.com)
* Pull the required model in your terminal:
  ```bash
  ollama pull llama3.2