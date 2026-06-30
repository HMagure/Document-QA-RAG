import os
import streamlit as st

from rag import process_document


# Page Configuration


st.set_page_config(
    page_title="Document Q&A RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Question Answering")
st.write("Upload a PDF or TXT file and ask questions about it.")


# Session State


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar


st.sidebar.header("Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)


# Process Uploaded File


if uploaded_file is not None:

    upload_path = os.path.join("uploads", uploaded_file.name)

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.qa_chain is None:

        with st.spinner("Processing document..."):

            st.session_state.qa_chain = process_document(upload_path)

        st.sidebar.success("Document processed successfully!")


# Display Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Input


question = st.chat_input("Ask a question about your document")

if question:

    if st.session_state.qa_chain is None:

        st.warning("Please upload a document first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):

            result = st.session_state.qa_chain.invoke(
                {
                    "query": question
                }
            )

            answer = result["result"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)