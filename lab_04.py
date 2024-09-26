import sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to ensure the OpenAI client is initialized
def initialize_openai_client():
    if 'openai_instance' not in st.session_state:
        # Get the API key from Streamlit secrets
        api_key = st.secrets["openai_api_key"]
        # Initialize the OpenAI client and store it in session state
        st.session_state.openai_instance = OpenAI(api_key=api_key)

# Function to create the ChromaDB collection
def create_document_collection():
    if 'document_vectorDB' not in st.session_state:
        # Set up the ChromaDB client
        persist_path = os.path.join(os.getcwd(), "chroma_db")
        chroma_instance = chromadb.PersistentClient(path=persist_path)
        document_collection = chroma_instance.get_or_create_collection("DocumentCollection")

        initialize_openai_client()

        # Define the directory containing the PDF files
        pdf_directory = os.path.join(os.getcwd(), "Lab-04-DataFiles")
        if not os.path.exists(pdf_directory):
            st.error(f"Directory not found: {pdf_directory}")
            return None

        # Process each PDF file in the directory
        for pdf_file in os.listdir(pdf_directory):
            if pdf_file.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, pdf_file)
                try:
                    # Extract text from the PDF
                    with open(file_path, "rb") as pdf:
                        pdf_reader = PdfReader(pdf)
                        extracted_text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])

                    # Generate embeddings for the extracted text
                    response = st.session_state.openai_instance.embeddings.create(
                        input=extracted_text, model="text-embedding-3-small"
                    )
                    embedding_result = response.data[0].embedding

                    # Add the document to ChromaDB
                    document_collection.add(
                        documents=[extracted_text],
                        metadatas=[{"filename": pdf_file}],
                        ids=[pdf_file],
                        embeddings=[embedding_result]
                    )
                except Exception as e:
                    st.error(f"Error processing {pdf_file}: {str(e)}")

        # Store the collection in session state
        st.session_state.document_vectorDB = document_collection

    return st.session_state.document_vectorDB

# Function to query the vector database
def search_vector_db(collection, query_text):
    initialize_openai_client()
    try:
        # Generate embedding for the query
        response = st.session_state.openai_instance.embeddings.create(
            input=query_text, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Query the ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return results['documents'][0], [metadata['filename'] for metadata in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return [], []

# Function to get chatbot response using OpenAI's GPT model
def generate_chatbot_response(query_text, context_data):
    initialize_openai_client()
    # Construct the prompt for the GPT model
    gpt_prompt = f"""You are an AI assistant with knowledge from specific documents. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information.

Context:
{context_data}

User Question: {query_text}

Answer:"""

    try:
        # Generate streaming response using OpenAI's chat completion
        response_stream = st.session_state.openai_instance.chat.completions.create(
            model="gpt-4o",  # Using the latest GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": gpt_prompt}
            ],
            stream=True  # Enable streaming
        )
        return response_stream
    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return None

# Initialize session state for chat history, system readiness, and collection
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'app_ready' not in st.session_state:
    st.session_state.app_ready = False
if 'document_collection' not in st.session_state:
    st.session_state.document_collection = None

# Page content
st.title("Lab 4 - Document Chatbot")

# Check if the system is ready, if not, prepare it
if not st.session_state.app_ready:
    # Show a spinner while processing documents
    with st.spinner("Processing documents and preparing the system..."):
        st.session_state.document_collection = create_document_collection()
        if st.session_state.document_collection:
            # Set the system as ready and show a success message
            st.session_state.app_ready = True
            st.success("AI ChatBot is Ready!!!")
        else:
            st.error("Failed to create or load the document collection. Please check the file path and try again.")

# Only show the chat interface if the system is ready
if st.session_state.app_ready and st.session_state.document_collection:
    st.subheader("Chat with the AI Assistant")

    # Display chat history
    for msg in st.session_state.message_history:
        if isinstance(msg, dict):
            # New format (dictionary with 'role' and 'content' keys)
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        elif isinstance(msg, tuple):
            # Old format (tuple with role and content)
            role, content = msg
            # Convert 'You' to 'user', and assume any other role is 'assistant'
            with st.chat_message("user" if role == "You" else "assistant"):
                st.markdown(content)

    # User input
    user_query = st.chat_input("Ask a question about the documents:")

    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query the vector database
        relevant_texts, relevant_documents = search_vector_db(st.session_state.document_collection, user_query)
        context_data = "\n".join(relevant_texts)

        # Get streaming chatbot response
        response_stream = generate_chatbot_response(user_query, context_data)

        # Display AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            complete_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    complete_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(complete_response + "▌")
            response_placeholder.markdown(complete_response)

        # Add to chat history (new format)
        st.session_state.message_history.append({"role": "user", "content": user_query})
        st.session_state.message_history.append({"role": "assistant", "content": complete_response})

        # Display relevant documents
        with st.expander("Referenced documents used"):
            for doc in relevant_documents:
                st.write(f"- {doc}")

elif not st.session_state.app_ready:
    st.info("The system is still preparing. Please wait...")
else:
    st.error("Failed to create or load the document collection. Please check your setup.")