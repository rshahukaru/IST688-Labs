import sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize the OpenAI client if not done already
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        # Retrieve the OpenAI API key from Streamlit secrets
        api_key = st.secrets["openai_api_key"]
        # Store the OpenAI client in session state
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Create or get the ChromaDB collection
def create_lab4_collection():
    if 'Lab4_vectorDB' not in st.session_state:
        # Set up ChromaDB client
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("Lab4Collection")

        # Get the directory with the PDF files
        pdf_dir = os.path.join(os.getcwd(), "Lab-04-DataFiles")
        if not os.path.exists(pdf_dir):
            st.error(f"PDF directory not found: {pdf_dir}")
            return None

        # List all the PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

        # Get existing IDs from the collection to avoid duplicates
        existing_ids = collection.get(ids=pdf_files)['ids']
        missing_files = set(pdf_files) - set(existing_ids)

        if missing_files:
            ensure_openai_client()

            # Process files that are not in the collection
            for filename in missing_files:
                filepath = os.path.join(pdf_dir, filename)
                try:
                    # Extract text from the PDF
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])

                    # Generate text embeddings using OpenAI
                    response = st.session_state.openai_client.embeddings.create(
                        input=text, model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding

                    # Add the document and its embedding to ChromaDB
                    collection.add(
                        documents=[text],
                        metadatas=[{"filename": filename}],
                        ids=[filename],
                        embeddings=[embedding]
                    )
                    st.info(f"File added to collection: {filename}")
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
        else:
            st.info("All PDFs are already part of the collection.")

        # Store the collection in session state
        st.session_state.Lab4_vectorDB = collection

    return st.session_state.Lab4_vectorDB

# Query the vector database using a search query
def query_vector_db(collection, query_text):
    ensure_openai_client()
    try:
        # Generate an embedding for the search query
        response = st.session_state.openai_client.embeddings.create(
            input=query_text, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Query the ChromaDB collection with the generated embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return results['documents'][0], [meta['filename'] for meta in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the collection: {str(e)}")
        return [], []

# Function to get the chatbot's response using OpenAI's GPT model
def get_chatbot_response(query, context):
    ensure_openai_client()
    # Formulate the prompt for the GPT model
    prompt = f"""You are an AI assistant with knowledge from specific documents. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information. Also, ignore case.

Context:
{context}

User Question: {query}

Answer:"""

    try:
        # Fetch the chatbot response from OpenAI's GPT model
        response_stream = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True  # Enable streaming
        )
        return response_stream
    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return None

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None

# Page content and title
st.title("Lab 4 - Document Chatbot")

# Check if the system is ready; if not, initialize it
if not st.session_state.system_ready:
    with st.spinner("Preparing system and processing documents..."):
        st.session_state.collection = create_lab4_collection()
        if st.session_state.collection:
            st.session_state.system_ready = True
            st.success("AI Chatbot is ready!")
        else:
            st.error("Error creating or loading the document collection. Please check the file paths.")

# Show the chat interface only if the system is ready
if st.session_state.system_ready and st.session_state.collection:
    st.subheader("Chat with the AI Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, dict):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif isinstance(message, tuple):
            role, content = message
            with st.chat_message("user" if role == "You" else "assistant"):
                st.markdown(content)

    # Capture user input
    user_input = st.chat_input("Ask a question about the documents:")

    if user_input:
        # Show user query
        with st.chat_message("user"):
            st.markdown(user_input)

        # Query the vector database
        relevant_texts, relevant_docs = query_vector_db(st.session_state.collection, user_input)
        context = "\n".join(relevant_texts)

        # Get the chatbot response
        response_stream = get_chatbot_response(user_input, context)

        # Display the chatbot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        # Add user query and chatbot response to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        # Show relevant documents used
        with st.expander("Relevant documents used"):
            for doc in relevant_docs:
                st.write(f"- {doc}")

# Show message if the system is still being prepared
elif not st.session_state.system_ready:
    st.info("System is still being prepared. Please wait...")
else:
    st.error("Failed to load the document collection. Please check your file paths.")
