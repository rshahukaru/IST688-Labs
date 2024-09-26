import sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

import streamlit as st

def main():
    st.title("Lab 04 - ChromaDB")

    # Fix for sqlite3 on Streamlit Cloud
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    import chromadb

    # Initialize OpenAI client if not already present
    def setup_openai_client():
        if 'openai_client' not in st.session_state:
            api_key = st.secrets["openai_api_key"]
            st.session_state.openai_client = OpenAI(api_key=api_key)

    # Create or retrieve a ChromaDB collection
    def initialize_document_collection():
        if 'doc_collection' not in st.session_state:
            # Set up persistent ChromaDB client
            storage_location = os.path.join(os.getcwd(), "chroma_db")
            chroma_client = chromadb.PersistentClient(path=storage_location)
            document_collection = chroma_client.get_or_create_collection("DocumentCollection")

            setup_openai_client()

            # Directory containing PDF files
            pdf_directory = os.path.join(os.getcwd(), "Lab-04-DataFiles")
            if not os.path.exists(pdf_directory):
                st.error(f"Directory not found: {pdf_directory}")
                return None

            # Iterate through PDF files and process each one
            for pdf_file in os.listdir(pdf_directory):
                if pdf_file.endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, pdf_file)
                    try:
                        # Extract text from the PDF
                        with open(file_path, "rb") as pdf:
                            pdf_reader = PdfReader(pdf)
                            extracted_text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])

                        # Generate text embeddings using OpenAI
                        embedding_result = st.session_state.openai_client.embeddings.create(
                            input=extracted_text, model="text-embedding-ada-002"
                        )
                        vector_representation = embedding_result.data[0].embedding

                        # Add extracted data to ChromaDB collection
                        document_collection.add(
                            documents=[extracted_text],
                            metadatas=[{"filename": pdf_file}],
                            ids=[pdf_file],
                            embeddings=[vector_representation]
                        )
                    except Exception as err:
                        st.error(f"Failed to process {pdf_file}: {str(err)}")

            # Save collection in session state
            st.session_state.doc_collection = document_collection

        return st.session_state.doc_collection

    # Query the ChromaDB collection using vector similarity
    def perform_vector_search(collection, user_query):
        setup_openai_client()
        try:
            # Generate embedding for the query
            embedding_result = st.session_state.openai_client.embeddings.create(
                input=user_query, model="text-embedding-ada-002"
            )
            query_vector = embedding_result.data[0].embedding

            # Retrieve matching documents from the collection
            search_results = collection.query(
                query_embeddings=[query_vector],
                n_results=3
            )
            return search_results['documents'][0], [meta['filename'] for meta in search_results['metadatas'][0]]
        except Exception as err:
            st.error(f"Failed to search the database: {str(err)}")
            return [], []

    # Function to generate AI chatbot response based on user input and document context
    def provide_ai_response(user_query, document_context):
        setup_openai_client()
        # Construct a prompt for GPT
        gpt_prompt = f"""You are a document-based AI assistant. Use the following context to answer the user's query. If the answer is not in the context, respond with 'I don't have that information.'

    Context:
    {document_context}

    User Query: {user_query}

    Answer:"""

        try:
            # Stream the response using OpenAI's GPT model
            response_stream = st.session_state.openai.chat.completions.create(
                model="gpt-4",  # Using GPT-4 model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": gpt_prompt}
                ],
                stream=True  # Stream the response
            )
            return response_stream
        except Exception as err:
            st.error(f"Error generating response: {str(err)}")
            return None

    # Initialize session variables for the chat app
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = False
    if 'doc_collection' not in st.session_state:
        st.session_state.doc_collection = None

    # Display the chatbot app interface
    st.title("Lab 4 - Document AI Chatbot")

    # Prepare the system if it's not ready
    if not st.session_state.app_ready:
        with st.spinner("Setting up documents and initializing the system..."):
            st.session_state.doc_collection = initialize_document_collection()
            if st.session_state.doc_collection:
                st.session_state.app_ready = True
                st.success("AI Chatbot is now ready!")
            else:
                st.error("Error creating or loading the document collection. Verify file paths.")

    # Show chat interface once the system is ready
    if st.session_state.app_ready and st.session_state.doc_collection:
        st.subheader("Ask the AI Assistant about the documents")

        # Display chat history
        for entry in st.session_state.conversation_history:
            if isinstance(entry, dict):
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])

        # Accept user input
        user_input = st.chat_input("Enter your question:")
        
        if user_input:
            # Display user's message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Search vector database for relevant context
            retrieved_texts, document_sources = perform_vector_search(st.session_state.doc_collection, user_input)
            document_context = "\n".join(retrieved_texts)

            # Get streaming response from the AI
            ai_reply_stream = provide_ai_response(user_input, document_context)

            # Display AI's response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                complete_response = ""
                for chunk in ai_reply_stream:
                    if chunk.choices[0].delta.content is not None:
                        complete_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(complete_response + "▌")
                response_placeholder.markdown(complete_response)

            # Update chat history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": complete_response})

            # Show relevant documents
            with st.expander("Referenced Documents"):
                for doc in document_sources:
                    st.write(f"- {doc}")

    elif not st.session_state.app_ready:
        st.info("The system is initializing. Please wait...")
    else:
        st.error("Error loading the document collection. Please check your setup.")
