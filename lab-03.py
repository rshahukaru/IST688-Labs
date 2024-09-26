import streamlit as st
from openai import OpenAI

def main():
    st.title("Lab 03")

# Show title and description.
st.markdown('<h1 style="color: #03B8AC;">LAB-03: Revanth Shahukaru - Document question answering and Chatbot</h1>', unsafe_allow_html=True)
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via ⁠ st.file_uploader ⁠.
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # Sidebar options for summarizing 
    st.sidebar.title("Options")
    
    # Model selection
    openAI_model = st.sidebar.selectbox("Choose the GPT Model", ("mini", "regular"))
    model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

    # Summary options
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ),
    )

    if uploaded_file:
        # Process the uploaded file
        document = uploaded_file.read().decode()

        # Instruction based on user selection on the sidebar menu
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Generate the summary using the OpenAI API
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            stream=True,
        )

        # Stream the summary response to the app
        st.write_stream(stream)

    # Set up the session state to hold chatbot messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.messages:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input for the chatbot
    if prompt := st.chat_input("Ask the chatbot a question or interact:"):
        # Append user input to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the user input in the chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response from OpenAI using the same model
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.messages,
            stream=True,
        )

        # Stream the assistant's response
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        # Append the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

