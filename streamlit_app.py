import streamlit as st
from transformers import pipeline

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses a custom model from Hugging Face to generate responses. "
    "To use this app, you need to provide your Hugging Face model name or path."
)

# Ask user for their Hugging Face model name or path.
model_name_or_path = st.text_input("Hugging Face Model Name or Path")
if not model_name_or_path:
    st.info("Please add your Hugging Face model name or path to continue.", icon="🗝️")
else:
    # Load the Hugging Face model using the pipeline
    try:
        chatbot = pipeline("text-generation", model=model_name_or_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

    # Create a session state variable to store the chat messages.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the Hugging Face model.
        response = chatbot(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]

        # Display the assistant's response and store it in session state.
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
