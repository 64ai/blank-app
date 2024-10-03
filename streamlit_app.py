import streamlit as st
from transformers import pipeline

# Load the Hugging Face model once when the app starts
def load_model():
    return pipeline("text-generation", model="baofamily/gemma-2b-finetuned-model-llama-factory", device=0)


# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a chatbot using the Gemma 2B fine-tuned model to generate responses."
)

# Load the Hugging Face model once
chatbot = load_model()

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
    response = chatbot(prompt, truncation=False, max_length=1500,
                       num_return_sequences=1)[0]["generated_text"]

    # Display the assistant's response and store it in session state.
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
