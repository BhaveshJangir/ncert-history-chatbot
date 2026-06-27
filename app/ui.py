import streamlit as st
import requests
import os

# Backend API URL (use env var for Docker, default to localhost for local)
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="NCERT History Tutor", page_icon="📚", layout="centered")

st.title("📚 NCERT Class 10 History Tutor")
st.markdown("Ask me anything about your Class 10 History textbook! I will provide exact chapter and page citations for every fact.")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question (e.g. Who wrote Hind Swaraj?)"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Send request to backend (only send the last 10 messages to save context/tokens)
    payload = {
        "query": prompt,
        "history": st.session_state.messages[-10:]
    }
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching the textbook..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Internal Server Error")
