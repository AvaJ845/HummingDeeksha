import streamlit as st
import ollama
from PIL import Image
import time

# Initialize Ollama client
client = ollama.Client()

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there. \nHow can I be of assistance today?"}
        ]
    if 'model' not in st.session_state:
        st.session_state.model = "deepseek-r1:8b"

def get_ai_response(message):
    """Generate AI response using Ollama"""
    try:
        response = client.generate(
            model=st.session_state.model,
            prompt=message
        )
        return response.response
    except Exception as e:
        return f"Error: Unable to get response from AI model. {str(e)}"

def main():
    # Set page config
    st.set_page_config(
        page_title="DeepSeek Chat",
        page_icon="🤖",
        layout="centered"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS for chat styling
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            background-color: #1a1a1a;
            color: white;
        }
        .stMarkdown {
            color: white;
        }
        .main {
            background-color: #101010;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title
    st.title("DeepSeek Chat")

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input form
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with milliseconds delay
            response = get_ai_response(prompt)
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)  # Add a slight delay between words
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sidebar for settings
    with st.sidebar:
        st.title("Settings")
        
        # Model selection
        model_options = ["deepseek-r1:8b", "deepseek-r1:32b"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=model_options.index(st.session_state.model)
        )
        
        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
            st.rerun()

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello there. \nHow can I be of assistance today?"}
            ]
            st.rerun()

if __name__ == "__main__":
    main()