import streamlit as st
from huggingface_hub import HfApi, InferenceClient
import json
import requests
import time
from datetime import datetime
import os

# Constants for API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
DEEPSEEK_MODELS = {
    "deepseek-ai/deepseek-coder-6.7b-base": "Code generation model (6.7B)",
    "deepseek-ai/deepseek-coder-33b-base": "Code generation model (33B)",
    "deepseek-ai/deepseek-math-7b-base": "Math problem solving (7B)",
    "deepseek-ai/deepseek-math-7b-instruct": "Math instruction (7B)",
}

class CloudLLMClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.api = HfApi(token=api_key)
        
    def check_api_status(self):
        """Check if the API key is valid and service is accessible"""
        try:
            test_response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=self.headers
            )
            return test_response.status_code == 200
        except Exception:
            return False

    def get_model_info(self, model_id):
        """Get information about a specific model"""
        try:
            model_info = self.api.model_info(model_id)
            return {
                "name": model_info.modelId,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "last_modified": model_info.lastModified,
                "tags": model_info.tags
            }
        except Exception as e:
            st.error(f"Error fetching model info: {str(e)}")
            return None

    async def generate_response(self, model_id, prompt, max_length=1000, temperature=0.7):
        """Generate response from the model"""
        try:
            client = InferenceClient(model=model_id, token=self.api_key)
            response = await client.text_generation(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                stream=True
            )
            return True, response
        except Exception as e:
            return False, f"Error generating response: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm DeepSeek, how can I help you today?"}
        ]
    if 'model_id' not in st.session_state:
        st.session_state.model_id = list(DEEPSEEK_MODELS.keys())[0]
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

def render_chat_interface(client):
    """Render the chat interface"""
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get streaming response
            success, response_stream = await client.generate_response(
                st.session_state.model_id,
                prompt
            )
            
            if success:
                async for token in response_stream:
                    full_response += token + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                message_placeholder.markdown(f"🚫 {response_stream}")

def render_model_info(client):
    """Render model information and settings"""
    st.header("Model Information")
    
    model_info = client.get_model_info(st.session_state.model_id)
    if model_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Downloads", model_info["downloads"])
            st.metric("Likes", model_info["likes"])
        with col2:
            st.write("Last Modified:", model_info["last_modified"])
            st.write("Tags:", ", ".join(model_info["tags"]))

def main():
    st.set_page_config(
        page_title="Cloud DeepSeek Chat",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS
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

    # Sidebar for settings and API key
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Enter Hugging Face API Key",
            type="password",
            value=st.session_state.api_key
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.rerun()

        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(DEEPSEEK_MODELS.keys()),
            format_func=lambda x: f"{x.split('/')[-1]} - {DEEPSEEK_MODELS[x]}",
            index=list(DEEPSEEK_MODELS.keys()).index(st.session_state.model_id)
        )
        
        if selected_model != st.session_state.model_id:
            st.session_state.model_id = selected_model
            st.rerun()

        # Generation settings
        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused."
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum number of tokens to generate."
        )

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm DeepSeek, how can I help you today?"}
            ]
            st.rerun()

    # Main content
    if not st.session_state.api_key:
        st.error("""
        🔑 Please enter your Hugging Face API key in the sidebar to continue.
        
        To get an API key:
        1. Go to https://huggingface.co/
        2. Create an account or sign in
        3. Go to Settings > Access Tokens
        4. Create a new token with read access
        """)
        return

    # Initialize client
    client = CloudLLMClient(st.session_state.api_key)
    
    # Check API status
    if not client.check_api_status():
        st.error("❌ Invalid API key or service unavailable. Please check your API key and try again.")
        return

    # Main interface
    st.title("💬 DeepSeek Chat")
    
    # Show model info
    with st.expander("📊 Model Information", expanded=False):
        render_model_info(client)
    
    # Render chat interface
    render_chat_interface(client)

if __name__ == "__main__":
    main()
