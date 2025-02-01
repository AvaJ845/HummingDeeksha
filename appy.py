#appy.py
#dev
# appy.py
#dev
import streamlit as st
from huggingface_hub import HfApi, InferenceClient
import json
import requests
import time
from datetime import datetime
import os

# Constants for API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"

# Organized model dictionary with creator information
MODEL_CATEGORIES = {
    "OpenAI & EleutherAI Models": {
        "gpt2": {
            "name": "GPT-2 Base Model",
            "creator": "OpenAI",
            "description": "Original GPT-2 model, specialized in text generation"
        },
        "distilgpt2": {
            "name": "Distilled GPT-2",
            "creator": "Hugging Face",
            "description": "Smaller, faster version of GPT-2 with similar capabilities"
        },
        "EleutherAI/gpt-neo-125M": {
            "name": "GPT-Neo 125M",
            "creator": "EleutherAI",
            "description": "Open-source alternative to GPT-3"
        }
    },
    "Meta & BigScience Models": {
        "facebook/opt-125m": {
            "name": "OPT 125M Model",
            "creator": "Meta AI",
            "description": "Open Pretrained Transformer, Meta's open-source alternative to GPT-3"
        },
        "bigscience/bloom-560m": {
            "name": "BLOOM 560M Model",
            "creator": "BigScience",
            "description": "Multilingual large language model"
        }
    },
    "DeepSeek Specialized Models": {
        "deepseek-ai/deepseek-coder-6.7b-base": {
            "name": "DeepSeek Coder 6.7B",
            "creator": "DeepSeek AI",
            "description": "Specialized code generation model (6.7B parameters)"
        },
        "deepseek-ai/deepseek-coder-33b-base": {
            "name": "DeepSeek Coder 33B",
            "creator": "DeepSeek AI",
            "description": "Advanced code generation model (33B parameters)"
        },
        "deepseek-ai/deepseek-math-7b-base": {
            "name": "DeepSeek Math 7B",
            "creator": "DeepSeek AI",
            "description": "Specialized mathematics problem-solving model"
        }
    }
}

# Flatten model dictionary for compatibility with existing code
AVAILABLE_MODELS = {
    model_id: info["name"]
    for category in MODEL_CATEGORIES.values()
    for model_id, info in category.items()
}

def debug_api_key():
    """Debug API key configuration"""
    try:
        if 'hf_api_key' not in st.secrets:
            return False, "API key not found in secrets"
        
        if not st.secrets['hf_api_key']:
            return False, "API key is empty"
        
        if not st.secrets['hf_api_key'].startswith('hf_'):
            return False, "API key format is invalid"
        
        return True, "API key format is valid"
    except Exception as e:
        return False, f"Error checking API key: {str(e)}"

[Previous CloudLLMClient class and other functions remain the same...]

def render_model_info(client):
    """Render model information and settings"""
    st.header("Model Information")
    
    # Add educational section about model creators
    st.subheader("📚 About the Models")
    
    for category_name, models in MODEL_CATEGORIES.items():
        with st.expander(f"{category_name}"):
            for model_id, info in models.items():
                st.markdown(f"""
                **{info['name']}**  
                🏢 Creator: {info['creator']}  
                📝 Description: {info['description']}
                """)
                st.divider()
    
    # Display specific model info
    st.subheader("Current Model Statistics")
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
        page_title="Cloud LLM Chat",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

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

    with st.sidebar:
        st.title("🔧 Debug Information")
        with st.expander("Show Debug Info"):
            key_valid, key_message = debug_api_key()
            st.write(f"API Key Status: {key_message}")
            
            if 'hf_api_key' in st.secrets:
                masked_key = st.secrets['hf_api_key'][:5] + "..." + st.secrets['hf_api_key'][-4:]
                st.write(f"API Key Format: {masked_key}")
            else:
                st.write("API Key: Not found in secrets")

        st.title("⚙️ Settings")
        
        # Update model selection to show categories
        st.subheader("🤖 Model Selection")
        category = st.selectbox(
            "Select Category",
            options=list(MODEL_CATEGORIES.keys())
        )
        
        # Filter models by selected category
        category_models = {
            model_id: info["name"]
            for model_id, info in MODEL_CATEGORIES[category].items()
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(category_models.keys()),
            format_func=lambda x: f"{x.split('/')[-1]} - {category_models[x]}"
        )
        
        if selected_model != st.session_state.model_id:
            st.session_state.model_id = selected_model
            st.rerun()

        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make the output more random, lower values make it more focused."
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=100,
            max_value=2000,
            value=st.session_state.max_length,
            step=100,
            help="Maximum number of tokens to generate."
        )

        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
            st.rerun()

        if max_length != st.session_state.max_length:
            st.session_state.max_length = max_length
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]
            st.rerun()

    client = CloudLLMClient()
    
    if not client.check_api_status():
        st.warning("""
        Please ensure you have:
        1. Added the correct API key to Streamlit secrets
        2. The API key starts with 'hf_'
        3. Your API key has 'read' and 'inference' permissions
        4. You have an active internet connection
        
        To fix this:
        1. Visit https://huggingface.co/settings/tokens
        2. Create a new token with required permissions
        3. Click ⋮ (three dots) in the top right
        4. Select 'Settings' → 'Secrets'
        5. Add: hf_api_key = "your_new_api_key"
        """)
        return

    st.title("💬 Language Model Chat")
    
    with st.expander("📊 Model Information", expanded=False):
        render_model_info(client)
    
    render_chat_interface(client)

if __name__ == "__main__":
    main()
