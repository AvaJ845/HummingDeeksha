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

class CloudLLMClient:
    def __init__(self):
        api_key_valid, message = debug_api_key()
        if not api_key_valid:
            st.error(f"API Key Configuration Error: {message}")
            self.api_key = None
            return
            
        self.api_key = st.secrets["hf_api_key"]
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.api = HfApi(token=self.api_key)

    def check_api_status(self):
        """Check if the API key is valid and service is accessible"""
        try:
            if not self.api_key:
                return False
                
            # Test with a basic model to verify access
            response = requests.get(
                f"{HUGGINGFACE_API_URL}gpt2/",
                headers=self.headers
            )
            
            if response.status_code == 401:
                st.error("""
                API Key is invalid or lacks necessary permissions. Please:
                1. Visit https://huggingface.co/settings/tokens
                2. Create a new token with 'read' and 'inference' permissions
                3. Update your API key in Streamlit settings
                """)
                return False
                
            if response.status_code != 200:
                st.error(f"API request failed with status code: {response.status_code}")
                return False
                
            return True
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Hugging Face API. Please check your internet connection.")
            return False
        except Exception as e:
            st.error(f"Unexpected error checking API status: {str(e)}")
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

    def generate_response(self, model_id, prompt, max_length=1000, temperature=0.7):
        """Generate response from the model"""
        try:
            client = InferenceClient(
                model=model_id,
                token=self.api_key
            )
            response = client.text_generation(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True
            )
            return True, response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return False, """Error: Unauthorized access. Please:
                1. Visit https://huggingface.co/settings/tokens
                2. Create a new token with 'read' and 'inference' permissions
                3. Accept the model's terms of use at its Hugging Face page
                4. Update your API key in Streamlit settings"""
            elif e.response.status_code == 403:
                model_url = f"https://huggingface.co/{model_id}"
                return False, f"""Error: Model requires acceptance of terms of use. Please:
                1. Visit {model_url}
                2. Click the 'Accept terms & conditions' button
                3. Sign in to your Hugging Face account if needed
                4. Return here to try again
                
                Model page: {model_url}"""
            else:
                return False, f"Error: {str(e)}"
        except Exception as e:
            return False, f"Error generating response: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
    if 'model_id' not in st.session_state:
        st.session_state.model_id = "gpt2"  # Start with a more accessible model
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_length' not in st.session_state:
        st.session_state.max_length = 1000
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = list(MODEL_CATEGORIES.keys())[0]

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
            
            success, response = client.generate_response(
                st.session_state.model_id,
                prompt,
                max_length=st.session_state.max_length,
                temperature=st.session_state.temperature
            )
            
            if success:
                words = response.split()
                full_response = ""
                for word in words:
                    full_response += word + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                message_placeholder.markdown(f"🚫 {response}")

def render_model_info(client):
    """Render model information and settings"""
    st.header("Model Information")
    
    # Display current model info first
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
    
    # Add educational section about model creators
    st.subheader("📚 About Available Models")
    
    # Create tabs for each category
    tabs = st.tabs([category_name for category_name in MODEL_CATEGORIES.keys()])
    
    for idx, (category_name, models) in enumerate(MODEL_CATEGORIES.items()):
        with tabs[idx]:
            for model_id, info in models.items():
                st.markdown(f"""
                **{info['name']}**  
                🏢 Creator: {info['creator']}  
                📝 Description: {info['description']}
                """)
                st.divider()

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
            options=list(MODEL_CATEGORIES.keys()),
            key="model_category"
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
