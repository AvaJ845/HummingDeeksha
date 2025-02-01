import streamlit as st
import ollama
import subprocess
import json
import requests
import time
import platform
import os
from datetime import datetime
from PIL import Image

# Initialize Ollama client
client = ollama.Client()

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return True
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_service():
    """Attempt to start Ollama service"""
    try:
        if platform.system().lower() == "windows":
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        time.sleep(2)  # Wait for service to start
        return True
    except Exception as e:
        st.error(f"Failed to start Ollama: {str(e)}")
        return False

def get_available_models():
    """Get list of locally available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            return models.get('models', [])
        return []
    except:
        return []

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there. \nHow can I be of assistance today?"}
        ]
    if 'model' not in st.session_state:
        st.session_state.model = "deepseek-r1:8b"
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"

def get_ai_response(message):
    """Generate AI response using Ollama"""
    try:
        response = client.generate(
            model=st.session_state.model,
            prompt=message
        )
        return True, response.response
    except Exception as e:
        return False, f"Error: Unable to get response from AI model. {str(e)}"

def pull_model(model_name):
    """Pull a model from Ollama"""
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                if "downloading" in output.lower():
                    try:
                        progress = float(output.split("%")[0].split()[-1]) / 100
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {int(progress * 100)}%")
                    except:
                        pass
                
        return_code = process.poll()
        progress_bar.empty()
        status_text.empty()
        
        if return_code == 0:
            st.success(f"Successfully pulled {model_name}")
            return True
        else:
            st.error(f"Failed to pull {model_name}")
            return False
    except Exception as e:
        st.error(f"Error pulling model: {str(e)}")
        return False

def remove_model(model_name):
    """Remove a model from local storage"""
    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success(f"Successfully removed {model_name}")
            return True
        else:
            st.error(f"Failed to remove {model_name}: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error removing model: {str(e)}")
        return False

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def render_chat_interface():
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
            
            success, response = get_ai_response(prompt)
            if success:
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.markdown(f"🚫 {response}")

def render_model_manager():
    """Render the model manager interface"""
    models = get_available_models()
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Available Models")
        if not models:
            st.info("No models found. Pull some models to get started!")
        else:
            for model in models:
                with st.expander(f"📦 {model['name']}"):
                    model_info = get_model_info(model['name'])
                    if model_info:
                        st.write(f"**Size:** {format_size(model_info.get('size', 0))}")
                        st.write(f"**Modified:** {datetime.fromtimestamp(model_info.get('modified', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                        if 'details' in model_info:
                            st.write("**Details:**")
                            st.json(model_info['details'])
                    
                    if st.button("🗑️ Remove", key=f"remove_{model['name']}"):
                        if remove_model(model['name']):
                            st.rerun()

    with col2:
        st.header("Pull New Model")
        popular_models = [
            "llama2",
            "mistral",
            "codellama",
            "phi",
            "neural-chat",
            "starling-lm",
            "deepseek-coder",
            "stable-beluga",
            "falcon"
        ]
        
        custom_model = st.text_input(
            "Enter model name",
            placeholder="e.g., llama2:13b or mistral:latest"
        )
        
        selected_model = st.selectbox(
            "Or choose from popular models",
            [""] + popular_models,
            format_func=lambda x: "Select a model..." if x == "" else x
        )
        
        model_to_pull = custom_model or selected_model
        
        if model_to_pull:
            if st.button(f"📥 Pull {model_to_pull}"):
                if pull_model(model_to_pull):
                    time.sleep(1)
                    st.rerun()

def main():
    st.set_page_config(
        page_title="Ollama Chat & Model Manager",
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

    # Check if Ollama is installed
    try:
        version_output = subprocess.check_output(["ollama", "version"], text=True)
        st.sidebar.success(f"Ollama installed: {version_output.strip()}")
    except:
        st.error("""
        🚫 Ollama is not installed! Please install it first:
        1. Visit: https://ollama.com/download
        2. Follow the installation instructions for your system
        3. Refresh this page
        """)
        return

    # Check if Ollama service is running
    if not check_ollama_running():
        st.warning("⚠️ Ollama service is not running")
        if st.button("Start Ollama Service"):
            if start_ollama_service():
                st.success("✅ Ollama service started successfully!")
                st.rerun()
            else:
                st.error("Failed to start Ollama service")
        return

    # Sidebar navigation and settings
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Chat", "Model Manager"])
        st.session_state.current_page = page.lower()

        st.title("Settings")
        
        # Get available models for selection
        available_models = [model['name'] for model in get_available_models()]
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=available_models.index(st.session_state.model) if st.session_state.model in available_models else 0
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

    # Main content
    if st.session_state.current_page == "chat":
        st.title("💬 Chat Interface")
        render_chat_interface()
    else:
        st.title("🤖 Model Manager")
        render_model_manager()

if __name__ == "__main__":
    main()