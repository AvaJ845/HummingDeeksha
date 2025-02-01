import streamlit as st
import subprocess
import json
import requests
import time
import platform
import os
from datetime import datetime

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

def get_model_info(model_name):
    """Get detailed information about a specific model"""
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def pull_model(model_name):
    """Pull a model from Ollama"""
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                if "downloading" in output.lower():
                    try:
                        # Extract progress percentage
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

def main():
    st.set_page_config(
        page_title="Ollama Model Manager",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Ollama Model Manager")

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

    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Available Models")
        models = get_available_models()
        
        if not models:
            st.info("No models found. Pull some models to get started!")
        else:
            for model in models:
                with st.expander(f"📦 {model['name']}"):
                    model_info = get_model_info(model['name'])
                    
                    if model_info:
                        st.write(f"**Size:** {format_size(model_info.get('size', 0))}")
                        st.write(f"**Modified:** {datetime.fromtimestamp(model_info.get('modified', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Show model details in a cleaner format
                        if 'details' in model_info:
                            st.write("**Details:**")
                            st.json(model_info['details'])
                    
                    if st.button("🗑️ Remove", key=f"remove_{model['name']}"):
                        if remove_model(model['name']):
                            st.rerun()

    with col2:
        st.header("Pull New Model")
        
        # Popular models list
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
        
        # Custom model input
        custom_model = st.text_input(
            "Enter model name",
            placeholder="e.g., llama2:13b or mistral:latest"
        )
        
        # Model selection
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

if __name__ == "__main__":
    main()