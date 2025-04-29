import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import requests
from requests.exceptions import RequestException
import json
from json.decoder import JSONDecodeError

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from langchain_core.messages import HumanMessage, AIMessage
# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 AI Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["qwen2.5-coder:7b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")


# initiate the chat engine

@st.cache_resource
def init_chat_engine():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        # Test connection and model availability
        response = requests.post(
            f"{base_url}/api/tags",
            timeout=5
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Verify JSON response
        try:
            models = response.json()
        except JSONDecodeError:
            st.error("⚠️ Invalid response from Ollama service")
            return None
            
        # Initialize chat engine
        return ChatOllama(
            model=selected_model,
            base_url=base_url,
            temperature=0.3,
            timeout=30  # Add timeout for API calls
        )
    except RequestException as e:
        st.error(f"⚠️ Connection error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error initializing chat engine: {str(e)}")
        return None

# Initialize the chat engine
llm_engine = init_chat_engine()
# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Qwen. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    try:
        if llm_engine is None:
            return "⚠️ Chat engine is not initialized. Please check if Ollama service is running."
        
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        response = processing_pipeline.invoke({})
        
        # Validate response
        if not response or not isinstance(response, str):
            return "⚠️ Invalid response from the model"
            
        return response
        
    except JSONDecodeError:
        return "⚠️ Error: Received invalid response from Ollama"
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessage(content=msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
