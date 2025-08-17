import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
import json
import os
from huggingface_hub import InferenceClient  # Direct API client

# --- Load Menu Data ---
@st.cache_data
def load_menu():
    with open("data/menu_data.json", "r", encoding='utf-8') as f:
        return json.load(f)
menu = load_menu()

# --- RAG Setup ---
@st.cache_resource
def load_knowledge_base():
    loader = TextLoader("data/faqs.txt", encoding='utf-8')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# Initialize components
vector_db = load_knowledge_base()
client = InferenceClient(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

def get_menu_response(item_name):
    """Directly fetch menu item details"""
    for category in ["makanan", "minuman"]:
        for item in menu[category]:
            if item_name.lower() in item['name'].lower():
                return (
                    f"**{item['name']}**\n"
                    f"Price: Rp {item['price']}\n"
                    f"Description: {item['description']}"
                )
    return None

def query_llm(prompt):
    """Robust LLM query handling"""
    try:
        response = client.post(
            json={"inputs": prompt},
            model="google/flan-t5-small",
            parameters={"max_length": 200}
        )
        return response[0]['generated_text'] if response else "I didn't understand. Please try again."
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# --- Streamlit UI ---
st.title("🍽️ Kedai Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about our menu or services:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # First check for direct menu item match
    menu_response = None
    for item in menu["makanan"] + menu["minuman"]:
        if item['name'].lower() in prompt.lower():
            menu_response = (
                f"**{item['name']}**\n"
                f"Price: Rp {item['price']}\n"
                f"Description: {item['description']}"
            )
            break
    
    if menu_response:
        response = menu_response
    elif "menu" in prompt.lower():
        response = "**Menu Makanan:**\n" + "\n".join(
            f"- {item['name']} (Rp {item['price']})" 
            for item in menu["makanan"]
        ) + "\n\n**Menu Minuman:**\n" + "\n".join(
            f"- {item['name']} (Rp {item['price']})" 
            for item in menu["minuman"]
        )
    else:
        response = query_llm(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
