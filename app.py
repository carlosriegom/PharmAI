import streamlit as st
import os
import sys
import json
import numpy as np
import torch
import sys
import os
from sentence_transformers import SentenceTransformer
import seaborn as sns
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

# Funciones chatbot
# Agrega la ruta del directorio donde están las funciones del chatbot
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from utils import *

# Configuración de la página
st.set_page_config(page_title="PharmAI Chatbot", page_icon="💊")
st.title("💬 PharmAI: Asistente de Medicamentos")

# Inicializar historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
query = st.chat_input("Escribe tu pregunta sobre un medicamento...")

if query:
    # Mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 1. Cargar modelo 
    model, tokenizer = load_llama_model()

    # 2. Generar respuesta
    respuesta = answer_query(query, model, tokenizer)

    # Mostrar y guardar la respuesta del asistente
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)
