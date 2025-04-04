import streamlit as st
from utils_chatbot_streamlit.procesar_json_para_chatbot import (
    search_faiss,
    answer_query,
)

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
prompt = st.chat_input("Escribe tu pregunta sobre un medicamento...")

if prompt:
    # Mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Buscar los fragmentos más relevantes con FAISS
    fragments = search_faiss(prompt, k=5)

    # Generar la respuesta con tu función real
    respuesta = answer_query(prompt, fragments)

    # Mostrar y guardar la respuesta del asistente
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)
