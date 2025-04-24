import os
# Desactivar file watcher de Streamlit para evitar errores con Torch
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import sys
import logging
import streamlit as st

# Agrega la ruta del directorio donde están las funciones de audio y chatbot
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from audio.utils_audio import load_whisper_model, preprocess_audio_file, transcribe_audio_file, load_tts_model, obtain_audio_response
from utils import load_llama_model, load_gpt2_model, answer_query

# Configuración de logging
enable_dir = "logs"
os.makedirs(enable_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(enable_dir, "app_audio.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Configuración de la página
st.set_page_config(page_title="PharmAI Chatbot Audio", page_icon="💊")
st.title("💊 PharmAI: Asistente de Medicamentos")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de audio
try:
    audio_value = st.audio_input("🔴 Pulsa para grabar tu pregunta")
except AttributeError:
    audio_value = st.experimental_audio_input("🔴 Pulsa para grabar tu pregunta")

query = None

# Procesar audio si existe
if audio_value:
    try:
        # Guardar bytes en archivo WAV
        os.makedirs("./audio", exist_ok=True)
        output_path = os.path.join("./audio", "recorded.wav")
        data = audio_value if isinstance(audio_value, (bytes, bytearray)) else audio_value.read()
        with open(output_path, "wb") as f:
            f.write(data)
        logging.info(f"Audio guardado en {output_path}")

        # Transcripción con Whisper
        model_wh, device = load_whisper_model("medium")
        preprocessed = preprocess_audio_file(output_path)
        transcript = transcribe_audio_file(model_wh, device, preprocessed)
        query = transcript.strip()
        st.audio(audio_value)
        st.markdown(f"**Transcripción:** {query}")
        logging.info(f"Transcripción: {query}")
    except Exception as e:
        logging.exception("Error al procesar audio_input")
        st.error(f"Se produjo un error al procesar el audio: {e}")

# Entrada de texto
text_query = st.chat_input("O escribe tu pregunta sobre un medicamento...")
if text_query:
    query = text_query.strip()

# Si hay query de audio o texto, procesar
if query:
    logging.info(f"User query: {query}")
    # Mostrar usuario\    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Cargar modelo y generar respuesta
    model_name = "llama2"  # Cambia a "gpt2" para usar Llama 2 chat 7b
    respuesta = answer_query(query, model_name)

    # Mostrar respuesta
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)
        logging.info(f"Respuesta del chatbot: {respuesta}")

        # Generar respuesta de audio
        if respuesta is not None:
            tts = load_tts_model()
            audio_bytes = obtain_audio_response(respuesta, model=tts)
            st.audio(audio_bytes)

        # Guardar respuesta de audio
        audio_path = os.path.join("./audio", "response.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)