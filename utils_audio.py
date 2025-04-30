import whisper
import torch
import os
import soundfile as sf
import librosa
import tempfile
import io
from TTS.api import TTS

# pip install streamlit-webrtc openai-whisper librosa soundfile tt

# Variable global para el modelo TTS
_tts_model = None

def load_tts_model(model_name: str = "tts_models/es/css10/vits"):
    """
    Carga el modelo TTS de Coqui por defecto VITS en español (CSS10).
    Devuelve la instancia de TTS.
    """
    global _tts_model
    if _tts_model is None:
        # Carga el modelo una sola vez (descarga si es necesario)
        _tts_model = TTS(model_name=model_name, progress_bar=False, gpu=torch.cuda.is_available())
    return _tts_model


def obtain_audio_response(text: str, model=None, sample_rate: int = 22050) -> bytes:
    """
    Genera una respuesta en audio a partir de un texto usando el modelo TTS cargado.
    - text: texto a convertir.
    - model: instancia TTS (si no se proporciona, se carga por defecto).
    - sample_rate: frecuencia de muestreo del WAV.

    Retorna bytes de un WAV en formato PCM.
    """
    # Cargar modelo si hace falta
    if model is None:
        model = load_tts_model()
    # Generar waveform (numpy array)
    wav = model.tts(text)
    # Guardar en buffer WAV en formato PCM
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def load_whisper_model(model_size: str = "small"):
    """
    Carga el modelo Whisper indicado.
    Devuelve (model, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size).to(device)
    return model, device


def preprocess_audio_file(input_path: str, target_sr: int = 16000) -> str:
    """
    Carga un archivo de audio, lo convierte a mono y lo resamplea a target_sr.
    Devuelve la ruta de un nuevo WAV en formato PCM 16kHz mono.
    """

    # Leer audio manteniendo la tasa original
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Resample si es necesario
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Guardar WAV preprocesado
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base}_16k_mono.wav")
    sf.write(out_path, y, target_sr, subtype='PCM_16')
    return out_path


def transcribe_audio_file(model, device, audio_path: str) -> str:
    """
    Transcribe el archivo de audio usando Whisper.
    Retorna el texto transcrito.
    """

    # Usar Whisper para transcribir
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    return text