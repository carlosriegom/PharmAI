import whisper
import torch
import os
import soundfile as sf
import librosa
import tempfile
import io
from TTS.api import TTS
import numpy as np
import librosa
import os
import librosa.display
from scipy.signal import butter, filtfilt
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

try:
    import noisereduce as nr

    _HAS_NOISEREDUCE = True
except ImportError:
    _HAS_NOISEREDUCE = False

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
        _tts_model = TTS(
            model_name=model_name, progress_bar=False, gpu=torch.cuda.is_available()
        )
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
    sf.write(out_path, y, target_sr, subtype="PCM_16")
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


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    """
    Diseña un filtro Butterworth pasa-banda.

    Parámetros:
    - lowcut: frecuencia de corte baja en Hz.
    - highcut: frecuencia de corte alta en Hz.
    - fs: frecuencia de muestreo en Hz.
    - order: orden del filtro (defecto=4).

    Retorna:
    - b, a: coeficientes del filtro.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(
    y: np.ndarray,
    sr: int,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Aplica un filtro pasa-banda para conservar sólo la banda de voz.

    Parámetros:
    - y: señal de audio (1D numpy array).
    - sr: frecuencia de muestreo en Hz.
    - lowcut, highcut: límites de frecuencia en Hz.
    - order: orden del filtro.

    Retorna:
    - Señal filtrada.
    """
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return filtfilt(b, a, y)


def pre_emphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Aplica filtro de pre-énfasis para realzar frecuencias altas.

    Fórmula:
        y[n] = x[n] - coef * x[n-1]

    Parámetros:
    - y: señal mono (1D numpy array).
    - coef: coeficiente de pre-énfasis (defecto=0.97).

    Retorna:
    - Señal con pre-énfasis aplicada.
    """
    return np.append(y[0], y[1:] - coef * y[:-1])


def normalize_peak(y: np.ndarray) -> np.ndarray:
    """
    Normalización por pico: escala la señal para que su amplitud máxima sea 1.

    Parámetros:
    - y: señal de audio.

    Retorna:
    - Señal normalizada.
    """
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def normalize_rms(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalización RMS: ajusta la señal para que su RMS coincida con target_rms.

    Parámetros:
    - y: señal de audio.
    - target_rms: RMS objetivo (defecto=0.1).

    Retorna:
    - Señal normalizada.
    """
    rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / rms) if rms > 0 else y


def frame_signal(
    y: np.ndarray, sr: int, frame_length: int = None, hop_length: int = None
) -> np.ndarray:
    """
    Divide la señal en frames de tamaño fijo.

    Parámetros:
    - y: señal de audio.
    - sr: frecuencia de muestreo.
    - frame_length: tamaño del frame en muestras (defecto 25 ms).
    - hop_length: salto entre frames en muestras (defecto 10 ms).

    Retorna:
    - Matriz de shape (n_frames, frame_length).
    """
    if frame_length is None:
        frame_length = int(0.025 * sr)
    if hop_length is None:
        hop_length = int(0.010 * sr)
    num_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(num_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0]),
    )
    return frames


def apply_window(frames: np.ndarray, window_fn=np.hamming) -> np.ndarray:
    """
    Aplica ventana de suavizado a cada frame para reducir leakage.

    Parámetros:
    - frames: matriz (n_frames, frame_length).
    - window_fn: función de ventana (defecto np.hamming).

    Retorna:
    - Frames aplicando la ventana.
    """
    win = window_fn(frames.shape[1])
    return frames * win[None, :]


def reduce_noise(y: np.ndarray, sr: int, prop_decrease: float = 1.0) -> np.ndarray:
    """
    Realiza reducción de ruido por spectral gating usando noisereduce.

    Parámetros:
    - y: señal de audio.
    - sr: frecuencia de muestreo.
    - prop_decrease: proporción de reducción (0 a 1).

    Retorna:
    - Señal con ruido reducido.

    Nota: requiere instalar noisereduce (pip install noisereduce).
    """
    if not _HAS_NOISEREDUCE:
        raise ImportError("Para usar reduce_noise instala la librería noisereduce.")
    return nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)


def plot_waveform(y: np.ndarray, sr: int, title: str):
    """
    Grafica la forma de onda (waveform) de la señal de audio.

    Parámetros:
    - y: señal de audio (1D numpy array).
    - sr: frecuencia de muestreo.
    - title: título para el gráfico.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.show()


def plot_spectrogram(y: np.ndarray, sr: int, title: str):
    """
    Grafica el espectrograma (espectrograma logarítmico).

    Parámetros:
    - y: señal de audio (1D numpy array).
    - sr: frecuencia de muestreo.
    - title: título para el gráfico.
    """
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.show()


def preprocess_audio(
    path: str,
    sr: int = 16000,
    trim_db: float = 20.0,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    pre_coef: float = 0.97,
    rms_target: float = 0.1,
    reduce_noise_flag: bool = False,
) -> np.ndarray:
    """
    Pipeline completo de preprocesado de audio:
    1) Carga y conversión a mono.
    2) (Opcional) Reducción de ruido.
    3) Recorte de silencio inicial y final.
    4) Pre-énfasis.
    5) Filtro pasa-banda para voz.
    6) Normalización por RMS.

    Parámetros:
    - path: ruta al archivo de audio.
    - sr: frecuencia de muestreo deseada.
    - trim_db: umbral dB para recorte de silencio.
    - lowcut, highcut: límites de frecuencia para filtrado.
    - pre_coef: coeficiente de pre-énfasis.
    - rms_target: RMS objetivo para normalización.
    - reduce_noise_flag: activar reducción de ruido.

    Retorna:
    - Señal procesada lista para extracción de features.
    """

    # 1) Carga
    y, _ = librosa.load(path, sr=sr, mono=True)

    # 2) Reducción de ruido si se solicita
    if reduce_noise_flag:
        y = reduce_noise(y, sr)

    # 3) Recorte de silencio
    y, _ = librosa.effects.trim(y, top_db=trim_db)

    # 4) Pre-énfasis
    y = pre_emphasis(y, coef=pre_coef)

    # 5) Filtrado pasa-banda
    y = bandpass_filter(y, sr, lowcut, highcut)

    # 6) Normalización RMS
    y = normalize_rms(y, target_rms=rms_target)

    return y


def extract_mfcc(y, sr, n_mfcc=13, plot=False):
    """
    Calcula MFCCs y opcionalmente los muestra.
    Retorna: mfccs (n_mfcc × T)
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if plot:
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mfccs, x_axis="time")
        plt.ylabel("MFCC")
        plt.colorbar()
        plt.title("MFCCs")
        plt.tight_layout()
        plt.show()
    return mfccs


def extract_chroma(y, sr, plot=False):
    """
    Calcula Chroma STFT y opcionalmente lo muestra.
    Retorna: chroma (12 × T)
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    if plot:
        plt.figure(figsize=(10, 2))
        librosa.display.specshow(chroma, y_axis="chroma", x_axis="time")
        plt.title("Chroma STFT")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    return chroma


def extract_spectral_contrast(y, sr, n_bands=6, plot=False):
    """
    Calcula contraste espectral y opcionalmente lo muestra.
    Retorna: contrast (n_bands+1 × T)
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
    if plot:
        plt.figure(figsize=(10, 2))
        librosa.display.specshow(contrast, x_axis="time")
        plt.ylabel("Bandas")
        plt.colorbar()
        plt.title("Spectral Contrast")
        plt.tight_layout()
        plt.show()
    return contrast


def extract_tonnetz(y, sr, plot=False):
    """
    Calcula Tonnetz y opcionalmente lo muestra.
    Retorna: tonnetz (6 × T)
    """
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    if plot:
        plt.figure(figsize=(10, 2))
        librosa.display.specshow(tonnetz, y_axis="tonnetz", x_axis="time")
        plt.title("Tonnetz")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    return tonnetz


def extract_zcr(y, sr, plot=False):
    """
    Calcula Zero-Crossing Rate (1×T). Si plot=True, dibuja un line plot.
    """
    zcr = librosa.feature.zero_crossing_rate(y)  # shape=(1, T)
    if plot:
        times = librosa.frames_to_time(np.arange(zcr.shape[1]), sr=sr)
        plt.figure(figsize=(10, 3))
        plt.plot(times, zcr[0], linewidth=1)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("ZCR")
        plt.title("Zero-Crossing Rate")
        plt.tight_layout()
        plt.show()
    return zcr


def extract_centroid_rolloff(y, sr, plot=False):
    """
    Calcula Spectral Centroid y Roll-off (cada uno 1×T).
    Si plot=True, dibuja ambos en un mismo line plot.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # (1, T)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # (1, T)
    if plot:
        times = librosa.frames_to_time(np.arange(centroid.shape[1]), sr=sr)
        plt.figure(figsize=(10, 4))
        plt.plot(times, centroid[0], label="Centroid", linewidth=1)
        plt.plot(times, rolloff[0], label="Rolloff", linewidth=1)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title("Spectral Centroid & Roll-off")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return centroid, rolloff
