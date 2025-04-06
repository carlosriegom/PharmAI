# utils.py

# Librerías
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

##-------FUNCIONES GENERALES---------------------------------------------------------------##

# Función para cargar un archivo JSON y convertirlo en un diccionario
def load_json_dict(file_path):
    """
    Carga un archivo JSON y lo convierte en un diccionario.

    Args:
        file_path (str): Ruta del archivo JSON a cargar.

    Returns:
        dict: Diccionario con los datos del archivo JSON.
    """
    # Verifica si el archivo existe
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    # Carga el archivo JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Si data no es un diccionario, lanza un error
            if not isinstance(data, dict):
                raise ValueError("El JSON cargado no es un diccionario.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al cargar el archivo JSON: {e}")
    
    # Verifica si el diccionario está vacío
    if not data:
        raise ValueError("El archivo JSON está vacío.")
    
    return data

# Función para guardar un diccionario en un archivo JSON
def save_dict_to_json(dictionary, filename):
    """
    Guarda un diccionario en un archivo JSON.

    Args:
        dictionary (dict): Diccionario a guardar.
        filename (str): Ruta del archivo donde se guardará el JSON.

    Raises:
        ValueError: Si el diccionario no es válido.
        IOError: Si ocurre un error al escribir el archivo.
    """
    # Verifica si el diccionario es válido
    if not isinstance(dictionary, dict):
        raise ValueError("El objeto proporcionado no es un diccionario.")
    
    try:
        # Guarda el diccionario en un archivo JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4, ensure_ascii=False)
    except TypeError as e:
        raise ValueError(f"Error al serializar el diccionario a JSON: {e}")
    except IOError as e:
        raise IOError(f"Error al escribir el archivo JSON: {e}")
    
# Función para cargar un json como lista de diccionarios
def load_json(file_path):
    """
    Carga un archivo JSON y lo convierte en una lista de diccionarios.

    Args:
        file_path (str): Ruta del archivo JSON a cargar.

    Returns:
        list: Lista de diccionarios con los datos del archivo JSON.
    """
    # Verifica si el archivo existe
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    # Carga el archivo JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Si data no es una lista, lanza un error
            if not isinstance(data, list):
                raise ValueError("El JSON cargado no es una lista.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al cargar el archivo JSON: {e}")
    
    # Verifica si la lista está vacía
    if not data:
        raise ValueError("El archivo JSON está vacío.")
    
    return data

##-------FUNCIONES PARA CHATBOT------------------------------------------------------------##
def load_llama_model():

    # Detectar el dispositivo disponible: CUDA, MPS (para Mac con Apple Silicon) o CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Usando dispositivo:", device)

    # Nombre del modelo a cargar (Llama-2-7b)
    #model_name = "meta-llama/Llama-2-7b-hf" # Llama2 normal
    model_name = "meta-llama/Llama-2-7b-chat-hf" # Llama2 chat
    
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Cargar el modelo, especificando el tipo de datos y usando device_map="auto" para aprovechar la GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map="auto",
    )

    return model, tokenizer

def retrieve_relevant_fragments_prueba(query, model, fragments, index, k=5):
    """
    Recupera los fragmentos más relevantes para la consulta utilizando un modelo de similitud (ej. FAISS).

    Parámetros:
    - query (str): Consulta del usuario.
    - top_k (int): Número de fragmentos a recuperar.

    Retorna:
    - list: Lista de fragmentos relevantes.
    """
    # Simulación de la búsqueda, usando FAISS o similar.
    # Esto debería ser reemplazado por la implementación real que recupera los fragmentos relevantes.
    # Suponemos que "retrieved_fragments" es el resultado de una búsqueda en base de datos vectorial.
    '''
    retrieved_fragments = [
        {
            "medicamento": "Aspirina",
            "categoria": "efectos_secundarios",
            "texto": "Puede causar náuseas y dolor de estómago.",
        },
        {
            "medicamento": "Paracetamol",
            "categoria": "efectos_secundarios",
            "texto": "Puede causar problemas hepáticos en dosis altas.",
        },
    ]
    '''
    retrieved_fragments = [
        {
            "medicamento": "Paracetamol",
            "categoria": "efectos_secundarios",
            "texto": "Puede causar problemas hepáticos en dosis altas.",
        }
    ]
    return retrieved_fragments

def retrieve_relevant_fragments(query, model, fragments, index, k=10):
    """
    Realiza una búsqueda en FAISS para encontrar los fragmentos más similares a la consulta.

    Parámetros:
    - query (str): La consulta en lenguaje natural.
    - k (int): Número de resultados a recuperar.

    Retorna:
    - Lista de fragmentos de texto relevantes.
    """

    # Converir la consulta a minúsculas
    query = query.lower()
    
    # Convertir la consulta en embedding
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)

    # Buscar los k embeddings más cercanos
    distances, indices = index.search(query_embedding, k)

    # Recuperar los fragmentos correspondientes, incluyendo las distancias
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(fragments):  # Asegurar que el índice es válido
            results.append(
                {
                    **fragments[idx],  # Añadir los datos del fragmento
                    "distance": distances[0][i],  # Añadir la distancia de similitud
                }
            )

    return results

def format_context(retrieved_fragments, max_fragments=5, max_text_length=2000):
    """
    Formatea los fragmentos recuperados en un contexto para el modelo. Transforma una lista de diccionarios en un texto estructurado.

    Parámetros:
    - retrieved_fragments (list): Lista de fragmentos recuperados (diccionarios)
    - max_fragments (int): Número máximo de fragmentos a utilizar
    - max_text_length (int): Longitud máxima del texto a mostrar por fragmento

    Retorna:
    - str: Contexto formateado para el modelo
    """
    context = ""
    
    # Asegurar que no se intenten tomar más fragmentos de los que existen
    num_fragments = min(len(retrieved_fragments), max_fragments)

    for i, frag in enumerate(retrieved_fragments[:num_fragments]):
        # Verificar que cada fragmento tenga las claves necesarias
        if not all(key in frag for key in ["medicamento", "categoria", "texto"]):
            print(f"Advertencia: Fragmento {i+1} no tiene la estructura esperada.")
            continue  # Saltar fragmentos mal formateados

        medicamento = frag["medicamento"]
        categoria = frag["categoria"]
        texto = frag["texto"]

        # Limitar la longitud del texto
        truncated_text = (
            texto[:max_text_length] + "..." if len(texto) > max_text_length else texto
        )

        # Construcción del contexto
        context += f"\nFragmento {i+1}:\n"
        context += f"Medicamento: {medicamento}\n"
        context += f"Categoría: {categoria}\n"
        context += f"Información: {truncated_text}\n"

    return context

def build_prompt(context, query):
    """
    Construye el prompt para el modelo con base en el contexto y la consulta,
    incluyendo un ejemplo de cómo debe formatear la respuesta.

    Parámetros:
    - context (str): Contexto a proporcionar al modelo
    - query (str): Consulta del usuario

    Retorna:
    - str: Prompt completo para el modelo
    """
    prompt = f"""
    
    1. OBJETIVO GENERAL:
    Eres un asistente médico especializado en información sobre medicamentos. Debes responder a la pregunta del usuario basándote únicamente en la información proporcionada. No debes inventar ni suponer información adicional.

    2. FORMATO DEL CONTEXTO:
    El contexto se presenta como un texto con varios fragmentos que contiene información de uno o varios medicamentos presentes en la pregunta del ususario, donde cada fragmento tiene el siguiente formato:
    - Medicamento: Nombre del medicamento
    - Categoría: Categoría de la información (ej. efectos secundarios, interacciones)
    - Información: Texto relevante sobre el medicamento, el cual debes analizar antes de responder.

    3. FORMATO DE RESPUESTA:
    Debes responder de manera clara y precisa a la pregunta formulada por el usuario, utilizando ÚNICAMENTE el contexto que se te está proporcionando. Si la información proporcionada no es suficiente para responder completamente, indica qué datos faltan.

    4. EJEMPLOS DE CONSULTA Y DE RESPUESTA:
    - EJEMPLO 1:
        Pregunta: ¿Cuáles son los efectos secundarios de la aspirina?
        Contexto:
            "medicamento": "Aspirina"
            "categoria": "efectos_secundarios"
            "texto": "Puede causar náuseas y dolor de estómago."
        Respuesta:
        La aspirina puede causar efectos secundarios como náuseas y dolor de estómago, según la información proporcionada en el fragmento. Si necesitas más detalles, por favor consulta la ficha técnica completa.
    - EJEMPLO 2:
        Pregunta: ¿Puedo tomar medicamentoA si estoy embarazada?
        Contexto:
            "medicamento": "medicamentoA"
            "categoria": "contraindicaciones"
            "texto": "No se recomienda su uso durante el embarazo."
        Respuesta:
        No se recomienda el uso de medicamentoA durante el embarazo, según la información proporcionada en el fragmento. Si necesitas más detalles, por favor consulta la ficha técnica completa.

    5. INSTRUCCIONES FINALES:
    Básandote ÚNICAMENTE en la información proporcionada en ({context}), responde a la siguiente pregunta:
    {query}

    6. RECUERDA:
    - No debes inventar información ni suponer datos que no estén presentes en el contexto.
    - Si es posible, referencia el fragmento específico que respalda tu respuesta, indicando el medicamento y la categoría.
    - Si la información proporcionada no es suficiente para responder completamente, indica qué datos faltan.
    - Si la pregunta no está relacionada con medicamentos, indica que no puedes ayudar en ese caso.

    Respuesta:"""
    return prompt

def generate_answer(query, context, tokenizer, model):
    """
    Genera una respuesta basada en los fragmentos recuperados usando LLaMA.

    Parámetros:
    - query (str): La consulta del usuario
    - context (string): Texto formateado con los fragmentos recuperados 
    - tokenizer: Tokenizador del modelo
    - model: Modelo generativo

    Retorna:
    - str: Respuesta generada
    """

    # Construimos el prompt para el modelo
    prompt = build_prompt(context, query)

    # Tokenizamos el prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Detectar el dispositivo disponible: CUDA, MPS (para Mac con Apple Silicon) o CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Usando dispositivo:", device)

    input_ids = input_ids.to(device)

    # Generamos la respuesta usando el modelo
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 1000,  # Limita la longitud de salida
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    # Decodificamos la respuesta generada
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extraemos solo la parte de la respuesta después del prompt
    response = response[
        len(prompt) :
    ].strip()  # Ajuste para capturar la respuesta correctamente

    return response

def answer_query(query, model, tokenizer):
    """
    Realiza una consulta y genera una respuesta utilizando el modelo.

    Parámetros:
    - query (str): La consulta del usuario

    Retorna:
    - str: Respuesta generada
    """

    # 1. Converir la consulta a minúsculas
    query = query.lower()

    # 2. Recuperamos los fragmentos relevantes para la consulta
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # old
    fragments = load_json("../../data/outputs/5_chatbot/contexto_medicamentos_chatbot.json") 
    index = faiss.read_index("../../data/outputs/5_chatbot/faiss_index_old.bin") # old

    # 3. Busca los fragmentos relevantes
    retrieved_fragments = retrieve_relevant_fragments_prueba(query, embedding_model, fragments, index, k=10)
    #retrieved_fragments = retrieve_relevant_fragments(query, embedding_model, fragments, index, k=7)

    # 4. Aplicamos formateo al contexto
    print(f"Fragmentos recuperados: {retrieved_fragments}")
    context = format_context(retrieved_fragments)

    # 5. Generamos la respuesta del modelo
    print(f"Contexto: {context}")
    response = generate_answer(query, context, tokenizer, model)

    return response