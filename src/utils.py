# utils.py

# Librerías
import os
import json
import torch
import os
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from optimum.neural_compressor import PostTrainingQuantConfig, INCQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPT2TokenizerFast, GPT2LMHeadModel

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

# Función para seleccionar el dispositivo (CPU, CUDA o MPS)
def _select_device():
    """Determina el dispositivo disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Función para contar tokens en un texto (MODELO GPT-2)
def num_tokens_gpt2(texto: str) -> int:
    """
    Cuenta el número de tokens usando el tokenizador GPT-2 de tiktoken.
    """
    encoding = tiktoken.get_encoding("gpt2")
    num_tokens = len(encoding.encode(texto))
    print(f"Número de tokens que entran al modelo GPT-2: {num_tokens}")


# Función para generar la respuesta del modelo
def generate_answer(query: str, context: str, model, tokenizer, model_name = "llama2"):
    """
    Genera una respuesta usando GPT-2 o Llama2 según model_name.

    Parámetros:
    - query: la consulta del usuario.
    - context: texto con fragmentos recuperados.
    - model_name: "gpt2" o "llama2".

    Retorna:
    - str: la respuesta generada.
    """

    # Construir el prompt según el modelo
    if model_name == "gpt2":
        prompt = build_prompt(context, query, model_name)
        num_tokens_gpt2(prompt) # obtener el número de tokens
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
    else:  # llama2-chat
        prompt = build_prompt(context, query)
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

    # Seleccionar dispositivo
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Usando dispositivo:", device)

    # Mover modelo y tensores
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if model_name == "gpt2":
       # Búsqueda con beams (más determinista y coherente)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    elif model_name == "llama2":
        # Generación de texto
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
            )

    # Decodificar y extraer solo la respuesta
    #full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #response = full_text[len(prompt):].strip()

    # número de tokens de entrada
    input_len = inputs["input_ids"].shape[-1]

    # descartamos los tokens del prompt
    gen_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return response


# Función para cargar el modelo LLaMA y el tokenizador
def load_llama_model():
    # Detectar el dispositivo disponible: CUDA, MPS (para Mac con Apple Silicon) o CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Usando dispositivo:", device)

    # Nombre del modelo a cargar (Llama-2-7b Chat)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # Cargar el tokenizador incluyendo el token de autenticación
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Cargar el modelo, especificando el tipo de datos y usando device_map="auto" para aprovechar la GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map="auto",
    )

    return model, tokenizer


# Función para cargar el modelo GPT-2
def load_gpt2_model(model_name="gpt2-medium"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Usando dispositivo:", device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT‑2 no tiene *pad* por defecto ⇒ usa el EOS
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    return model, tokenizer


# [PRUEBA]: Función para buscar fragmentos relevantes para el modelo (RAG)
def retrieve_relevant_fragments_prueba(query, model, fragments, index, k=10):
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


# Función para buscar fragmentos relevantes para el modelo (RAG)
def retrieve_relevant_fragments(query, embedding_model, fragments, index, model_name):
    """
    Realiza una búsqueda en FAISS para encontrar los fragmentos más similares a la consulta.

    Parámetros:
    - query (str): La consulta en lenguaje natural.
    - k (int): Número de resultados a recuperar.

    Retorna:
    - Lista de fragmentos de texto relevantes.
    """

    if model_name == "llama2":
        k = 10
    elif model_name == "gpt2":
        k = 3
    
    # Convertir la consulta en embedding
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)

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


# Función para formatear el contexto para el modelo
def format_context(retrieved_fragments, max_fragments=10, max_text_length=2000):
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


# Función para construir el prompt para el modelo
def build_prompt(context, query, model_name="llama2"):
    """
    Construye el prompt para el modelo con base en el contexto y la consulta,
    incluyendo un ejemplo de cómo debe formatear la respuesta.

    Parámetros:
    - context (str): Contexto a proporcionar al modelo
    - query (str): Consulta del usuario

    Retorna:
    - str: Prompt completo para el modelo
    """
    if model_name == "llama2":
        prompt = f"""
        
        1. OBJETIVO GENERAL:
        Eres un asistente médico especializado en información sobre medicamentos. Debes responder a la pregunta del usuario basándote únicamente en la información proporcionada. No debes inventar ni suponer información adicional.

        2. FORMATO DEL CONTEXTO:
        El contexto se presenta como un texto con varios fragmentos que contiene información de uno o varios medicamentos presentes en la pregunta del ususario, donde cada fragmento tiene el siguiente formato:
        - Medicamento: Nombre del medicamento
        - Categoría: Categoría de la información (ej. efectos secundarios, interacciones)
        - Información: Texto relevante sobre el medicamento, el cual debes analizar antes de responder.

        3. FORMATO DE RESPUESTA:
        - Debes responder de manera clara y precisa a la pregunta formulada por el usuario, utilizando ÚNICAMENTE el contexto que se te está proporcionando. 
        - No debes inventar información ni suponer datos que no estén presentes en el contexto.
        - Si la información proporcionada no es suficiente para responder completamente, dilo e indica qué datos faltan.
        - Incluye una mención explícita a los textos que respaldan tu respuesta, indicando para todos ellos el medicamento, la categoría y el enlace de la ficha técnica de los medicamentos correspondientes.
        - Si la pregunta no está relacionada con medicamentos, indica que no puedes ayudar en ese caso.

        4. EJEMPLO DE CONSULTA Y DE RESPUESTA ESPERADA:
        Pregunta: ¿Cuáles son los efectos secundarios de la aspirina?
        Contexto:
            "medicamento": "Aspirina"
            "categoria": "efectos_secundarios"
            "texto": "Puede causar náuseas y dolor de estómago."
        Respuesta:
        La aspirina puede causar efectos secundarios como náuseas y dolor de estómago (extraído de la ficha técnica, de la sección "efectos_secundarios" del medicamento "ASPIRINA": "la aspirina tiene como efectos secundarios, entre ottros, la aparición de náuseas y dolor de tripa"). Si necesitas más detalles, por favor consulta la ficha técnica completa.

        5. INSTRUCCIONES FINALES:
        Básandote ÚNICAMENTE en la información proporcionada en ({context}), responde a la siguiente pregunta:
        {query}

        Respuesta:"""

    elif model_name == "gpt2":
        prompt = f"""
        INSTRUCCIÓN:
        Eres un asistente médico especializado en información sobre medicamentos. Debes responder a la pregunta del usuario basándote únicamente en el contexto proporcionado. 

        EJEMPLO:
        Pregunta: ¿Cuáles son los efectos secundarios de la aspirina?
        Respuesta: La aspirina puede causar náuseas y dolor de estómago.

        Contexto:
        {context}

        Pregunta: {query}
        Respuesta a la pregunta:"""
        
    return prompt


# Función para generar la respuesta del modelo
def load_model_and_tokenizer(model_name: str):
    """
    Carga el modelo y el tokenizador apropiado según model_name.
    """

    # Detectar dispositivo
    device = _select_device()
    
    if model_name == "gpt2":
        ''' 
        Nombre | Parámetros aproximados | Comentario
        gpt2 | 124 M | “GPT-2 small”, el valor por defecto
        gpt2-medium | 350 M | Intermedio
        gpt2-large| 774 M | Grande
        gpt2-xl | 1 500 M | Extra-large
        distilgpt2 | 82 M | Versión destilada, más ligera
        '''
        # En dispositivos MPS, evitar cargas grandes sin cuantización
        model_name_gpt2 = "gpt2-large"
        if device == "mps" and model_name_gpt2 in ("gpt2-large", "gpt2-xl"):
            # --- Cuantización dinámica INT8 con Optimum ---
            # 1) Definimos la configuración de cuantización
            quant_config = PostTrainingQuantConfig(approach="dynamic")
            # 2) Creamos el quantizador a partir del modelo Hugging Face
            quantizer = INCQuantizer.from_pretrained(
                model_name_gpt2,
                quantization_config=quant_config
            )
            # 3) Cargamos el modelo base en FP32
            base_model = GPT2LMHeadModel.from_pretrained(model_name_gpt2)
            # 4) Aplicamos cuantización dinámica INT8
            model = quantizer.quantize(model=base_model)
            # 5) Movemos el modelo cuantizado a MPS
            model.to(device)
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name_gpt2)
            model = GPT2LMHeadModel.from_pretrained(model_name_gpt2)

        # Aseguramos un token de pad
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        

    elif model_name == "llama2":
        # Llama2-chat usa SentencePiece; trust_remote_code para cargar implementaciones custom
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    return tokenizer, model


# Función para responder a la consulta del usuario
def answer_query(query, model_name="llama2"):
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
    fragments = load_json("./data/outputs/5_chatbot/contexto_medicamentos_chatbot.json") 
    index = faiss.read_index("./data/outputs/5_chatbot/faiss_index_IndexFlatL2.bin") # old

    # 3. Busca los fragmentos relevantes
    #retrieved_fragments = retrieve_relevant_fragments_prueba(query, embedding_model, fragments, index, k=5)
    retrieved_fragments = retrieve_relevant_fragments(query, embedding_model, fragments, index, model_name)

    # 4. Aplicamos formateo al contexto
    print(f"Fragmentos recuperados: {retrieved_fragments}")
    context = format_context(retrieved_fragments, max_fragments=10, max_text_length=2000)

    # 5. Cargar el modelo y el tokenizador
    tokenizer, model = load_model_and_tokenizer(model_name)

    # 6. Generamos la respuesta del modelo en base al prompt y el contexto
    print(f"Contexto: {context}")
    response = generate_answer(query, context, model, tokenizer, model_name)

    return response


