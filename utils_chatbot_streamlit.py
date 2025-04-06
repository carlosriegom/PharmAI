# utils_chatbot_streamlit.py

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss


def load_json(json_path):
    """Carga un archivo JSON y lo devuelve como lista de diccionarios."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def extract_fragments(data):
    """
    Extrae los fragmentos relevantes de los campos de interés para cada medicamento.
    Devuelve una lista de diccionarios con medicamento, categoría y texto.
    """
    fragments = []
    campos_interes = [
        "indicaciones",
        "posologia",
        "contraindicaciones",
        "advertencias",
        "interacciones",
        "fertilidad_embarazo",
        "efectos_conducir",
        "reacciones_adversas",
        "sobredosis",
        "Propiedades_farmacocineticas",
        "excipientes",
        "incompatibilidades",
        "precauciones_conservacion",
        "Descripcion_Nivel_Anatomico",
        "Descripcion_Nivel_2_Subgrupo_Terapeutico",
    ]

    for medicamento in data:
        nombre_medicamento = medicamento.get(
            "nombre_medicamento", "Desconocido"
        ).replace(".txt", "")
        for campo in campos_interes:
            valor = medicamento.get(campo)
            texto = "" if valor is None else valor.strip()
            if texto:
                fragments.append(
                    {
                        "medicamento": nombre_medicamento,
                        "categoria": campo,
                        "texto": texto,
                    }
                )
    return fragments


def load_model(model_name="multi-qa-MiniLM-L6-dot-v1", device="mps"):
    """
    Carga el modelo de SentenceTransformer especificado en el dispositivo indicado.
    """
    torch.set_num_threads(8)
    model = SentenceTransformer(model_name, device=device)
    return model


def generate_embeddings(model, texts, batch_size=64):
    """
    Genera embeddings a partir de una lista de textos usando el modelo dado.
    """
    embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size
    )
    return embeddings


def save_embeddings(embeddings, output_path):
    """
    Guarda los embeddings en un archivo .npy.
    """
    np.save(output_path, embeddings)


def load_embeddings(embeddings_path):
    """
    Carga embeddings desde un archivo .npy.
    """
    return np.load(embeddings_path)


def create_faiss_index(embeddings):
    """
    Crea un índice FAISS plano (FlatL2) a partir de los embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, output_path):
    """
    Guarda el índice FAISS en un archivo.
    """
    faiss.write_index(index, output_path)


def load_faiss_index(index_path):
    """
    Carga un índice FAISS desde un archivo.
    """
    return faiss.read_index(index_path)


def search_faiss(query, model, index, fragments, k=5):
    """
    Busca los k fragmentos más similares a la consulta en el índice FAISS.
    """
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(fragments):
            results.append(
                {
                    **fragments[idx],
                    "distance": distances[0][i],
                }
            )
    return results


def save_search_results(results, output_path):
    """
    Guarda los resultados de búsqueda en un archivo de texto.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, res in enumerate(results):
            f.write(f"Resultado {i+1}:\n")
            f.write(f"Medicamento: {res['medicamento']}\n")
            f.write(f"Categoría: {res['categoria']}\n")
            f.write(f"Texto: {res['texto']}\n\n")
