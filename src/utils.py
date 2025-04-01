# utils.py

# Librerías
import os
import json

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
        with open(file_path, 'r') as file:
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
        with open(filename, 'w') as f:
            json.dump(dictionary, f, indent=4)
    except TypeError as e:
        raise ValueError(f"Error al serializar el diccionario a JSON: {e}")
    except IOError as e:
        raise IOError(f"Error al escribir el archivo JSON: {e}")
