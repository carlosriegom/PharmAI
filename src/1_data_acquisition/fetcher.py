# fetcher.py
"""
Este script se encarga de leer la lista de medicamentos obtenida como output del proceso ETL anterior ('medicamentos.csv') y descargar los PDFs con las fichas técnicas de cada medicamento. Cada PDF se guarda en la carpeta "data/outputs/1_data_acquisition/fetcher/" con el nombre del medicamento, en formato 'NombreMedicamento.pdf'.
"""

# Librerías
import os
import re
import pandas as pd
import requests
import time

# Función para limpiar el nombre del archivo
def sanitize_filename(nombre):
    """
    Limpia el nombre del archivo removiendo caracteres no válidos que puedan causar problemas al guardar el archivo.

    Parámetros:
        nombre (str): Nombre original del archivo.

    Retorna:
        str: Nombre del archivo limpio, donde los caracteres no válidos se han reemplazado por "_".
    """
    return re.sub(r'[\\/:"*?<>|]', "_", nombre)

# Función para descargar el PDF
def descargar_pdf(nombre, url, carpeta_pdf):
    """
    Descarga el PDF correspondiente a la ficha técnica de un medicamento a partir de su URL.
    La descarga se realiza en streaming para manejar archivos de gran tamaño.

    Parámetros:
        nombre (str): Nombre del medicamento, que se usará para nombrar el archivo PDF.
        url (str): URL desde donde se descarga el PDF.
        carpeta_pdf (str): Ruta del directorio donde se guardará el PDF.

    Retorna:
        bool: True si la descarga se realizó correctamente, False en caso de error.
    """
    try:
        respuesta = requests.get(url, stream=True, timeout=1)
        if respuesta.status_code == 200:
            ruta_pdf = os.path.join(carpeta_pdf, f"{nombre}.pdf")
            with open(ruta_pdf, "wb") as archivo:
                # Descargar el contenido en trozos de 1024 bytes
                for trozo in respuesta.iter_content(chunk_size=1024):
                    if trozo:
                        archivo.write(trozo)
            print(f"Descargado: {ruta_pdf}")
            return True
        else:
            print(f"Error al descargar {nombre} - Código de estado: {respuesta.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Fallo en la solicitud para {nombre}: {e}")
        return False

# Función para procesar el CSV de medicamentos y descargar los PDFs
def procesar_medicamentos(csv_path, carpeta_pdf):
    """
    Lee el archivo CSV que contiene la lista de medicamentos y descarga el PDF de la ficha técnica para cada medicamento.

    Parámetros:
        csv_path (str): Ruta del archivo CSV de medicamentos.
        carpeta_pdf (str): Ruta del directorio donde se guardarán los PDFs.
    """
    try:
        medicamentos = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer el archivo CSV en '{csv_path}': {e}")
        return
    
    # Crear la carpeta para los PDFs si no existe
    os.makedirs(carpeta_pdf, exist_ok=True)
    
    # Iterar sobre cada medicamento y, si existe la URL del PDF, proceder a la descarga
    for _, fila in medicamentos.iterrows():
        # Limpiar el nombre para usarlo como nombre del archivo (se reemplazan espacios por "_")
        nombre_archivo = sanitize_filename(fila["nombre"].replace(" ", "_"))
        url_pdf = fila["pdf_url"]
        
        # Verificar que la URL no sea nula (NaN)
        if pd.notna(url_pdf):
            time.sleep(1)  # Esperar un segundo entre descargas para evitar sobrecargar el servidor
            descargar_pdf(nombre_archivo, url_pdf, carpeta_pdf)

# Función principal
def main():
    """
    Función principal que orquesta el proceso completo:
    - Lee el CSV de medicamentos.
    - Descarga las fichas técnicas en formato PDF.
    """
    # Definir la ruta del archivo CSV (ajustar según la estructura de directorios)
    csv_path = os.path.join("data", "outputs", "1_data_acquisition", "spider", "medicamentos.csv")

    # Definir la carpeta destino para guardar los PDFs
    # Crear la carpeta de salida si no existe
    os.makedirs(os.path.join("data", "outputs", "1_data_acquisition", "fetcher"), exist_ok=True)
    carpeta_pdf = os.path.join("data", "outputs", "1_data_acquisition", "fetcher")
    
    procesar_medicamentos(csv_path, carpeta_pdf)

# Ejecución del script
if __name__ == "__main__":
    main()
