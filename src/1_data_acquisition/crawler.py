# crawler.py
"""
Este módulo extrae el texto de cada archivo PDF generado en el paso anterior, utilizando la librería pdfminer, y guarda el contenido en un archivo .txt. Los archivos PDF se encuentran en la ruta: '../../data/outputs/1_data_acquisition/fetcher' y los archivos de texto se guardarán en: '../../data/outputs/1_data_acquisition/crawler' con el formato 'NombreMedicamento.txt'.
"""

# Librerías
import os
from pdfminer.high_level import extract_text

# Función para convertir PDF a TXT
def convertir_pdf_a_txt(ruta_pdf, ruta_txt):
    """
    Extrae el texto de un archivo PDF y lo guarda en un archivo de texto.

    Parámetros:
        ruta_pdf (str): Ruta del archivo PDF a procesar.
        ruta_txt (str): Ruta donde se guardará el archivo de texto resultante.

    Retorna:
        None. Imprime mensajes en consola indicando el éxito o error en la conversión.
    """
    try:
        # Extraer el texto del PDF utilizando pdfminer
        texto = extract_text(ruta_pdf)
        with open(ruta_txt, "w", encoding="utf-8") as archivo:
            archivo.write(texto)
        print(f"Texto guardado en: {ruta_txt}")
    except Exception as e:
        print(f"Error procesando {ruta_pdf}: {e}")

# Función para procesar todos los PDFs en una carpeta
def procesar_pdfs(carpeta_pdf, carpeta_txt):
    """
    Recorre todos los archivos PDF en una carpeta, extrae su contenido y lo guarda
    en archivos de texto en la carpeta especificada.

    Parámetros:
        carpeta_pdf (str): Ruta de la carpeta que contiene los archivos PDF.
        carpeta_txt (str): Ruta de la carpeta donde se guardarán los archivos de texto.

    Retorna:
        None.
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_txt):
        os.makedirs(carpeta_txt)
    
    # Iterar sobre cada archivo en la carpeta de PDFs
    for nombre_archivo in os.listdir(carpeta_pdf):
        if nombre_archivo.lower().endswith(".pdf"):
            ruta_pdf = os.path.join(carpeta_pdf, nombre_archivo)
            nombre_txt = os.path.splitext(nombre_archivo)[0] + ".txt"
            ruta_txt = os.path.join(carpeta_txt, nombre_txt)
            convertir_pdf_a_txt(ruta_pdf, ruta_txt)

# Función principal
def main():
    """
    Función principal que orquesta el proceso de conversión de archivos PDF a archivos de texto.
    """
    # Definir la ruta de la carpeta que contiene los PDFs y la carpeta de salida para los TXT
    carpeta_pdf = os.path.join("data", "outputs", "1_data_acquisition", "fetcher")

    # Crear la carpeta de salida si no existe
    os.makedirs(os.path.join("data", "outputs", "1_data_acquisition", "crawler"), exist_ok=True)
    carpeta_txt = os.path.join("data", "outputs", "1_data_acquisition", "crawler")
    
    procesar_pdfs(carpeta_pdf, carpeta_txt)

# Ejecución del script
if __name__ == "__main__":
    main()
