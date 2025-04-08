# spider.py
"""
Este script obtiene información de medicamentos utilizando la API de CIMA, genera
un archivo CSV con los datos recopilados y analiza el CSV para mostrar la cantidad
de valores nulos en determinadas columnas.

El proceso se divide en dos pasos:
1. fetch_and_save_data: Solicita los datos a la API, procesa la información,
   elimina duplicados y guarda el resultado en un archivo CSV.
2. analyze_csv: Lee el CSV generado y muestra un conteo de valores nulos por columna.
"""

# Librerías
import requests
import pandas as pd
import math
import os

# Función para consultar la API de CIMA y guardar los datos de los medicamentos en un CSV
def fetch_and_save_data(csv_path):
    """
    Consulta la API de CIMA para obtener los datos de medicamentos, procesa la
    información y la guarda en un archivo CSV.

    Parámetros:
        csv_path (str): Ruta donde se guardará el archivo CSV.
    """
    # Definición de variables y parámetros para la petición
    url = "https://cima.aemps.es/cima/rest/medicamentos"
    headers = {"Accept": "application/json"}
    tamanio_pagina = 25

    # Primera petición para obtener el total de filas
    try:
        params = {"pagina": 1, "tamanioPagina": tamanio_pagina}
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        total_filas = data.get("totalFilas", 0)
    except Exception as e:
        print("Error en la solicitud inicial:", e)
        total_filas = 0

    # Si no se obtienen filas, se muestra un mensaje
    if total_filas == 0:
        print("No se han obtenido resultados.")
        registros = []
    else:
        # Calcular el total de páginas a consultar
        total_paginas = math.ceil(total_filas / tamanio_pagina)
        print("Total de medicamentos:", total_filas)
        print("Total de páginas:", total_paginas)
        
        registros = []
        # Iterar por cada página para obtener todos los registros
        for pagina in range(1, total_paginas + 1):
            params = {"pagina": pagina, "tamanioPagina": tamanio_pagina}
            print(f"Recuperando página {pagina} de {total_paginas}...")
            try:
                response = requests.get(url, params=params, headers=headers)
                data = response.json()
            except Exception as e:
                print(f"Error en la página {pagina}: {e}")
                continue  # Salta a la siguiente página si ocurre algún error

            # Procesar cada registro de la respuesta
            for med in data.get("resultados", []):
                try:
                    nregistro = med.get("nregistro")
                    nombre_med = med.get("nombre")
                    # Obtener el nombre de los principios activos desde "vtm"
                    principios_activos = med.get("vtm", {}).get("nombre")

                    # Buscar en "docs" el documento de tipo 1 (ficha técnica en PDF)
                    pdf_url = None
                    for doc in med.get("docs", []):
                        if doc.get("tipo") == 1:
                            pdf_url = doc.get("url")
                            break

                    registros.append({
                        "nregistro": nregistro,
                        "nombre": nombre_med,
                        "principios_activos": principios_activos,
                        "pdf_url": pdf_url,
                    })
                except Exception as e:
                    print("Error procesando medicamento:", e)

    # Crear un DataFrame a partir de la lista de registros
    df = pd.DataFrame(registros)

    # Ordenar el DataFrame por 'nregistro' de forma descendente y eliminar duplicados por 'nombre'
    df.sort_values("nregistro", ascending=False, inplace=True)
    print(f"Nº registros en el DataFrame antes de eliminar duplicados: {len(df)}")
    df.drop_duplicates(subset="nombre", keep="first", inplace=True)
    print(f"Nº registros tras eliminar duplicados: {len(df)}")

    # Asegurar que el directorio donde se va a guardar el CSV existe
    directorio = os.path.dirname(csv_path)
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(csv_path, index=False)
    print(f"Archivo CSV guardado en: {csv_path}")

# Función para analizar el CSV y mostrar el conteo de valores nulos
def analyze_csv(csv_path):
    """
    Lee el archivo CSV con los datos de medicamentos y muestra el conteo de
    valores nulos en cada columna.

    Parámetros:
        csv_path (str): Ruta del archivo CSV a analizar.
    """
    try:
        df = pd.read_csv(csv_path)
        print("Conteo de valores nulos en el CSV:")
        print(df.isnull().sum())
    except Exception as e:
        print("Error al leer el archivo CSV:", e)

# Función principal que ejecuta el proceso completo
def main():
    """
    Función principal que ejecuta el proceso completo: descarga de datos,
    generación del CSV y análisis de valores nulos.
    """
    print("Directorio de trabajo actual:", os.getcwd())
    # Definir la ruta del archivo CSV (se asume una estructura de directorios)
    # Crear la ruta del CSV en la carpeta de salida si no existe
    os.makedirs(os.path.join("data", "outputs", "1_data_acquisition", "spider"), exist_ok=True)
    csv_path = os.path.join("data", "outputs", "1_data_acquisition", "spider", "medicamentos.csv")

    # Obtener datos de la API y guardar en CSV
    fetch_and_save_data(csv_path)
    
    # Leer el CSV y mostrar análisis de valores nulos
    analyze_csv(csv_path)

# Ejecución del script
if __name__ == "__main__":
    main()
