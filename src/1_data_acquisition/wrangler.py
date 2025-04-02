# wrangler.py

# Este script se encarga de procesar archivos TXT que contienen información sobre medicamentos.
# Extrae secciones relevantes, limpia el texto y guarda los resultados en formato JSON.
# El script permite procesar un solo archivo o varios archivos a la vez, y maneja errores de lectura y escritura.
# Además, incluye funcionalidades para limpiar caracteres especiales y formatear fechas.

# Librerías
import re
import json
import os
import logging
from typing import Union
import sys

# Configuración del logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Función para guardar los datos en un archivo JSON
def guardar_json(data: dict, output_path: str) -> None:
    """Guarda los datos en un archivo JSON en la ruta especificada."""
    try:
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except (OSError, IOError) as e:
        logging.exception("Error guardando JSON")
        raise RuntimeError(f"Error guardando JSON: {str(e)}") from e


# Función para limpiar el diccionario
def limpiar_diccionario(data: dict) -> dict:
    """
    Aplica técnicas de limpieza al diccionario de datos, incluyendo:
    - Eliminación del carácter rectangular vertical (U+F06D)
    - Normalización de espacios y saltos de línea
    - Eliminación de numeraciones tipo "X de Y"
    - Conversión a minúsculas para todos los campos, excepto para el código ATC
    - Limpieza específica de la sección de reacciones adversas
    """

    # 1. Limpieza específica de fecha_revision
    def limpiar_fecha_revision(data: dict) -> dict:
        if "fecha_revision" in data and data["fecha_revision"]:
            texto = data["fecha_revision"].strip()  # Limpiamos espacios en blanco
            # Primer intento: buscar la subcadena exacta "la información detallada"
            indice = texto.lower().find("la información detallada")
            if indice != -1:
                texto = texto[:indice].strip()
            else:
                # Si no se encuentra, usamos una expresión regular flexible
                patron = re.compile(
                    r"(?i)\s*la\s+informaci[oó]n\s+detallada.*$", re.IGNORECASE
                )
                texto = patron.sub("", texto).strip()

            # Nueva funcionalidad: conversión de "MM/YYYY" → "mes año"
            match = re.fullmatch(r"(\d{1,2})/(\d{4})", texto)
            if match:
                mes_num, anio = match.groups()
                meses = {
                    "1": "enero",
                    "01": "enero",
                    "2": "febrero",
                    "02": "febrero",
                    "3": "marzo",
                    "03": "marzo",
                    "4": "abril",
                    "04": "abril",
                    "5": "mayo",
                    "05": "mayo",
                    "6": "junio",
                    "06": "junio",
                    "7": "julio",
                    "07": "julio",
                    "8": "agosto",
                    "08": "agosto",
                    "9": "septiembre",
                    "09": "septiembre",
                    "10": "octubre",
                    "11": "noviembre",
                    "12": "diciembre",
                }
                mes = meses.get(mes_num, mes_num)
                texto = f"{mes} {anio}"

            data["fecha_revision"] = texto
        return data

    data = limpiar_fecha_revision(data)

    # Función para convertir texto a minúsculas
    def convertir_a_minusculas(texto: str) -> str:
        return texto.lower() if texto else texto

    # Función para limpiar saltos de línea
    def limpiar_saltos_linea(texto: str) -> str:
        return re.sub(r"[\n\r]+", " ", texto) if texto else texto

    # Función para limpiar numeraciones
    def limpiar_numeraciones(texto: str) -> str:
        if texto:
            texto = re.sub(r"(?i)\b\d+\s+de\s+\d+\b", "", texto)
            return re.sub(r"\s+", " ", texto).strip()
        return texto

    # Función unificada de limpieza de caracteres especiales
    def limpiar_caracteres_especiales(texto: str) -> str:
        if not texto:
            return texto

        replacements = {
            "\uf06d": " ",  # 
            "\u2502": " ",  # │
            "\u2551": " ",  # ║
            "\u007c": " ",  # |
            "\uffe8": " ",  # ￨
            "\u066d": " ",  # ٭
            "\u2022": " ",  # •
            "\u25cf": " ",  # ●
        }

        texto = texto.encode("utf-8", "ignore").decode("utf-8")
        patron = re.compile("|".join(re.escape(k) for k in replacements.keys()))
        texto = patron.sub(lambda x: replacements[x.group()], texto)
        texto = re.sub(r"[^\x00-\x7F\u00A0-\u00FF\u0100-\u017F]", " ", texto)
        return re.sub(r"\s+", " ", texto).strip()

    # Función para limpiar referencias a secciones
    def limpiar_referencias_secciones(texto: str) -> str:
        if not texto:
            return texto

        patron = re.compile(
            r"\(?\bver\s+secci[oó]n\s+(\d+\.\d+(?:\s*(?:,|y|e)\s*\d+\.\d+)*)+\.?\)?\.?",
            flags=re.IGNORECASE,
        )
        return patron.sub("", texto).strip()

    # Función para limpiar el apartado de reacciones adversas
    def limpiar_reacciones_adversas(texto: str) -> str:
        if not texto:
            return texto

        patron = re.compile(
            r"\s*se invita a los profesionales sanitarios a notificar las sospechas de reacciones adversas a través del sistema español de farmacovigilancia de medicamentos de uso humano: https://www\.notificaram\.es\s*",
            flags=re.IGNORECASE,
        )
        return patron.sub("", texto).strip()

    # Aplicar las funciones de limpieza a cada sección del diccionario
    for seccion, contenido in data.items():
        if isinstance(contenido, str):
            # Para todos los campos, se aplica conversión a minúsculas, excepto para el código ATC
            if seccion.lower() != "atc":
                contenido = convertir_a_minusculas(contenido)

            contenido = limpiar_saltos_linea(contenido)
            contenido = limpiar_caracteres_especiales(contenido)
            contenido = limpiar_numeraciones(contenido)
            contenido = limpiar_referencias_secciones(contenido)

            if seccion == "reacciones_adversas":
                contenido = limpiar_reacciones_adversas(contenido)

            contenido = re.sub(r"\s+", " ", contenido).strip()
            data[seccion] = contenido

    return data


# Función para extraer las secciones relevantes de un archivo TXT, exportar a JSON y guardar
def extract_secciones(file_path: str) -> dict:
    """Extrae las secciones relevantes de un archivo TXT excluyendo los títulos y genera un diccionario."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except (FileNotFoundError, IOError) as e:
        logging.exception(f"Error leyendo archivo {file_path}")
        raise RuntimeError(f"Error leyendo archivo: {str(e)}") from e

    regex_secciones = {
        "indicaciones": r"^4\.1\.?\s*Indicaciones\s+terapéuticas",
        "posologia": r"^4\.2\.?\s*Posolog[ií]a\s+y\s+forma\s+de\s+administraci[oó]n",
        "contraindicaciones": r"^4\.3\.?\s*Contraindicaciones",
        "advertencias": r"^4\.4\.?\s*Advertencias\s+y\s+precauciones\s+especiales\s+de\s+empleo",
        "interacciones": r"^4\.5\.?\s*Interacci[oó]n\s+con\s+otros\s+medicamentos",
        "fertilidad_embarazo": r"^4\.6\.?\s*Fertilidad,\s+embarazo\s+y\s+lactancia",
        "efectos_conducir": r"^4\.7\.?\s*Efectos\s+sobre\s+la\s+capacidad\s+para\s+conducir",
        "reacciones_adversas": r"^4\.8\.?\s*Reacciones\s+adversas",
        "sobredosis": r"^4\.9\.?\s*Sobredosis",
        "ATC": r"^5\.1\.?\s*Propiedades\s+farmacodin[aá]micas",
        "Propiedades_farmacocineticas": r"^5\.2\.?\s*Propiedades\s+farmacocin[eé]ticas",
        "excipientes": r"^6\.1\.?\s*Lista\s+de\s+excipientes",
        "incompatibilidades": r"^6\.2\.?\s*Incompatibilidades",
        "precauciones_conservacion": r"^6\.4\.?\s*Precauciones\s+especiales\s+de\s+conservaci[oó]n",
        "fecha_revision": r"^10\.\s*FECHA\s+DE\s+LA\s+REVISI[OÓ]N\s+DEL\s+TEXTO",
    }

    current_section = None
    data = {key: [] for key in regex_secciones.keys()}

    control_flags = {
        "in_propiedades_farmaco": False,
        "in_excipientes": False,
        "in_incompatibilidades": False,
        "in_precauciones": False,
        "in_sobredosis": False,
        "capture_atc": False,
        "ignore_until_10": False,
    }

    # Bucle principal para procesar las líneas del archivo
    for line in lines:
        line = line.strip()

        # ===== LÓGICA PARA PROPIEDADES FARMACOCINÉTICAS (5.2 a 6. DATOS FARMACÉUTICOS) =====
        if control_flags["in_propiedades_farmaco"]:
            if re.match(r"^6\s*\.?\s*DATOS\s+FARMACÉUTICOS", line, re.IGNORECASE):
                control_flags["in_propiedades_farmaco"] = False
            else:
                if line and not re.match(r"^5\.2", line):
                    data["Propiedades_farmacocineticas"].append(line)
            continue

        # ===== LÓGICA PARA EXCIPIENTES (6.1 a 6.2) =====
        if control_flags["in_excipientes"]:
            if re.match(
                r"^6\s*\.?\s*2\s*\.?\s*Incompatibilidades", line, re.IGNORECASE
            ):
                control_flags["in_excipientes"] = False
                control_flags["in_incompatibilidades"] = True
            else:
                if line and not re.match(r"^6\.1", line):
                    data["excipientes"].append(line)
            continue

        # ===== LÓGICA PARA INCOMPATIBILIDADES (6.2 a 6.3) =====
        if control_flags["in_incompatibilidades"]:
            if re.match(
                r"^6\s*\.?\s*3\s*\.?\s*Periodo\s+de\s+validez", line, re.IGNORECASE
            ):
                control_flags["in_incompatibilidades"] = False
            else:
                if line and not line.startswith("6.2"):
                    data["incompatibilidades"].append(line)
            continue

        # ===== LÓGICA PARA PRECAUCIONES (6.4 a 6.5) =====
        if control_flags["in_precauciones"]:
            if re.match(
                r"^6\s*\.?\s*5\s*\.?\s*NATURALEZA\s+Y\s+CONTENIDO\s+DEL\s+ENVASE",
                line,
                re.IGNORECASE,
            ):
                control_flags["in_precauciones"] = False
            else:
                if line and not line.startswith("6.4"):
                    data["precauciones_conservacion"].append(line)
            continue

        # ===== LÓGICA PARA SOBREDOSIS (4.9 a 5. PROPIEDADES) =====
        if control_flags["in_sobredosis"]:
            if re.match(
                r"^5\s*\.?\s*PROPIEDADES\s+FARMACOLÓGICAS", line, re.IGNORECASE
            ):
                control_flags["in_sobredosis"] = False
            else:
                if line and not re.match(r"^4\.9", line):
                    data["sobredosis"].append(line)
            continue

        # ===== DETECCIÓN DE SECCIONES PRINCIPALES =====
        section_found = False
        for section, pattern in regex_secciones.items():
            if re.match(pattern, line, re.IGNORECASE):
                current_section = section
                section_found = True

                # Activación de flags específicos
                if section == "Propiedades_farmacocineticas":
                    control_flags["in_propiedades_farmaco"] = True
                    current_section = None
                elif section == "excipientes":
                    control_flags["in_excipientes"] = True
                    current_section = None
                elif section == "incompatibilidades":
                    control_flags["in_incompatibilidades"] = True
                    current_section = None
                elif section == "precauciones_conservacion":
                    control_flags["in_precauciones"] = True
                    current_section = None
                elif section == "sobredosis":
                    control_flags["in_sobredosis"] = True
                    current_section = None
                elif section == "ATC":
                    control_flags["capture_atc"] = True
                    current_section = None
                break

        if section_found:
            continue

        # ===== CAPTURA DEL CÓDIGO ATC =====
        if control_flags["capture_atc"]:
            # Primer intento: buscar el código en la misma línea que "ATC" (con o sin ':')
            match = re.search(r"(?i)ATC:?\s*([A-Z0-9]{4,7})\b", line)
            if match:
                data["ATC"] = [match.group(1)]
                control_flags["capture_atc"] = False
                continue
            # Si no se encontró y la línea no está vacía, evaluar si la línea es únicamente el código
            if line:
                match = re.match(r"^([A-Z0-9]{4,7})$", line)
                if match:
                    data["ATC"] = [match.group(1)]
                    control_flags["capture_atc"] = False
                    continue
            continue

        # ===== CAPTURA NORMAL DE CONTENIDO =====
        if current_section and not any(control_flags.values()):
            data[current_section].append(line)

    # Post-procesamiento: convertir listas a strings y eliminar campos vacíos
    for key in data:
        if isinstance(data[key], list):
            data[key] = "\n".join(data[key]).strip()
            if not data[key]:
                data[key] = None

    return limpiar_diccionario(data)


# Función principal para procesar archivos
def procesar_archivos(input_path: str, output_path: str, n: int = 0) -> dict:
    """Procesa uno o varios archivos TXT y genera los JSON correspondientes.
    Si se procesa un solo archivo, se guarda con el mismo nombre.
    Si se procesan varios archivos, se combinan en un único JSON llamado 'medicamentos.json'.
    El parámetro n indica cuántos archivos de la carpeta se deben procesar (0 para todos).
    """
    resultados = {}
    total_procesados = 0
    errores = []

    if not os.path.exists(input_path):
        raise ValueError("La ruta de entrada no existe")

    try:
        if os.path.isfile(input_path):
            if input_path.lower().endswith(".txt"):
                data = extract_secciones(input_path)
                nombre_output = os.path.basename(input_path).replace(".txt", ".json")
                full_output_path = os.path.join(output_path, nombre_output)
                guardar_json(data, full_output_path)
                total_procesados += 1
                resultados[os.path.basename(input_path)] = "OK"
            else:
                raise ValueError("El archivo no es un TXT")

        elif os.path.isdir(input_path):
            txt_files = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(".txt")]
            )
            # Si n es mayor que 0, procesamos los n primeros; si n es 0, procesamos todos.
            if n > 0:
                txt_files = txt_files[:n]

            if len(txt_files) == 1:
                # Procesar un solo archivo y guardarlo con su nombre
                file_name = txt_files[0]
                file_path = os.path.join(input_path, file_name)
                try:
                    data = extract_secciones(file_path)
                    full_output_path = os.path.join(
                        output_path, file_name.replace(".txt", ".json")
                    )
                    guardar_json(data, full_output_path)
                    total_procesados += 1
                    resultados[file_name] = "OK"
                except Exception as e:
                    errores.append(file_name)
                    resultados[file_name] = f"Error: {str(e)}"
                    logging.exception(f"Error procesando el archivo {file_name}")

            else:
                # Procesar múltiples archivos y combinarlos en un único JSON
                json_final = {}
                total_archivos = len(txt_files)
                contador = 0
                for filename in txt_files:
                    file_path = os.path.join(input_path, filename)
                    try:
                        data = extract_secciones(file_path)
                        json_final[filename] = data
                        total_procesados += 1
                    except Exception as e:
                        errores.append(filename)
                        resultados[filename] = f"Error: {str(e)}"
                        logging.exception(f"Error procesando el archivo {filename}")
                    contador += 1
                    if contador % 100 == 0:
                        print(f"Procesados {contador} de {total_archivos} ficheros...")
                full_output_path = os.path.join(output_path, "medicamentos.json")
                guardar_json(json_final, full_output_path)
                # Mensaje final si no es múltiplo de 100
                if total_archivos % 100 != 0 or total_archivos == 0:
                    print(f"Procesados todos los {total_archivos} ficheros.")
        else:
            raise ValueError("La ruta no es válida")

    except Exception as e:
        resultados["error"] = f"Error general: {str(e)}"
        logging.exception("Error general en procesar_archivos")

    logging.info(f"Proceso completado. Archivos procesados: {total_procesados}")
    if errores:
        logging.warning(f"Archivos con errores ({len(errores)}): {errores}")

    return resultados


# Ejecución principal
if __name__ == "__main__":
    # Ruta donde lee los outputs (ficheros .txt) del proceso ETL anterior
    input_path = os.path.join(
        "..", "..", "data", "outputs", "1_data_acquisition", "crawler"
    )
    # Ruta donde guarda los outputs (ficheros .json) del proceso ETL
    # El output será el archivo 'medicamentos.json', conteniendo todos los ficheros procesados y estructurados
    output_path = os.path.join(
        "..", "..", "data", "outputs", "1_data_acquisition", "wrangler"
    )

    # Solicitar al usuario el número de archivos a procesar
    try:
        entrada = input(
            "Introduce el número de archivos TXT a procesar (o escribe 'todos' para procesar todos): "
        )
        if entrada.lower() == "todos":
            n = 0  # 0 indica que se procesarán todos los archivos
        else:
            n = int(entrada)
    except ValueError:
        logging.error(
            "El valor introducido no es un número entero ni la palabra 'todos'."
        )
        exit(1)

    # Llamada a la función de procesamiento
    try:
        procesar_archivos(input_path, output_path, n)
    except Exception as e:
        logging.error(f"ERROR: Fallo en el procesamiento - {str(e)}")
