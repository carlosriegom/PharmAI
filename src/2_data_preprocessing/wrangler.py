# wrangler.py

# Librerías
import re
import json
import os
import logging
from typing import Union
import sys

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Función para guardar los datos en un archivo JSON
def guardar_json(data: dict, output_path: str) -> None:
    """Guarda los datos en un archivo JSON en la ruta especificada."""
    try:
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
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
        if 'fecha_revision' in data and data['fecha_revision']:
            patron = re.compile(
                r'(\s*la información detallada y actualizada de este medicamento está disponible en la página web de la\s*'
                r'agencia española de medicamentos y productos sanitarios \(aemps\) http://www\.aemps\.gob\.es/\s*)',
                re.IGNORECASE | re.DOTALL
            )
            data['fecha_revision'] = patron.sub(' ', data['fecha_revision']).strip()
        return data
    
    data = limpiar_fecha_revision(data)

    # Función para convertir texto a minúsculas
    def convertir_a_minusculas(texto: str) -> str:
        return texto.lower() if texto else texto

    # Función para limpiar saltos de línea 
    def limpiar_saltos_linea(texto: str) -> str:
        return re.sub(r'[\n\r]+', ' ', texto) if texto else texto

    # Función para limpiar numeraciones
    def limpiar_numeraciones(texto: str) -> str:
        if texto:
            texto = re.sub(
                r'(?i)\b\d+\s+de\s+\d+\b',
                '', 
                texto
            )
            return re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    # Función unificada de limpieza de caracteres especiales
    def limpiar_caracteres_especiales(texto: str) -> str:
        if not texto:
            return texto

        replacements = {
            '\uF06D': ' ',   # 
            '\u2502': ' ',   # │
            '\u2551': ' ',   # ║
            '\u007C': ' ',   # |
            '\uFFE8': ' ',   # ￨
            '\u066D': ' ',   # ٭
            '\u2022': ' ',   # •
            '\u25CF': ' ',   # ●
        }

        texto = texto.encode('utf-8', 'ignore').decode('utf-8')
        patron = re.compile("|".join(re.escape(k) for k in replacements.keys()))
        texto = patron.sub(lambda x: replacements[x.group()], texto)
        texto = re.sub(r'[^\x00-\x7F\u00A0-\u00FF\u0100-\u017F]', ' ', texto)
        return re.sub(r'\s+', ' ', texto).strip()
    
    # Función para limpiar posología
    def limpiar_posologia(data: dict) -> dict:
        if 'posologia' in data and data['posologia']:
            patron = re.compile(
                r'(posolog[ií]a|forma\s+de\s+administraci[oó]n)\s*:\s*', 
                flags=re.IGNORECASE
            )
            texto_limpio = patron.sub('', data['posologia'])
            texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
            data['posologia'] = texto_limpio
        return data

    # Función para limpiar referencias a secciones
    def limpiar_referencias_secciones(texto: str) -> str:
        if not texto:
            return texto

        patron = re.compile(
            r'\(?\bver\s+secci[oó]n\s+(\d+\.\d+(?:\s*(?:,|y|e)\s*\d+\.\d+)*)+\.?\)?\.?',
            flags=re.IGNORECASE
        )
        return patron.sub('', texto).strip()

    # Función para limpiar el apartado de reacciones adversas
    def limpiar_reacciones_adversas(texto: str) -> str:
        if not texto:
            return texto
        
        patron = re.compile(
            r'\s*se invita a los profesionales sanitarios a notificar las sospechas de reacciones adversas a través del sistema español de farmacovigilancia de medicamentos de uso humano: https://www\.notificaram\.es\s*',
            flags=re.IGNORECASE
        )
        return patron.sub('', texto).strip()

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
            
            if seccion == 'posologia':
                contenido = limpiar_posologia({seccion: contenido})[seccion]
            if seccion == 'reacciones_adversas':
                contenido = limpiar_reacciones_adversas(contenido)
            
            contenido = re.sub(r'\s+', ' ', contenido).strip()
            data[seccion] = contenido
            
    return data

# Función para extraer las secciones relevantes de un archivo TXT, exportar a JSON y guardar
def extract_secciones(file_path: str) -> dict:
    """Extrae las secciones relevantes de un archivo TXT excluyendo los títulos y genera un diccionario."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except (FileNotFoundError, IOError) as e:
        logging.exception(f"Error leyendo archivo {file_path}")
        raise RuntimeError(f"Error leyendo archivo: {str(e)}") from e

    regex_secciones = {
        "indicaciones": r'^4\.1\.?\s*Indicaciones\s+terapéuticas',
        "posologia": r'^4\.2\.?\s*Posolog[ií]a\s+y\s+forma\s+de\s+administraci[oó]n',
        "contraindicaciones": r'^4\.3\.?\s*Contraindicaciones',
        "advertencias": r'^4\.4\.?\s*Advertencias\s+y\s+precauciones\s+especiales\s+de\s+empleo',
        "interacciones": r'^4\.5\.?\s*Interacci[oó]n\s+con\s+otros\s+medicamentos',
        "fertilidad_embarazo": r'^4\.6\.?\s*Fertilidad,\s+embarazo\s+y\s+lactancia',
        "efectos_conducir": r'^4\.7\.?\s*Efectos\s+sobre\s+la\s+capacidad\s+para\s+conducir',
        "reacciones_adversas": r'^4\.8\.?\s*Reacciones\s+adversas',
        "sobredosis": r'^4\.9\.?\s*Sobredosis',
        "ATC": r'^5\.1\.?\s*Propiedades\s+farmacodin[aá]micas',
        "Propiedades_farmacocineticas": r'^5\.2\.?\s*Propiedades\s+farmacocin[eé]ticas',
        "excipientes": r'^6\.1\.?\s*Lista\s+de\s+excipientes',
        "incompatibilidades": r'^6\.2\.?\s*Incompatibilidades',
        "precauciones_conservacion": r'^6\.4\.?\s*Precauciones\s+especiales\s+de\s+conservaci[oó]n',
        "fecha_revision": r'^10\.\s*FECHA\s+DE\s+LA\s+REVISI[OÓ]N\s+DEL\s+TEXTO'
    }

    current_section = None
    data = {key: [] for key in regex_secciones.keys()}
    
    control_flags = {
        'in_propiedades_farmaco': False,
        'in_excipientes': False,
        'in_incompatibilidades': False,
        'in_precauciones': False,
        'in_sobredosis': False,
        'capture_atc': False,
        'ignore_until_10': False
    }

    for line in lines:
        line = line.strip()
        
        # ===== LÓGICA PARA PROPIEDADES FARMACOCINÉTICAS (5.2 a 6. DATOS FARMACÉUTICOS) =====
        if control_flags['in_propiedades_farmaco']:
            if re.match(r'^6\s*\.?\s*DATOS\s+FARMACÉUTICOS', line, re.IGNORECASE):
                control_flags['in_propiedades_farmaco'] = False
            else:
                if line and not re.match(r'^5\.2', line):
                    data['Propiedades_farmacocineticas'].append(line)
            continue

        # ===== LÓGICA PARA EXCIPIENTES (6.1 a 6.2) =====
        if control_flags['in_excipientes']:
            if re.match(r'^6\s*\.?\s*2\s*\.?\s*Incompatibilidades', line, re.IGNORECASE):
                control_flags['in_excipientes'] = False
            else:
                if line and not re.match(r'^6\.1', line):
                    data['excipientes'].append(line)
            continue

        # ===== LÓGICA PARA INCOMPATIBILIDADES (6.2 a 6.3) =====
        if control_flags['in_incompatibilidades']:
            if re.match(r'^6\s*\.?\s*3\s*\.?\s*Periodo\s+de\s+validez', line, re.IGNORECASE):
                control_flags['in_incompatibilidades'] = False
            else:
                if line and not re.match(r'^6\.2', line):
                    data['incompatibilidades'].append(line)
            continue

        # ===== LÓGICA PARA PRECAUCIONES (6.4 a 6.5) =====
        if control_flags['in_precauciones']:
            if re.match(r'^6\s*\.?\s*5\s*\.?\s*NATURALEZA\s+Y\s+CONTENIDO\s+DEL\s+ENVASE', line, re.IGNORECASE):
                control_flags['in_precauciones'] = False
            else:
                if line and not re.match(r'^6\.4', line):
                    data['precauciones_conservacion'].append(line)
            continue

        # ===== LÓGICA PARA SOBREDOSIS (4.9 a 5. PROPIEDADES) =====
        if control_flags['in_sobredosis']:
            if re.match(r'^5\s*\.?\s*PROPIEDADES\s+FARMACOLÓGICAS', line, re.IGNORECASE):
                control_flags['in_sobredosis'] = False
            else:
                if line and not re.match(r'^4\.9', line):
                    data['sobredosis'].append(line)
            continue

        # ===== DETECCIÓN DE SECCIONES PRINCIPALES =====
        section_found = False
        for section, pattern in regex_secciones.items():
            if re.match(pattern, line, re.IGNORECASE):
                current_section = section
                section_found = True
                
                # Activación de flags específicos
                if section == "Propiedades_farmacocineticas":
                    control_flags['in_propiedades_farmaco'] = True
                    current_section = None
                elif section == "excipientes":
                    control_flags['in_excipientes'] = True
                    current_section = None
                elif section == "incompatibilidades":
                    control_flags['in_incompatibilidades'] = True
                    current_section = None
                elif section == "precauciones_conservacion":
                    control_flags['in_precauciones'] = True
                    current_section = None
                elif section == "sobredosis":
                    control_flags['in_sobredosis'] = True
                    current_section = None
                elif section == "ATC":
                    control_flags['capture_atc'] = True
                    current_section = None
                break

        if section_found:
            continue

        # ===== CAPTURA DEL CÓDIGO ATC =====
        if control_flags['capture_atc']:
            if "ATC:" in line:
                if match := re.search(r'ATC:\s*([A-Z0-9]{4,7})\b', line):
                    data["ATC"] = [match.group(1)]
                control_flags['capture_atc'] = False
            continue

        # ===== CAPTURA NORMAL DE CONTENIDO =====
        if current_section and not any(control_flags.values()):
            data[current_section].append(line)

    # Post-procesamiento
    for key in data:
        if isinstance(data[key], list):
            data[key] = '\n'.join(data[key]).strip()
            if not data[key]:  # Eliminar campos vacíos
                data[key] = None
    
    return limpiar_diccionario(data)

# Función principal para procesar archivos
def procesar_archivos(input_path: str, output_path: str) -> dict:
    """Procesa un archivo o carpeta y genera los JSON correspondientes."""
    resultados = {}
    total_procesados = 0
    errores = []

    if not os.path.exists(input_path):
        raise ValueError("La ruta de entrada no existe")
    
    try:
        if os.path.isfile(input_path):
            if input_path.lower().endswith('.txt'):
                data = extract_secciones(input_path)
                guardar_json(data, output_path)
                logging.info(f"Json guardado en: {output_path}")
                total_procesados += 1
                resultados[os.path.basename(input_path)] = "OK"
            else:
                raise ValueError("El archivo no es un TXT")
        
        elif os.path.isdir(input_path):
            json_final = {}
            for filename in os.listdir(input_path):
                if filename.lower().endswith('.txt'):
                    file_path = os.path.join(input_path, filename)
                    try:
                        data = extract_secciones(file_path)
                        json_final[filename] = data
                        total_procesados += 1
                    except Exception as e:
                        errores.append(filename)
                        resultados[filename] = f"Error: {str(e)}"
                        logging.exception(f"Error procesando el archivo {filename}")
            guardar_json(json_final, output_path)
        else:
            raise ValueError("La ruta no es válida")
    
    except Exception as e:
        resultados["error"] = f"Error general: {str(e)}"
        logging.exception("Error general en procesar_archivos")
    
    logging.info(f"Proceso completado. Archivos procesados: {total_procesados}")
    if errores:
        logging.warning(f"Archivos con errores ({len(errores)}): {errores}")
    
    return resultados

# Ejecución
if __name__ == "__main__":
    # Configuración manual de rutas
    input_path = os.path.join("..", "..", "data", "inputs", "1_data_acquisition", "wrangler", "A.A.S._100_mg_COMPRIMIDOS.txt")
    nombre_archivo_txt = os.path.basename(input_path).replace(".txt", "")
    output_path = os.path.join("..", "..", "data", "outputs", "1_data_acquisition", "wrangler", f"{nombre_archivo_txt}.json")
    
    # Procesar archivos
    try:
        logging.info(f"Iniciando procesamiento de: {input_path}")
        resultados = procesar_archivos(input_path, output_path)
        logging.info("Ejecución completada")
    except Exception as e:
        logging.error(f"ERROR: Fallo en el procesamiento - {str(e)}")
        raise
