# PharmAI

### Tabla de contenidos

1. [Requisitos](#1-requisitos)
2. [Estructura del proyecto](#2-estructura-del-proyecto)  
   2.1 [Adquisición de los datos](#21-adquisición-de-los-datos)  
   2.2 [Preprocesamiento de datos](#22-preprocesamiento-de-datos)  
   2.3 [Análisis exploratorio de datos (EDA)](#23-análisis-exploratorio-de-datos-eda)  
   2.4 [Machine Learning](#24-machine-learning)  
   2.5 [Deep Learning: Chatbot](#25-chatbot)


## 1. Requisitos

Para poder ejecutar el proyecto, es necesario tener instalado Python 3.11.11 o superior y las siguientes librerías:

```bash
pip install -r requirements.txt
```
Con esto, ya tenremos todas las dependencias necesarias para ejecutar el proyecto.

## 2. Estructura del proyecto

Este proyecto es end-to-end y está dividido en varias partes:

### 2.1 Adquisición de los datos

Para la adquisición de los datos, se ha utilizado un scraper que obtiene la información de los medicamentos desde la página web de la [AEMPS (Agencia Española de Medicamentos y Productos Sanitarios)](https://cima.aemps.es/cima/publico/lista.html). El procedimiento es el siguiente:

1. **Spider**

   En esta parte hacemos una consulta a la API de medicamentos de la AEMPS para extraer información de todos los medicamentos autorizados (número de registro, nombre, principios activos y ficha técnica en PDF), procesa los resultados paginados eliminando duplicados y genera un fichero csv con los registros más recientes ordenados por número de registro. Para ejectutar el spider, se utiliza el siguiente comando:

   ```bash
   python blablabla.py
   ```

   Este fichero csv se llama `Medicamentos.csv`, guardado en la carpeta `data/outputs/1_data_acquisition/spider` y contiene la siguiente información:

      | Columna              | Descripción                                                                 |
   |----------------------|-----------------------------------------------------------------------------|
   | **nregistro**        | Número de registro oficial del medicamento en la AEMPS (identificador único). |
   | **nombre**           | Nombre comercial del medicamento (formato descriptivo).                     |
   | **principios_activos** | Sustancias farmacológicamente activas que componen el medicamento.         |
   | **pdf_url**          | Enlace directo a la ficha técnica en PDF (cuando está disponible).          |

<br>




2. **Fetcher**

   En esta parte descargamos las distintas fichas técnicas para todos los medicamentos que hemos obtenido en la parte anterior del spider.

3. **Crawler**

4. **Wrangler**




4. **Data Acquisition**
   - Sacar Lista de Principios activos (+2K).
   - Scraper hacia esta url: https://cima.aemps.es/cima/publico/lista.html, donde tendríamos que meter como input cada principio activo, de ahí habría que clickar en EXPORTAR, que nos descarga un excel con el NÚMERO DE REGISTRO
   - Scraper hacia URL de este formato: https://cima.aemps.es/cima/dochtml/ft/47178/FT_47178.html, que nos parsearíamos en local la información de cada medicamento (FICHA TÉCNICA)
   - Estructuración en JSON con una IA generativa.

5. **Data Preprocessing**
   - Read the technical data sheet and extract JSON with medication fields (prompt 1)

6. **Exploratory Data Analysis (EDA)**
   - Basic analysis
   - Clustering

7. **Contraindications Detection**
   - Provide the medication to the model and have it explain contraindications

8. **Alternative Suggestions**
   - Provide the medication and medication history, and the model offers alternative medications without contraindications
