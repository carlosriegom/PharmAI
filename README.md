# PharmAI

### Tabla de contenidos

1. **Requisitos**
2. **Estructura del proyecto** <br>
   2.1 **Adquisición de los datos** <br>
   2.2 **Preprocesamiento de datos** <br>
   2.3 **Análisis exploratorio de datos (EDA)** <br>
   2.4 **Machine Learning** <br>
   2.5 **Chatbot** <br>


## 1. Requisitos







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
