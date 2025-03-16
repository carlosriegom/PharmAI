# PharmAI

# Index

1. **Data Acquisition**
   - Sacar Lista de Principios activos (+2K).
   - Scraper hacia esta url: https://cima.aemps.es/cima/publico/lista.html, donde tendríamos que meter como input cada principio activo, de ahí habría que clickar en EXPORTAR, que nos descarga un excel con el NÚMERO DE REGISTRO
   - Scraper hacia URL de este formato: https://cima.aemps.es/cima/dochtml/ft/47178/FT_47178.html, que nos parsearíamos en local la información de cada medicamento (FICHA TÉCNICA)
   - Estructuración en JSON con una IA generativa.

3. **Data Preprocessing**
   - Read the technical data sheet and extract JSON with medication fields (prompt 1)

4. **Exploratory Data Analysis (EDA)**
   - Basic analysis
   - Clustering

5. **Contraindications Detection**
   - Provide the medication to the model and have it explain contraindications

6. **Alternative Suggestions**
   - Provide the medication and medication history, and the model offers alternative medications without contraindications
