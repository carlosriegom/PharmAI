# PharmAI

# Index

1. **Data Acquisition**
   - Get list of active ingredients
   - For each active ingredient, get all medications
   - For each medication, obtain the technical data sheet

2. **Data Preprocessing**
   - Read the technical data sheet and extract JSON with medication fields (prompt 1)

3. **Exploratory Data Analysis (EDA)**
   - Basic analysis
   - Clustering

4. **Contraindications Detection**
   - Provide the medication to the model and have it explain contraindications

5. **Alternative Suggestions**
   - Provide the medication and medication history, and the model offers alternative medications without contraindications
