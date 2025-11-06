from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load the trained model and preprocessor
model = joblib.load('models/randomforestclassifier_disease.pkl')
loaded_preprocessor = joblib.load('models/preprocessor_disease.pkl')
preprocessor = loaded_preprocessor['scaler']

SYMPTOM_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = {}
    for symptom in SYMPTOM_COLUMNS:
        value = request.form.get(symptom)
        if value is not None:
            if symptom == 'sex':
                symptoms[symptom] = 1.0 if value.lower() == 'male' else 0.0
            else:
                symptoms[symptom] = float(value)
        else:
            # Handle missing values, e.g., set to 0 or mean, or raise an error
            symptoms[symptom] = 0.0 # Placeholder, consider better imputation

    # Convert symptoms to a DataFrame row
    symptoms_df = pd.DataFrame([symptoms], columns=SYMPTOM_COLUMNS)
    
    # Impute missing values with the mean from the preprocessor (StandardScaler)
    if hasattr(preprocessor, 'mean_'):
        symptoms_df = symptoms_df.fillna(pd.Series(preprocessor.mean_, index=SYMPTOM_COLUMNS))
    else:
        symptoms_df = symptoms_df.fillna(0) # Fallback to 0 if mean_ is not available

    # Scale the input features
    scaled_symptoms = preprocessor.transform(symptoms_df)
    
    # Make prediction and get probabilities
    prediction = model.predict(scaled_symptoms)
    prediction_proba = model.predict_proba(scaled_symptoms)

    # Interpret prediction with confidence threshold
    max_proba = np.max(prediction_proba)
    confidence_threshold = 0.4 # Lowered confidence to 40%

    if max_proba < confidence_threshold:
        result = "Disease not identified: Low prediction confidence."
    elif prediction[0] == 1:
        result = "Presence of Heart Disease"
    else:
        result = "Absence of Heart Disease"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)