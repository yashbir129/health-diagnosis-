from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.nlp_utils import extract_patient_info
import PyPDF2

app = Flask(__name__)

# Load the trained model and preprocessor
MODEL_DIR = 'models'
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor_heart.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'randomforestclassifier_heart.pkl') # Assuming RandomForest is the chosen model

model = None
preprocessor = None

try:
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
    else:
        print("Preprocessor file not found.")
        preprocessor = None

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Model file not found.")
        model = None

except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    preprocessor = None
    model = None

def preprocess_data(df, preprocessor):
    """Applies preprocessing steps to the dataframe."""
    if df is None:
        return None

    imputer = preprocessor['imputer']
    scaler = preprocessor['scaler']

    imputed_data = imputer.transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    processed_data = scaler.transform(imputed_df)

    return processed_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def get_precautions(disease_name):
    # connection = create_db_connection() # Commented out for now
    # precautions = []
    # if connection:
    #     try:
    #         cursor = connection.cursor()
    #         sql = """
    #             SELECT p.description FROM precautions p
    #             JOIN diseases d ON p.disease_id = d.id
    #             WHERE d.name = %s
    #         """
    #         cursor.execute(sql, (disease_name,))
    #         precautions = [row[0] for row in cursor.fetchall()]
    #     except Exception as e:
    #         print(f"Error fetching precautions: {e}")
    #     finally:
    #         connection.close()
    # return precautions
    return [f"Precaution for {disease_name}: Consult a doctor.", f"Precaution for {disease_name}: Rest and hydrate."]

def get_doctors(disease_name):
    # connection = create_db_connection() # Commented out for now
    # doctors = []
    # if connection:
    #     try:
    #         cursor = connection.cursor()
    #         # This is a simplified approach. In a real app, you might map diseases to specializations.
    #         # For now, let's suggest general physicians or specialists based on disease type.
    #         if disease_name in ["Common Cold", "Flu"]:
    #             sql = "SELECT name, specialization, contact_info FROM doctors WHERE specialization = %s OR specialization = %s"
    #             cursor.execute(sql, ("General Physician", "Family Medicine"))
    #         elif disease_name == "Migraine":
    #             sql = "SELECT name, specialization, contact_info FROM doctors WHERE specialization = %s"
    #             cursor.execute(sql, ("Neurologist",))
    #         elif disease_name == "Diabetes":
    #             sql = "SELECT name, specialization, contact_info FROM doctors WHERE specialization = %s"
    #             cursor.execute(sql, ("Endocrinologist",))
    #         elif disease_name in ["Heart Disease", "Hypertension"]:
    #             sql = "SELECT name, specialization, contact_info FROM doctors WHERE specialization = %s"
    #             cursor.execute(sql, ("Cardiologist",))
    #         else:
    #             sql = "SELECT name, specialization, contact_info FROM doctors WHERE specialization = %s"
    #             cursor.execute(sql, ("General Physician",))

    #         doctors = [{ 'name': row[0], 'specialization': row[1], 'contact_info': row[2] } for row in cursor.fetchall()]
    #     except Exception as e:
    #         print(f"Error fetching doctors: {e}")
    #     finally:
    #         connection.close()
    # return doctors
    return [{'name': 'Dr. Placeholder', 'specialization': 'General Physician', 'contact_info': '123-456-7890'}]

def predict_disease(processed_data):
    if model:
        try:
            # Make prediction
            prediction = model.predict(processed_data)
            print(f"[DEBUG] Raw prediction: {prediction}")
            # Assuming the model outputs a single class label
            # You might need to map this label back to a disease name
            # For now, let's return a dummy disease name based on input for demonstration
            # In a real scenario, you'd have a mapping from model output to disease names
            disease_map = {
                0: "No Heart Disease",
                1: "Heart Disease"
            }
            predicted_output = prediction[0]
            predicted_disease_name = disease_map.get(predicted_output, "Unknown Disease")
            print(f"[DEBUG] Predicted disease name: {predicted_disease_name}")

            return predicted_disease_name
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Prediction Error"
    return "Model Not Loaded"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Redirect GET requests to the main page
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        patient_name = request.form['name']
        age = request.form.get('age')
        sex = request.form.get('sex')
        
        pdf_file = request.files.get('pdf_report')
        condition_description = request.form.get('condition_description', '')
        extracted_info = {}

        # Process PDF if uploaded
        if pdf_file and pdf_file.filename != '':
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n"
                
                # Use NLP to extract information from the medical report
                extracted_info = extract_patient_info(pdf_text)

            except Exception as e:
                print(f"Error reading PDF: {e}")
                # Handle PDF reading error, maybe set a flash message
        
        # Process condition description if provided
        if condition_description and condition_description.strip():
            try:
                # Extract medical information from user's description
                description_info = extract_patient_info(condition_description)
                # Merge with existing extracted info (PDF takes precedence if both exist)
                for key, value in description_info.items():
                    if key not in extracted_info:
                        extracted_info[key] = value
                    else:
                        # If both PDF and description have the same info, keep the PDF value
                        print(f"Keeping PDF value for {key}: {extracted_info[key]}")
                
                # Store the original description for display
                extracted_info['user_description'] = condition_description
                
            except Exception as e:
                print(f"Error processing condition description: {e}")
                # Store the description even if NLP extraction fails
                extracted_info['user_description'] = condition_description

        # Create a DataFrame for prediction
        patient_data = pd.DataFrame([{
            'age': float(age) if age else 0,
            'sex': float(sex) if sex else 0,
            'cp': float(extracted_info.get('cp') or 0),
            'trestbps': float(extracted_info.get('trestbps') or 0),
            'chol': float(extracted_info.get('chol') or 0),
            'fbs': float(extracted_info.get('fbs') or 0),
            'restecg': float(extracted_info.get('restecg') or 0),
            'thalach': float(extracted_info.get('thalach') or 0),
            'exang': float(extracted_info.get('exang') or 0),
            'oldpeak': float(extracted_info.get('oldpeak') or 0),
            'slope': float(extracted_info.get('slope') or 0),
            'ca': float(extracted_info.get('ca') or 0),
            'thal': float(extracted_info.get('thal') or 0)
        }])

        print(f"[DEBUG] Input patient_data_df:\n{patient_data.to_string()}")

        # Preprocess the data before prediction
        if preprocessor:
            processed_data = preprocess_data(patient_data, preprocessor)
            predicted_disease = predict_disease(processed_data)
        else:
            predicted_disease = "Preprocessor Not Loaded"
        precautions = get_precautions(predicted_disease)
        doctors = get_doctors(predicted_disease)

        return render_template('result.html',
                               patient_name=patient_name,
                               predicted_disease=predicted_disease,
                               precautions=precautions,
                               doctors=doctors,
                               extracted_info=extracted_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
