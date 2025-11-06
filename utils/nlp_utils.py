import spacy
import re

# Load the English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_patient_info(report_text):
    """
    Extracts patient information from a medical report using spaCy.
    Args:
        report_text (str): The full text of the medical report.
    Returns:
        dict: A dictionary containing extracted patient information.
    """
    doc = nlp(report_text)
    
    patient_info = {
        "cp": None,
        "trestbps": None,
        "chol": None,
        "fbs": None,
        "restecg": None,
        "thalach": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None
    }

    # Extract features using regex
    cp_match = re.search(r"Chest\s*Pain\s*Type\s*:\s*(\d)", report_text, re.IGNORECASE)
    if cp_match:
        patient_info["cp"] = int(cp_match.group(1))

    trestbps_match = re.search(r"Resting\s*Blood\s*Pressure\s*:\s*(\d+)", report_text, re.IGNORECASE)
    if trestbps_match:
        patient_info["trestbps"] = int(trestbps_match.group(1))

    chol_match = re.search(r"Cholesterol\s*:\s*(\d+)", report_text, re.IGNORECASE)
    if chol_match:
        patient_info["chol"] = int(chol_match.group(1))

    fbs_match = re.search(r"Fasting\s*Blood\s*Sugar\s*>\s*120\s*mg\s*/\s*dl\s*:\s*(\d)", report_text, re.IGNORECASE)
    if fbs_match:
        patient_info["fbs"] = int(fbs_match.group(1))

    restecg_match = re.search(r"Resting\s*Electrocardiographic\s*Results\s*:\s*(\d)", report_text, re.IGNORECASE)
    if restecg_match:
        patient_info["restecg"] = int(restecg_match.group(1))

    thalach_match = re.search(r"Maximum\s*Heart\s*Rate\s*Achieved\s*:\s*(\d+)", report_text, re.IGNORECASE)
    if thalach_match:
        patient_info["thalach"] = int(thalach_match.group(1))

    exang_match = re.search(r"Exercise\s*Induced\s*Angina\s*:\s*(\d)", report_text, re.IGNORECASE)
    if exang_match:
        patient_info["exang"] = int(exang_match.group(1))

    oldpeak_match = re.search(r"ST\s*depression\s*induced\s*by\s*exercise\s*relative\s*to\s*rest\s*:\s*([\d.]+)", report_text, re.IGNORECASE)
    if oldpeak_match:
        patient_info["oldpeak"] = float(oldpeak_match.group(1))

    slope_match = re.search(r"The\s*slope\s*of\s*the\s*peak\s*exercise\s*ST\s*segment\s*:\s*(\d)", report_text, re.IGNORECASE)
    if slope_match:
        patient_info["slope"] = int(slope_match.group(1))

    ca_match = re.search(r"Number\s*of\s*major\s*vessels\s*\(0\s*-\s*3\)\s*colored\s*by\s*flourosopy\s*:\s*(\d)", report_text, re.IGNORECASE)
    if ca_match:
        patient_info["ca"] = int(ca_match.group(1))

    thal_match = re.search(r"Thal\s*:\s*(\d)", report_text, re.IGNORECASE)
    if thal_match:
        patient_info["thal"] = int(thal_match.group(1))

    return patient_info

if __name__ == "__main__":
    sample_report = """
    Patient Name: John Doe
    Age: 45
    Gender: Male
    Chest Pain Type: 1
    Resting Blood Pressure: 130
    Cholesterol: 200
    Fasting Blood Sugar > 120 mg/dl: 0
    Resting Electrocardiographic Results: 0
    Maximum Heart Rate Achieved: 150
    Exercise Induced Angina: 0
    ST depression induced by exercise relative to rest: 1.5
    The slope of the peak exercise ST segment: 2
    Number of major vessels (0-3) colored by flourosopy: 1
    Thal: 3
    """

    extracted_data = extract_patient_info(sample_report)
    print(extracted_data)