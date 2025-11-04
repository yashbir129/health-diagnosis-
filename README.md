# Health Diagnosis AI/ML System

## Project Overview
This project develops a Health Diagnosis AI/ML System designed to predict potential diseases based on a patient's medical history and current conditions. Beyond prediction, the system will also suggest appropriate doctors and provide precautionary measures.

## Features
- **Disease Prediction**: Utilizes Machine Learning models to predict diseases from patient data.
- **Doctor Recommendation**: Suggests specialized doctors based on predicted diseases.
- **Precautionary Measures**: Provides advice and precautions for predicted health conditions.
- **User-friendly Interface**: A web-based interface for easy data input and result visualization.

## Technology Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: MySQL
- **Machine Learning**: Scikit-learn, Pandas, NumPy

## Project Structure
```
Health-Diagnosis-AI-ML/
├── app.py                         # Main Flask application entry point
├── requirements.txt               # All Python dependencies
├── README.md                      # Project overview and setup guide
│
├── config/
│   ├── db_config.py               # MySQL database connection setup
│   └── model_config.py            # Model paths, hyperparameters, etc.
│
├── static/                        # Frontend static files
│   ├── css/
│   │   └── style.css              # Website styling (colors, layout)
│   ├── js/
│   │   └── script.js              # (Optional) Client-side interactions
│   └── images/
│       └── logo.svg               # Project logo / visuals
│
├── templates/                     # HTML templates for Flask
│   ├── index.html                 # Home page / User input form
│   ├── result.html                # Display model prediction result
│   ├── about.html                 # About project and team info
│   └── contact.html               # Contact / feedback page
│
├── data/
│   ├── health_data.csv            # Dataset for training
│   └── processed_data.csv         # Cleaned dataset after preprocessing
│
├── models/
│   ├── logistic_model.pkl         # Saved Logistic Regression model
│   ├── decision_tree_model.pkl    # Saved Decision Tree model
│   ├── random_forest_model.pkl    # Saved Random Forest model
│   └── model_trainer.py           # Script to train and save ML models
│
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis (Seaborn + Pandas)
│   └── ModelTesting.ipynb         # Testing accuracy of different ML models
│
└── utils/
    ├── data_preprocessing.py      # Handle missing values, normalization, encoding
    ├── evaluation.py              # Functions for accuracy, confusion matrix, etc.
    └── helper_functions.py        # Reusable helper utilities
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository_url>
cd Health-Diagnosis-AI-ML
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venc\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
- Ensure you have MySQL installed and running.
- Update `config/db_config.py` with your MySQL credentials.
- Create the necessary database and tables (schema to be defined).

### 5. Data Preparation
- Place your raw health data in `data/health_data.csv`.
- Run `utils/data_preprocessing.py` to clean and preprocess the data, saving it to `data/processed_data.csv`.

### 6. Model Training
- Run `models/model_trainer.py` to train and save the ML models (`.pkl` files) in the `models/` directory.

### 7. Run the Application
```bash
python app.py
```

### 8. Access the Application
Open your web browser and navigate to `http://127.0.0.1:5000` (or the address shown in your terminal).

## Usage
1. Input patient medical history and current conditions on the home page.
2. The system will predict potential diseases.
3. View predicted diseases, recommended doctors, and precautionary measures on the results page.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.
