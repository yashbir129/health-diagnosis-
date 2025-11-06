import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_heart_disease_model():
    """Loads the Cleveland heart disease dataset, preprocesses it, trains models, and saves them."""
    
    # Define column names
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed.cleveland.data')
    df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')

    # Separate features (X) and target (y)
    X = df.drop(columns=['num'])
    y = df['num']

    # Convert target to binary classification (0 for no disease, 1 for disease)
    y = y.apply(lambda x: 1 if x > 0 else 0)

    # Handle missing values using imputation
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Preprocess the features (scaling)
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    trained_models = {}
    model_performance = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        trained_models[name] = model
        model_performance[name] = {'accuracy': accuracy, 'report': report}

        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:\n{report}")

        # Save the trained model
        model_filename = os.path.join('models', f'{name.lower()}_heart.pkl')
        joblib.dump(model, model_filename)
        print(f"Saved {name} model to {model_filename}")

    # Save the preprocessor (scaler and imputer)
    preprocessor = {
        'imputer': imputer,
        'scaler': scaler
    }
    preprocessor_filename = os.path.join('models', 'preprocessor_heart.pkl')
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Saved preprocessor to {preprocessor_filename}")

    return trained_models, model_performance, preprocessor

if __name__ == '__main__':
    trained_models, performance, preprocessor_obj = train_heart_disease_model()

    if trained_models:
        print("\nModel Training Complete. Performance Summary:")
        for name, perf in performance.items():
            print(f"{name}: Accuracy = {perf['accuracy']:.4f}")