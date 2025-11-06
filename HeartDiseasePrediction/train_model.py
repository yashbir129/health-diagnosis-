import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model(dataset_path, models_dir):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Define features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing: Scale numerical features and encode labels
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the trained model and preprocessor
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'randomforestclassifier_disease.pkl'))
    joblib.dump({'scaler': scaler}, os.path.join(models_dir, 'preprocessor_disease.pkl'))

    print("Model and preprocessor trained and saved successfully.")

if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    train_and_save_model(dataset_path, models_dir)