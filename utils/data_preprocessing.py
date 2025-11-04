import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

def preprocess_data(df):
    """Applies preprocessing steps to the dataframe."""
    if df is None:
        return None

    # Separate target variable if it exists (assuming 'Disease' is the target)
    # For now, we'll assume the target is not in the input data for prediction
    # If training, you would separate X and y here.

    # Identify categorical and numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    processed_data = preprocessor.fit_transform(df)

    # Get feature names after one-hot encoding
    # This part can be tricky with ColumnTransformer, especially for prediction
    # For training, you'd typically save the preprocessor and use it for new data
    # For now, we'll return the processed numpy array.
    return processed_data, preprocessor

if __name__ == '__main__':
    # Example usage (assuming health_data.csv exists in the data directory)
    data_filepath = '..\\data\\health_data.csv'
    df = load_data(data_filepath)

    if df is not None:
        print("Original Data Head:")
        print(df.head())

        processed_array, preprocessor_obj = preprocess_data(df)
        print("\nProcessed Data Shape:", processed_array.shape)
        print("\nFirst 5 rows of Processed Data:")
        print(processed_array[:5])

        # You would typically save the preprocessor for later use in prediction
        # import joblib
        # joblib.dump(preprocessor_obj, 'preprocessor.pkl')
