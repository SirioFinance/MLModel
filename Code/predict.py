import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.metrics import accuracy_score, f1_score, log_loss
import logging
import argparse
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    session = rt.InferenceSession(model_path)
    logging.info("ONNX model loaded.")
    return session

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    logging.info("Scaler loaded.")
    return scaler

def preprocess_data(data, scaler):
    # Drop unnecessary columns
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    # Process date column
    data['dt_opened'] = pd.to_datetime(data['dt_opened'], errors='coerce', dayfirst=True)

    # Extract date features
    data['year'] = data['dt_opened'].dt.year
    data['month'] = data['dt_opened'].dt.month
    data['day'] = data['dt_opened'].dt.day
    data['hour'] = data['dt_opened'].dt.hour

    # Drop the original 'dt_opened' column
    data = data.drop(columns=['dt_opened'])

    # Select numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns

    # Ensure the order of numerical columns matches the scaler
    scaler_columns = scaler.feature_names_in_
    # Filter the columns to only those that the scaler has seen before
    data_for_scaling = data[numerical_cols].reindex(columns=scaler_columns, fill_value=0)

    # Transform the data
    data_for_scaling = scaler.transform(data_for_scaling)

    # Create a new DataFrame to keep track of scaled values
    scaled_data = pd.DataFrame(data_for_scaling, columns=scaler_columns)

    return scaled_data

def make_prediction(session, input_data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Convert input data to NumPy array with the correct data type
    input_data = input_data.to_numpy().astype(np.float32)
    
    # Run the model prediction
    predictions = session.run([output_name], {input_name: input_data})[0]
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained ONNX model')
    parser.add_argument('model_path', type=str, help='Path to the trained ONNX model file')
    parser.add_argument('data_path', type=str, help='Path to the input data file for prediction')
    parser.add_argument('scaler_path', type=str, help='Path to the scaler file')
    args = parser.parse_args()

    # Load model and scaler
    session = load_model(args.model_path)
    scaler = load_scaler(args.scaler_path)

    # Load and preprocess input data
    input_data = pd.read_csv(args.data_path)
    input_data_processed = preprocess_data(input_data, scaler)

    # Log the shape of input_data_processed
    logging.info(f"Processed input data shape: {input_data_processed.shape}")

    # Make predictions
    predictions = make_prediction(session, input_data_processed)

    # If the dataset has true labels, calculate and log the metrics
    if 'target' in input_data.columns:
        true_labels = input_data['target']  # Replace 'target' with actual column name if different
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        logloss = log_loss(true_labels, predictions)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Log Loss: {logloss:.4f}")
    else:
        logging.warning("True labels not found in input data. Skipping metrics calculation.")

    # Print predictions
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
