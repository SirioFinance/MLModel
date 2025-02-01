import pandas as pd
import numpy as np
import onnxruntime as rt
import logging
import argparse
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    providers = ['CPUExecutionProvider']
    session = rt.InferenceSession(model_path, providers=providers)
    logging.info("ONNX model loaded.")
    # Log model input and output names for debugging
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logging.info(f"Model input name: {input_name}")
    logging.info(f"Model output name: {output_name}")
    return session

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    logging.info("Scaler loaded.")
    return scaler

def load_data(data_input):
    """Load data from file path, numpy array, or numpy file"""
    logging.info(f"Loading data of type: {type(data_input)}")
    
    # If input is already a numpy array
    if isinstance(data_input, np.ndarray):
        columns = [
            'loan_id', 'dt_opened', 'block_opened', 'position_days', 'hour_opened', 
            'day_opened', 'month_opened', 'weth_borrow_balance', 'weth_collateral_balance',
            'weth_borrow_balance_share', 'weth_collateral_balance_share',
            'usdc_borrow_balance', 'usdc_collateral_balance',
            'usdc_borrow_balance_share', 'usdc_collateral_balance_share',
            'wbtc_borrow_balance', 'wbtc_collateral_balance',
            'wbtc_borrow_balance_share', 'wbtc_collateral_balance_share',
            'comp_borrow_balance', 'comp_collateral_balance',
            'comp_borrow_balance_share', 'comp_collateral_balance_share',
            'uni_borrow_balance', 'uni_collateral_balance',
            'uni_borrow_balance_share', 'uni_collateral_balance_share',
            'link_borrow_balance', 'link_collateral_balance',
            'link_borrow_balance_share', 'link_collateral_balance_share',
            'collateral_usd_balance', 'borrow_usd_balance',
            'borrow-collateral-ratio', 'liquidation_event_happened'
        ]
        if data_input.shape[1] == len(columns):
            return pd.DataFrame(data_input, columns=columns)
        elif data_input.shape[1] == len(columns) + 1:
            return pd.DataFrame(data_input[:, 1:], columns=columns)
        else:
            raise ValueError(f"Unexpected number of columns in numpy array: {data_input.shape[1]}")
    
    # If input is a file path
    elif isinstance(data_input, str):
        if data_input.endswith('.csv'):
            return pd.read_csv(data_input)
        elif data_input.endswith('.npy'):
            try:
                data = np.load(data_input, allow_pickle=True)
                return load_data(data)  # Recursively handle the loaded numpy array
            except Exception as e:
                logging.error(f"Error loading .npy file: {e}")
                raise
        elif data_input.endswith('.npz'):
            try:
                data = np.load(data_input, allow_pickle=True)['arr_0']
                return load_data(data)  # Recursively handle the loaded numpy array
            except Exception as e:
                logging.error(f"Error loading .npz file: {e}")
                raise
        else:
            raise ValueError("Unsupported file format. Please use .csv, .npy, or .npz")
    else:
        raise ValueError("Input must be either a numpy array or a file path")

def preprocess_data(data, scaler):
    logging.info("Starting data preprocessing...")
    logging.info(f"Input columns: {data.columns.tolist()}")
    
    # Rename the first column to "loan_id"
    data = data.rename(columns={data.columns[0]: "loan_id"})
    logging.info("Renamed the first column to 'loan_id'.")
    
    # Drop unnecessary columns
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
        logging.info("Dropped 'Unnamed: 0' column.")
    
    # Get the scaler's feature names
    scaler_features = scaler.feature_names_in_
    logging.info(f"Required features for scaling: {scaler_features.tolist()}")
    
    # Add missing features
    missing_features = [feat for feat in scaler_features if feat not in data.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        for feat in missing_features:
            data[feat] = 0
    
    # Select and scale features
    data_to_scale = data[scaler_features]
    data_scaled = scaler.transform(data_to_scale)
    logging.info(f"Data scaled successfully. Shape: {data_scaled.shape}")
    
    # Extract loan IDs
    loan_ids = data['loan_id'].reset_index(drop=True)
    return data_scaled, loan_ids

def make_prediction(session, input_data):
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Look for probability output
    prob_output_name = None
    for name in output_names:
        if 'probability' in name.lower() or 'prob' in name.lower():
            prob_output_name = name
            break
    
    if prob_output_name is None:
        prob_output_name = output_names[0]
    
    logging.info(f"Using output name: {prob_output_name}")
    
    # Run prediction
    predictions = session.run([prob_output_name], {input_name: input_data.astype(np.float32)})[0]
    logging.info(f"Raw predictions shape: {predictions.shape}")
    logging.info(f"Raw predictions sample: {predictions[:5]}")
    
    # Handle different output formats
    if predictions.ndim > 1:
        if predictions.shape[1] == 2:
            probabilities = predictions[:, 1]
        else:
            probabilities = predictions[:, 0]
    else:
        probabilities = predictions
    
    logging.info(f"Processed probabilities sample: {probabilities[:5]}")
    return probabilities

def predict_risk(model_path, data_input, scaler_path):
    """Wrapper function to make predictions on either numpy array or file"""
    try:
        # Load model and scaler
        session = load_model(model_path)
        scaler = load_scaler(scaler_path)
        
        # Load and preprocess data
        input_data = load_data(data_input)
        input_data_scaled, loan_ids = preprocess_data(input_data, scaler)
        
        # Make predictions
        probabilities = make_prediction(session, input_data_scaled)
        
        # Format results
        results_df = pd.DataFrame({
            'loan_id': loan_ids,
            'liquidity_risk_probability': ['{:.6f}'.format(float(prob)) for prob in probabilities]
        })
        
        return results_df
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained ONNX model')
    parser.add_argument('model_path', type=str, help='Path to the trained ONNX model file')
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('scaler_path', type=str, help='Path to the scaler file')
    args = parser.parse_args()
    
    try:
        results_df = predict_risk(args.model_path, args.data_path, args.scaler_path)
        output_file = 'liquidity_risk_probabilities.csv'
        results_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()