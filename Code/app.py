import pandas as pd
import numpy as np
import os
import logging
import streamlit as st
from sklearn.preprocessing import StandardScaler
import onnxruntime as rt
import pickle
from PIL import Image
import base64
from io import BytesIO
from sklearn.metrics import log_loss, classification_report, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskPredictor:
    def __init__(self, model_path='xgboost_model.onnx', scaler_path='scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.session = rt.InferenceSession(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.columns = [
            'dt_opened', 'block_opened', 'position_days', 'hour_opened', 'day_opened',
            'month_opened', 'weth_borrow_balance', 'weth_collateral_balance',
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

    def load_data(self, uploaded_file):
        """Load data from uploaded file (CSV or NumPy)"""
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.npy'):
                file_bytes = uploaded_file.read()
                data = np.load(BytesIO(file_bytes), allow_pickle=True)
                if data.shape[1] == len(self.columns):
                    return pd.DataFrame(data, columns=self.columns)
                elif data.shape[1] == len(self.columns) + 1:
                    return pd.DataFrame(data[:, 1:], columns=self.columns)
                else:
                    raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
            elif uploaded_file.name.endswith('.npz'):
                file_bytes = uploaded_file.read()
                data = np.load(BytesIO(file_bytes), allow_pickle=True)['arr_0']
                return pd.DataFrame(data, columns=self.columns)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def preprocess_data(self, data):
        """Prepare the data for prediction"""
        # Rename the first column to "loan_id"
        data = data.rename(columns={data.columns[0]: "loan_id"})
        
        # Get the scaler's feature names
        scaler_features = self.scaler.feature_names_in_
        
        # Add missing features
        missing_features = [feat for feat in scaler_features if feat not in data.columns]
        for feat in missing_features:
            data[feat] = 0
        
        # Select and scale features
        data_to_scale = data[scaler_features]
        data_scaled = self.scaler.transform(data_to_scale)
        
        # Extract loan IDs
        loan_ids = data['loan_id'].reset_index(drop=True)
        return data_scaled, loan_ids

    def predict(self, input_data):
        """Make predictions on preprocessed data"""
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        
        # Look for probability output
        prob_output_name = output_names[0]
        for name in output_names:
            if 'probability' in name.lower() or 'prob' in name.lower():
                prob_output_name = name
                break
        
        # Run prediction
        predictions = self.session.run([prob_output_name], 
                                     {input_name: input_data.astype(np.float32)})[0]
        
        # Handle different output formats
        if predictions.ndim > 1 and predictions.shape[1] == 2:
            probabilities = predictions[:, 1]  # Take probability of class 1
        else:
            probabilities = predictions
        
        return probabilities

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    st.set_page_config(page_title="Liquidity Risk Predictor", layout="wide")
    
    # Create a container for the header
    header_container = st.container()
    with header_container:
        col1, col2 = st.columns([1, 4])
        with col1:
            # Load and display the image
            image = Image.open('/Users/andrewcosta/Desktop/MLModel-model-development-copy/sirio-finance.jpeg')
            st.image(image, width=150)
        with col2:
            st.title("Sirio Finance - Liquidity Risk Prediction Tool")
    
    st.write("""
    Upload a CSV or NumPy file containing loan data to predict liquidity risk probabilities.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'npy', 'npz'])
    
    if uploaded_file is not None:
        try:
            # Initialize predictor
            predictor = RiskPredictor()
            
            # Load and process data
            data = predictor.load_data(uploaded_file)
            if data is not None:
                st.write("### Data Preview")
                st.dataframe(data.head())
                
                # Make predictions
                data_scaled, loan_ids = predictor.preprocess_data(data)
                probabilities = predictor.predict(data_scaled)
                
                # Format results
                results_df = pd.DataFrame({
                    'loan_id': loan_ids,
                    'liquidity_risk_probability': ['{:.6f}'.format(float(p)) for p in probabilities]
                })
                
                # Display results
                st.write("### Prediction Results")
                st.dataframe(results_df)
                
                # Create download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="liquidity_risk_predictions.csv",
                    mime="text/csv"
                )
                
                # Display some statistics
                st.write("### Risk Statistics")
                risk_probs = results_df['liquidity_risk_probability'].astype(float)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Risk", f"{risk_probs.mean():.6f}")
                with col2:
                    st.metric("Max Risk", f"{risk_probs.max():.6f}")
                with col3:
                    st.metric("Min Risk", f"{risk_probs.min():.6f}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()