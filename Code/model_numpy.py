import pandas as pd
import numpy as np
import os
import logging
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnx import save_model
import pickle
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskModel:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.model_file = 'xgboost_model.onnx'  # Model file in ONNX format
        self.scaler_file = 'scaler.pkl'  # Scaler file path
        self.predictions_file = 'predictions.csv'  # Predictions file path
        self.feature_names = None
        self.feature_mapping = None
        
        # Define the expected columns
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
            'borrow_collateral_ratio', 'liquidation_event_happened'
        ]

    def load_and_prepare_data(self):
        """
        Load and prepare the data
        """
        logging.info("Loading dataset...")
        
        # Load the data first
        self.data = self.load_data(self.file_path)
        
        if isinstance(self.data, np.ndarray):
            # Check if we have an extra column (likely the index column)
            if self.data.shape[1] == 35:
                logging.info("Data has 35 columns, removing first column assuming it's the index")
                data_without_index = self.data[:, 1:]  # Remove first column
                self.df = pd.DataFrame(data_without_index, columns=self.columns)
                # Add the index column back as loan_id
                self.df['loan_id'] = self.data[:, 0]  # Use the first column as loan_id
            else:
                self.df = pd.DataFrame(self.data, columns=self.columns)
                # Add loan_id if not present
                if 'loan_id' not in self.df.columns:
                    self.df['loan_id'] = range(len(self.df))

        if self.df is None:
            raise ValueError("Failed to load data into DataFrame")

        logging.info("Preparing data...")
        
        # Convert liquidation_event_happened to int
        self.df['liquidation_event_happened'] = self.df['liquidation_event_happened'].astype(int)
        
        # Handle date conversion more robustly
        try:
            self.df['dt_opened'] = pd.to_datetime(self.df['dt_opened'], 
                                                format='%d-%m-%Y', 
                                                errors='coerce',
                                                dayfirst=True)
            default_date = pd.Timestamp('2022-08-26')
            self.df['dt_opened'] = self.df['dt_opened'].fillna(default_date)
            
        except Exception as e:
            logging.warning(f"Error converting dates: {e}")
            if 'block_opened' in self.df.columns:
                reference_block = 15412448
                reference_date = pd.to_datetime('2022-08-26')
                blocks_since_reference = self.df['block_opened'] - reference_block
                seconds_since_reference = blocks_since_reference * 12
                self.df['dt_opened'] = reference_date + pd.to_timedelta(seconds_since_reference, unit='s')
            else:
                self.df['dt_opened'] = pd.to_datetime('2022-08-26')
        
        self.df['dt_opened'] = self.df['dt_opened'].dt.strftime('%d-%m-%Y')

        # Convert all numeric columns to float
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        logging.info(f"Data shape after preparation: {self.df.shape}")

    def split_data(self):
        logging.info("Splitting data...")
        # Drop the target variable and keep loan_id
        feature_cols = [col for col in self.df.columns if col not in ['liquidation_event_happened']]
        X = self.df[feature_cols]
        y = self.df['liquidation_event_happened']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"Training set shape: {self.X_train.shape}")
        logging.info(f"Test set shape: {self.X_test.shape}")
        logging.info(f"Training set class distribution:\n{self.y_train.value_counts()}")
        logging.info(f"Test set class distribution:\n{self.y_test.value_counts()}")

    def oversample_minority_class(self):
        logging.info("Oversampling minority class...")
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
        
        # Get feature columns (excluding loan_id)
        feature_cols = [col for col in self.X_train.columns if col != 'loan_id']
        
        # Store loan_ids
        loan_ids = self.X_train['loan_id']
        
        # Oversample features
        X_resampled, y_resampled = oversampler.fit_resample(
            self.X_train[feature_cols], 
            self.y_train
        )
        
        # Create new DataFrame with oversampled data
        self.X_train = pd.DataFrame(X_resampled, columns=feature_cols)
        self.y_train = pd.Series(y_resampled)
        
        # Generate new loan_ids for synthetic samples
        new_loan_ids = range(len(loan_ids), len(self.X_train))
        self.X_train['loan_id'] = list(loan_ids) + list(new_loan_ids)

        logging.info(f"After oversampling - X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        logging.info(f"Class distribution after oversampling:\n{pd.Series(y_resampled).value_counts()}")

    def extract_date_features(self, X):
        logging.info("Extracting date features...")
        X = X.copy()
        
        # Convert dt_opened to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(X['dt_opened']):
            X['dt_opened'] = pd.to_datetime(X['dt_opened'], format='%d-%m-%Y', dayfirst=True)
        
        # Extract date features
        X['year'] = X['dt_opened'].dt.year
        X['month'] = X['dt_opened'].dt.month
        X['day'] = X['dt_opened'].dt.day
        X['hour'] = X['dt_opened'].dt.hour
        
        # Drop the original datetime column
        X = X.drop('dt_opened', axis=1)
        
        return X

    def preprocess_data(self):
        logging.info("Preprocessing data...")
        
        # First extract date features
        self.X_train = self.extract_date_features(self.X_train)
        self.X_test = self.extract_date_features(self.X_test)
        
        # Rename column with hyphen to use underscore
        self.X_train = self.X_train.rename(columns={'borrow-collateral-ratio': 'borrow_collateral_ratio'})
        self.X_test = self.X_test.rename(columns={'borrow-collateral-ratio': 'borrow_collateral_ratio'})
        
        # Convert all columns to numeric except loan_id
        for col in self.X_train.columns:
            if col != 'loan_id':
                self.X_train[col] = pd.to_numeric(self.X_train[col], errors='coerce')
                self.X_test[col] = pd.to_numeric(self.X_test[col], errors='coerce')
        
        # Fill any NaN values that might have been created
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # Get only numerical columns for scaling, excluding loan_id
        numerical_cols = [col for col in self.X_train.columns 
                         if col != 'loan_id']
        
        # Store feature names for later use
        self.feature_names = numerical_cols
        
        # Scale only numerical features
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        # Fit scaler on training data and transform both train and test
        self.X_train_scaled[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test_scaled[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        logging.info(f"Training data shape after preprocessing: {self.X_train_scaled.shape}")
        logging.info(f"Test data shape after preprocessing: {self.X_test_scaled.shape}")
        logging.info(f"Feature dtypes:\n{self.X_train_scaled.dtypes}")

    def train_xgboost_model(self):
        logging.info("Training XGBoost model...")
        
        # Get numerical columns in the same order, excluding loan_id
        numerical_cols = [col for col in self.X_train_scaled.columns if col != 'loan_id']
        
        # Create feature name mapping
        feature_names = [f'f{i}' for i in range(len(numerical_cols))]
        self.feature_mapping = dict(zip(numerical_cols, feature_names))
        
        # Rename columns to f0, f1, etc.
        X_train_renamed = self.X_train_scaled[numerical_cols].rename(columns=self.feature_mapping)
        
        self.xgb_model = xgb.XGBClassifier(
            subsample=0.8,
            min_child_weight=3,
            reg_lambda=1,
            reg_alpha=0,
            n_estimators=100,
            max_depth=9,
            learning_rate=0.3,
            colsample_bytree=0.6,
            gamma=0,
            random_state=42
        )
        
        # Train with renamed features
        self.xgb_model.fit(X_train_renamed, self.y_train)

        # Define the input type for ONNX conversion
        initial_types = [('input', FloatTensorType([None, X_train_renamed.shape[1]]))]
        
        # Convert XGBoost model to ONNX
        logging.info("Converting XGBoost model to ONNX...")
        onnx_model = convert_xgboost(self.xgb_model, initial_types=initial_types)
        
        # Save the ONNX model
        save_model(onnx_model, self.model_file)
        logging.info(f"Model saved to {self.model_file}")
        
        # Save the scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        logging.info(f"Scaler saved to {self.scaler_file}")
        
        # Save feature mapping for reference
        with open('feature_mapping.json', 'w') as f:
            json.dump(self.feature_mapping, f, indent=2)
        logging.info("Feature mapping saved to feature_mapping.json")

    def evaluate_model(self):
        logging.info("Evaluating model...")
        
        # Use the same numerical columns as in training
        numerical_cols = list(self.feature_mapping.keys())
        
        # Rename columns using the same mapping as training
        X_test_renamed = self.X_test_scaled[numerical_cols].rename(columns=self.feature_mapping)
        
        # Make predictions using renamed features
        y_pred_probs_xgb = self.xgb_model.predict_proba(X_test_renamed)[:, 1]
        
        # Log the probabilities
        formatted_probs = ['{:.6f}'.format(prob) for prob in y_pred_probs_xgb]
        #logging.info(f'Predicted Probabilities of Liquidity Risk: {formatted_probs}')
        
        # Create a DataFrame with the loan_id and its corresponding predicted probability
        results_df = pd.DataFrame({
            'loan_id': self.X_test['loan_id'],
            'liquidity_risk_probability': formatted_probs
        })
        
        # Save this DataFrame to a CSV file
        results_df.to_csv('liquidity_risk_probabilities.csv', index=False)
        logging.info("Liquidity risk probabilities and loan IDs saved to liquidity_risk_probabilities.csv")

        log_loss_xgb = log_loss(self.y_test, y_pred_probs_xgb)
        logging.info(f'X Gradient Boosting Log Loss: {log_loss_xgb:.4f}')

        # Additional evaluations
        y_pred = self.xgb_model.predict(X_test_renamed)
        logging.info("Classification Report:")
        logging.info(f"\n{classification_report(self.y_test, y_pred)}")

        auc = roc_auc_score(self.y_test, y_pred_probs_xgb)
        logging.info(f"ROC AUC Score: {auc:.4f}")

    def predict(self, new_data):
        logging.info("Predicting with new data...")
        if isinstance(new_data, np.ndarray):
            new_data = pd.DataFrame(new_data, columns=self.columns)
        
        new_data['dt_opened'] = pd.to_datetime(new_data['dt_opened'], format='%d-%m-%Y', dayfirst=True)
        new_data = self.extract_date_features(new_data)

        exclude_cols = ['loan_id', 'dt_opened', 'liquidation_event_happened']
        numerical_cols = new_data.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        new_data_scaled = self.scaler.transform(new_data[numerical_cols])

        session = rt.InferenceSession(self.model_file)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        predictions = session.run([output_name], {input_name: new_data_scaled.astype(np.float32)})[0]

        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_df.to_csv(self.predictions_file, index=False)
        logging.info(f"Predictions saved to {self.predictions_file}")

        return predictions_df

    def load_data(self, file_path):
        """
        Load data from either CSV or numpy file
        """
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_numpy()
        elif file_path.endswith('.npy'):
            try:
                # Load numpy file
                return np.load(file_path, allow_pickle=True)
            except Exception as e:
                logging.warning(f"Failed to load .npy file directly: {e}")
                try:
                    # Try reading as text and converting to numpy array
                    return pd.read_csv(file_path, header=0).to_numpy()
                except Exception as e2:
                    logging.error(f"Failed to load file as CSV: {e2}")
                    raise ValueError(f"Could not load file {file_path} as either numpy or CSV format")
        elif file_path.endswith('.npz'):
            return np.load(file_path, allow_pickle=True)['arr_0']
        else:
            raise ValueError("Unsupported file format. Please use .csv, .npy, or .npz")

def main():
    parser = argparse.ArgumentParser(description='Risk Model Training and Evaluation')
    parser.add_argument('file_path', type=str, help='Path to the dataset file')
    parser.add_argument('--predict', type=str, help='Path to new data file for predictions')
    args = parser.parse_args()

    try:
        model = RiskModel(args.file_path)
        model.load_and_prepare_data()
        model.split_data()
        model.oversample_minority_class()
        model.preprocess_data()
        model.train_xgboost_model()
        model.evaluate_model()

        if args.predict:
            predict_data = model.load_data(args.predict)
            predictions = model.predict(predict_data)
            print("Predictions for new data:", predictions)
            
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()