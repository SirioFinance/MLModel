import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import imblearn
import xgboost as xgb
import logging
import argparse
import os
import pickle
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskModel:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.model_file = 'xgboost_model.pkl'
    
    def load_and_prepare_data(self):
        if not self.file_path or not os.path.exists(self.file_path):
            logging.error("File path must be provided and the file must exist.")
            raise ValueError("File path must be provided and the file must exist.")

        logging.info("Loading dataset...")
        self.df = pd.read_csv(self.file_path, parse_dates=True)

        logging.info("Preparing data...")
        self.df = self.df.rename(columns={self.df.columns[0]: "loan_id"})
        self.df['dt_opened'] = pd.to_datetime(self.df['dt_opened'], format='%d-%m-%Y')
        self.df['liquidation_event_happened'] = self.df['liquidation_event_happened'].astype(int)

    def split_data(self):
        logging.info("Splitting data...")
        X = self.df.drop(['liquidation_event_happened'], axis=1)
        y = self.df['liquidation_event_happened']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def oversample_minority_class(self):
        logging.info("Oversampling minority class...")
        oversampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
        self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)

    def extract_date_features(self, X):
        logging.info("Extracting date features...")
        X['year'] = X['dt_opened'].dt.year
        X['month'] = X['dt_opened'].dt.month
        X['day'] = X['dt_opened'].dt.day
        X['hour'] = X['dt_opened'].dt.hour
        return X
 
    def preprocess_data(self):
        logging.info("Preprocessing data...")
        self.X_train = self.extract_date_features(self.X_train)
        self.X_test = self.extract_date_features(self.X_test)

        numerical_cols = self.X_train.select_dtypes(include=['number']).columns
        self.X_train_scaled = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test_scaled = self.scaler.transform(self.X_test[numerical_cols])

    def train_xgboost_model(self):
        logging.info("Training XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(
                subsample= 0.8,
                min_child_weight= 3,
                reg_lambda= 1,
                reg_alpha= 0,
                n_estimators=100,
                max_depth=9,
                learning_rate=0.3,
                colsample_bytree= 0.6,
                gamma= 0,
                random_state=42
            )
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        # Save the trained model
        with open(self.model_file, 'wb') as file:
            pickle.dump(self.xgb_model, file)
        logging.info(f"Model saved to {self.model_file}")

    def evaluate_model(self):
        logging.info("Evaluating model...")
        y_pred_probs_xgb = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
        log_loss_xgb = log_loss(self.y_test, y_pred_probs_xgb)
        logging.info(f'X Gradient Boosting Log Loss: {log_loss_xgb:.4f}')

    def predict(self, new_data_path):
        logging.info("Predicting with new data...")
        new_data = pd.read_csv(new_data_path, parse_dates=True)
        new_data = self.extract_date_features(new_data)

        numerical_cols = new_data.select_dtypes(include=['number']).columns
        new_data_scaled = self.scaler.transform(new_data[numerical_cols])
        
        # Load the model
        with open(self.model_file, 'rb') as file:
            loaded_model = pickle.load(file)
        
        predictions = loaded_model.predict(new_data_scaled)
        return predictions


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Risk Model Training and Evaluation')
    parser.add_argument('file_path', type=str, help='Path to the dataset file')
    parser.add_argument('--predict', type=str, help='Path to new data file for predictions')
    args = parser.parse_args()

    model = RiskModel(file_path=args.file_path)
    try:
        model.load_and_prepare_data()
        model.split_data()
        model.oversample_minority_class()
        model.preprocess_data()
        model.train_xgboost_model()
        model.evaluate_model()

        if args.predict:
            predictions = model.predict(args.predict)
            print("Predictions for new data:", predictions)
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()


