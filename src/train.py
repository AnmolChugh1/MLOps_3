import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

os.rename("models/20250726_205151_test_sample.joblib", "models/test_data.joblib")
os.rename("models/20250726_205151_regression_model.joblib", "models/regression_model.joblib")



MODEL_DIR = "models"
MODEL_FILE = "regression_model.joblib"
TEST_FILE = "test_sample.joblib"

RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def fetch_data():

    try:
        logging.info("Loading California Housing data...")
        dataset = fetch_california_housing()
        X, y = dataset.data, dataset.target
        logging.info(f"Data shape: {X.shape}, Target shape: {y.shape}")
        logging.info(f"Features: {dataset.feature_names}")
        return X, y
    except Exception as error:
        logging.error(f"Error fetching dataset: {error}")
        raise


def train_regression_model(X, y):

    try:
        logging.info("Splitting dataset into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
        )

        logging.info("Training linear regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

  
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        logging.info(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logging.info(f"Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        return model, (X_test, y_test)
    except Exception as error:
        logging.error(f"Training failed: {error}")
        raise


def persist_model_and_data(model, test_data):
  
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(MODEL_DIR, f"{timestamp}_{MODEL_FILE}")
        test_data_path = os.path.join(MODEL_DIR, f"{timestamp}_{TEST_FILE}")

        joblib.dump(model, model_path)
        joblib.dump(test_data, test_data_path)

        logging.info(f"Model saved at: {model_path}")
        logging.info(f"Test data saved at: {test_data_path}")

        logging.debug(f"Model intercept: {model.intercept_}")
        logging.debug(f"Model coefficients: {model.coef_}")
    except Exception as error:
        logging.error(f"Failed to save outputs: {error}")
        raise


def run_pipeline():
  
    logging.info("Initiating model training pipeline...")
    X, y = fetch_data()
    model, test_data = train_regression_model(X, y)
    persist_model_and_data(model, test_data)
    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
