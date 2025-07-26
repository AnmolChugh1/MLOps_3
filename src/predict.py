import numpy as np
import joblib
import os
import logging
import sys
from sklearn.metrics import r2_score, mean_squared_error


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


MODEL_DIRECTORY = "models"
MODEL_FILENAME = "linear_model.joblib"
TEST_DATA_FILENAME = "test_data.joblib"
MINIMUM_R2 = 0.5


def load_resources():
 
    model_path = os.path.join(MODEL_DIRECTORY, MODEL_FILENAME)
    test_data_path = os.path.join(MODEL_DIRECTORY, TEST_DATA_FILENAME)

    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Loading test data from: {test_data_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at: {test_data_path}")

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    logging.info("Model and test data successfully loaded.")
    logging.info(f"Test features shape: {X_test.shape}, Target shape: {y_test.shape}")
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
 
    logging.info("Generating predictions on test data...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"Root Mean Squared Error: {rmse:.4f}")

    logging.info("\nSample predictions (first 5 rows):")
    logging.info("Actual\tPredicted")
    for i in range(min(5, len(y_test))):
        logging.info(f"{y_test[i]:.2f}\t{y_pred[i]:.2f}")

    return r2, rmse


def verify_container():

    logging.info("Initiating container prediction check...")

    try:
        model, X_test, y_test = load_resources()
        r2, _ = evaluate_model(model, X_test, y_test)

        if r2 >= MINIMUM_R2:
            logging.info(f"\n✅ Container verification PASSED (R² = {r2:.4f})")
            sys.exit(0)
        else:
            logging.error(f"\n❌ Container verification FAILED (R² = {r2:.4f} < {MINIMUM_R2})")
            sys.exit(1)

    except Exception as error:
        logging.exception(f"\n❌ Verification failed due to an error: {str(error)}")
        sys.exit(1)


if __name__ == "__main__":
    verify_container()
