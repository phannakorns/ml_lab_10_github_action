import pandas as pd
import mlflow
from sklearn.datasets import load_iris
from mlflow.tracking import MlflowClient

def load_and_predict():
    """
    Simulates a production scenario by loading a model from a specific
    stage in the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "iris-classifier-prod"
    MODEL_STAGE = "Staging"  # Change to "Production" after transitioning the model stage

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    # Load the model from the Model Registry
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/1")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    # Prepare new sample data for prediction (using the first row of the dataset)
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    sample_data = df.drop(['target'], axis=1).iloc[0:1]

    # Map the numerical target to the species names
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    actual_label_numerical = iris_data.target[0]
    actual_label_name = target_names.get(actual_label_numerical, 'Unknown')

    # Use the loaded model to make a prediction
    # No manual preprocessing is needed because we logged the entire pipeline
    prediction_name = model.predict(sample_data.values)[0]

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data.values[0]}")
    print(f"Actual Label: {actual_label_name}")
    print(f"Predicted Label: {prediction_name}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()