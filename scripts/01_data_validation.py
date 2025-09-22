import pandas as pd
import mlflow

def validate_data(file_path='Iris.csv'):
    """
    Loads the Iris dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    mlflow.set_experiment("Iris Classification - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data from the CSV file
        try:
            df = pd.read_csv(file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return

        # 2. Perform simple validation checks
        num_rows, num_cols = df.shape
        num_classes = df['Species'].nunique()
        missing_values = df.isnull().sum().sum()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        validation_status = "Success"
        if missing_values > 0 or num_classes < 3:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")
        print("Data validation run finished.")

if __name__ == "__main__":
    validate_data()