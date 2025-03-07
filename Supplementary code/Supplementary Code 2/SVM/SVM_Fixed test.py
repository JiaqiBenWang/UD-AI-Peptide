import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Define file paths and range
test_data_path = r"C:\Users\zzh\Desktop\test_data\test_data_10000.csv"
model_base_path = r"C:\Users\zzh\Desktop\code\ZZH"

# Load the new test dataset
test_data = pd.read_csv(test_data_path)

# Split the 'pep0' column into v1, v2, v3, v4
test_data[['v1', 'v2', 'v3', 'v4']] = test_data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features
X_test = test_data[['v1', 'v2', 'v3', 'v4']]
y_test = test_data['PI']

# Loop through each trained model for prediction
results = []
for i in range(1000, 16001, 500):
    # Construct file paths related to the model
    model_filename = os.path.join(model_base_path, f"{i}PIbest_SVM_model.joblib")
    encoder_filename = os.path.join(model_base_path, f"{i}PI_SVM_one_hot_encoder.joblib")
    scaler_filename = os.path.join(model_base_path, f"{i}PI_SVM_scaler.joblib")

    # Check if files exist
    if not (os.path.exists(model_filename) and os.path.exists(encoder_filename) and os.path.exists(scaler_filename)):
        print(f"Model file {model_filename}, encoder file {encoder_filename}, or scaler file {scaler_filename} does not exist, skipping.")
        continue

    # Load the model, encoder, and scaler
    best_svm_model = joblib.load(model_filename)
    encoder = joblib.load(encoder_filename)
    scaler = joblib.load(scaler_filename)

    # Use the loaded One-Hot Encoder to encode the new test dataset
    X_test_encoded = encoder.transform(X_test)

    # Use the loaded StandardScaler to standardize the new test dataset
    X_test_scaled = scaler.transform(X_test_encoded)

    # Make predictions on the new test dataset
    test_predictions = best_svm_model.predict(X_test_scaled)

    # Evaluate the model's performance on the new test dataset
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Save evaluation results to the list
    results.append({
        'Model': f"{i} samples",
        'Test RMSE': test_rmse,
        'Test MSE': test_mse,
        'Test MAE': test_mae,
        'Test R^2': test_r2
    })

    # Save the new test dataset with predictions
    test_data[f'PI_PRE_SVM_{i}'] = test_predictions

# Save evaluation results as a CSV file
results_df = pd.DataFrame(results)
results_output_path = r"C:\Users\zzh\Desktop\Fixed test result\PI_SVM_model_evaluation_results.csv"
results_df.to_csv(results_output_path, index=False)
print(f"Model evaluation results have been saved as {results_output_path}")

# Save the new test dataset with all predictions
final_predictions_output_path = r"C:\Users\zzh\Desktop\Fixed test result\10000PI_SVM_test_data_with_all_predictions.csv"
test_data.to_csv(final_predictions_output_path, index=False)
print(f"The new test dataset with all predictions has been saved as {final_predictions_output_path}")
