import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Define file paths and ranges
test_data_path = r"C:\Users\zzh\Desktop\test_data\test_data_10000.csv"
model_base_path = r"C:\Users\zzh\Desktop\code\ZZH"

# Load new test dataset
test_data = pd.read_csv(test_data_path)

# Split the 'pep0' column into v1, v2, v3, v4
test_data[['v1', 'v2', 'v3', 'v4']] = test_data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features
X_test = test_data[['v1', 'v2', 'v3', 'v4']]
y_test = test_data['PI']

# Loop through models from 1000 to 16000 in steps of 500 for predictions
results = []
for i in range(1000, 16001, 500):
    # Build model-related file paths
    model_filename = os.path.join(model_base_path, f"{i}PIbest_random_forest_model.joblib")
    encoder_filename = os.path.join(model_base_path, f"{i}PI_RF_one_hot_encoder.joblib")

    # Check if files exist
    if not (os.path.exists(model_filename) and os.path.exists(encoder_filename)):
        print(f"Model file {model_filename}, encoder file {encoder_filename} do not exist, skipping.")
        continue

    # Load model and encoder
    best_rf_model = joblib.load(model_filename)
    encoder = joblib.load(encoder_filename)

    # Use the loaded One-Hot Encoder to encode the new test dataset
    X_test_encoded = encoder.transform(X_test)

    # Make predictions on the new test dataset
    test_predictions_new = best_rf_model.predict(X_test_encoded)

    # Evaluate the model's performance on the new test dataset
    test_rmse_new = np.sqrt(mean_squared_error(y_test, test_predictions_new))
    test_mse_new = mean_squared_error(y_test, test_predictions_new)
    test_mae_new = mean_absolute_error(y_test, test_predictions_new)
    test_r2_new = r2_score(y_test, test_predictions_new)

    # Save evaluation results
    results.append({
        'Model': f'{i} samples',
        'Test RMSE': test_rmse_new,
        'Test MSE': test_mse_new,
        'Test MAE': test_mae_new,
        'Test R^2': test_r2_new
    })

    # Save predictions to the test dataset
    test_data[f'PI_PRE_RF_{i}'] = test_predictions_new

# Save evaluation results to a CSV file
results_df = pd.DataFrame(results)
results_output_path = r"C:\Users\朱智慧\zzh\Fixed test result\PI_RF_model_evaluation_results.csv"
results_df.to_csv(results_output_path, index=False)
print(f"Model evaluation results have been saved as {results_output_path}")

# Save the new test dataset with all predictions
final_predictions_output_path = r"C:\Users\zzh\Desktop\Fixed test result\10000PI_RF_test_data_with_all_predictions.csv"
test_data.to_csv(final_predictions_output_path, index=False)
print(f"The new test dataset with all predictions has been saved as {final_predictions_output_path}")
