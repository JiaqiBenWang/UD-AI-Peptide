import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the best model
model_filename = '16000PIbest_SVM_model.joblib'
best_svm_model = joblib.load(model_filename)

# Load the original One-Hot Encoder and StandardScaler
encoder_filename = '16000PI_SVM_one_hot_encoder.joblib'
scaler_filename = '16000PI_SVM_scaler.joblib'
encoder = joblib.load(encoder_filename)
scaler = joblib.load(scaler_filename)

# Load the new test dataset
test_data_path = r"C:\Users\zzh\Desktop\test_data\test_data_144000.csv"
test_data = pd.read_csv(test_data_path)

# Split the 'pep0' column into v1, v2, v3, v4
test_data[['v1', 'v2', 'v3', 'v4']] = test_data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features
X_test = test_data[['v1', 'v2', 'v3', 'v4']]
y_test = test_data['PI']

# Use the loaded One-Hot Encoder to encode the new test dataset
X_test_encoded = encoder.transform(X_test)

# Use the loaded StandardScaler to standardize the new test dataset
X_test_scaled = scaler.transform(X_test_encoded)

# Make predictions on the new test dataset
test_predictions_new = best_svm_model.predict(X_test_scaled)

# Evaluate the model's performance on the new test dataset
test_rmse_new = np.sqrt(mean_squared_error(y_test, test_predictions_new))
test_mse_new = mean_squared_error(y_test, test_predictions_new)
test_mae_new = mean_absolute_error(y_test, test_predictions_new)
test_r2_new = r2_score(y_test, test_predictions_new)

print(f'Test RMSE on New Dataset (SVM): {test_rmse_new}')
print(f'Test MSE on New Dataset (SVM): {test_mse_new}')
print(f'Test MAE on New Dataset (SVM): {test_mae_new}')
print(f'Test R^2 on New Dataset (SVM): {test_r2_new}')

# Save the new test dataset with predictions
test_data['PI_PRE_SVM'] = test_predictions_new
test_data.to_csv('144000test_data_with_predictions.csv', index=False)
print("The new test dataset with predictions has been saved as test_data_with_predictions.csv")
