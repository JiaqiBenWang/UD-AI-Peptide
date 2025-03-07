import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the best model
model_filename = '2500APbest_random_forest_model.joblib'
best_rf_model = joblib.load(model_filename)

# Load the original One-Hot Encoder
encoder_filename = '2500AP_RF_one_hot_encoder.joblib'
encoder = joblib.load(encoder_filename)

# Load the new test dataset
test_data_path = r"C:\Users\zzh\Desktop\test_data\test_data_157500.csv"
test_data = pd.read_csv(test_data_path)

# Split the 'pep0' column into v1, v2, v3, v4
test_data[['v1', 'v2', 'v3', 'v4']] = test_data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features
X_test = test_data[['v1', 'v2', 'v3', 'v4']]
y_test = test_data['AP']

# Use the loaded One-Hot Encoder to encode the new test dataset
X_test_encoded = encoder.transform(X_test)

# Make predictions on the new test dataset
test_predictions_new = best_rf_model.predict(X_test_encoded)

# Evaluate the model's performance on the new test dataset
test_rmse_new = np.sqrt(mean_squared_error(y_test, test_predictions_new))
test_mse_new = mean_squared_error(y_test, test_predictions_new)
test_mae_new = mean_absolute_error(y_test, test_predictions_new)
test_r2_new = r2_score(y_test, test_predictions_new)

print(f'Test RMSE on New Dataset (RF): {test_rmse_new}')
print(f'Test MSE on New Dataset (RF): {test_mse_new}')
print(f'Test MAE on New Dataset (RF): {test_mae_new}')
print(f'Test R^2 on New Dataset (RF): {test_r2_new}')

# Save the new test dataset with predictions
test_data['AP_PRE_RF'] = test_predictions_new
test_data.to_csv('157500test_data_with_predictions.csv', index=False)
print("The new test dataset with predictions has been saved as test_data_with_predictions.csv")
