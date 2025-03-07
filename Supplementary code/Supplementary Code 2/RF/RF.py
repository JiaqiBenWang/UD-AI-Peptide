import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
import os
import tempfile

# Ensure the directory exists
temp_dir = "C:/Temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.mkdtemp(dir=temp_dir)

# Read data
data = pd.read_csv(r"C:\Users\zzh\Desktop\data0\data_tetrapeptides_16000.csv")

# Split the 'pep0' column into v1, v2, v3, v4
data[['v1', 'v2', 'v3', 'v4']] = data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features and target variable
X = data[['v1', 'v2', 'v3', 'v4']]
y = data['logP']

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split the dataset (80% training set, 20% validation set)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_encoded, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    X_encoded, y, data.index, test_size=0.2, random_state=42)

# Mark dataset divisions
data.loc[train_idx, 'subset'] = 'train'
data.loc[val_idx, 'subset'] = 'test'

# Define the hyperparameter search space
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'bootstrap': [True, False],
    # 'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.02]
}

# Use random search for hyperparameter optimization, specifying RMSE as the scoring metric
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',  # Use RMSE as the scoring metric
    verbose=1,
    n_jobs=-1,
    random_state=42)

random_search.fit(X_train, y_train)

# Output the best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Evaluate the best model on the validation set
best_RF_model = random_search.best_estimator_
y_val_pred = best_RF_model.predict(X_val)

# Calculate validation metrics
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
print(f"Validation MSE: {mse_val}")
print(f"Validation MAE: {mae_val}")
print(f"Validation R2 Score: {r2_val}")
print(f"Validation RMSE: {rmse_val}")

# Predictions on the training set
y_train_pred = best_RF_model.predict(X_train)

# Calculate training metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
print(f"Train MSE: {mse_train}")
print(f"Train MAE: {mae_train}")
print(f"Train R2 Score: {r2_train}")
print(f"Train RMSE: {rmse_train}")

# Predict on the entire dataset and save the predictions to 'logP_PRE_RF' column
data['logP_PRE_RF'] = best_RF_model.predict(encoder.transform(data[['v1', 'v2', 'v3', 'v4']]))

# Save the new dataset with predictions and dataset division labels
data.to_csv('16000data_with_predictions_and_subsets.csv', index=False)
print("New dataset with predictions and dataset division labels has been saved as data_with_predictions_and_subsets.csv")

# Save the best model
model_filename = '16000logPbest_random_forest_model.joblib'
joblib.dump(best_RF_model, model_filename)
print(f"The best model has been saved as {model_filename}")

# Save the encoder and standardizer
joblib.dump(encoder, '16000logP_RF_one_hot_encoder.joblib')
