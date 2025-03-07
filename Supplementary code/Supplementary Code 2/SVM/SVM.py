from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
import pandas as pd
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
data = pd.read_csv(r"C:\Users\zzh\Desktop\data0\data_tetrapeptides_1000.csv")

# Split the 'pep0' column into v1, v2, v3, v4
data[['v1', 'v2', 'v3', 'v4']] = data['pep0'].apply(lambda x: pd.Series(list(x)))

# Extract features and target variable
X = data[['v1', 'v2', 'v3', 'v4']]
y = data['AP']

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split the dataset (80% training set, 20% validation set)
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

# Standardize the data (z-score standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the SVR model
svm = SVR()

# Define the hyperparameter space for RandomizedSearchCV
param_distributions_svm = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': uniform(0.001, 100),  # Adjust the C parameter within a reasonable range
    'degree': randint(1, 5),
    'gamma': ['scale', 'auto'],
}

# Use RandomizedSearchCV for hyperparameter optimization
random_search_svm = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_distributions_svm,
    n_iter=50,  # Increase the number of iterations for RandomizedSearchCV
    scoring='neg_root_mean_squared_error',
    cv=5,  # Increase the number of cross-validation folds
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

# Perform RandomizedSearchCV for hyperparameter optimization on the training set
random_search_svm.fit(X_train_scaled, y_train)

# Output the best parameters and score
print("Best Parameters:", random_search_svm.best_params_)
print("Best Score:", random_search_svm.best_score_)

# Evaluate the best model on the validation set
best_svm_model = random_search_svm.best_estimator_
y_val_pred = best_svm_model.predict(X_val_scaled)

# Calculate validation metrics
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
print(f"Validation MSE: {mse_val}")
print(f"Validation MAE: {mae_val}")
print(f"Validation R2 Score: {r2_val}")
print(f"Validation RMSE: {rmse_val}")

# Predictions on the training set and calculate training metrics
y_train_pred = best_svm_model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
print(f"Train MSE: {mse_train}")
print(f"Train MAE: {mae_train}")
print(f"Train R2 Score: {r2_train}")
print(f"Train RMSE: {rmse_train}")

# Save the best model
model_filename = '1000APbest_SVM_model.joblib'
joblib.dump(best_svm_model, model_filename)
print(f"The best model has been saved as {model_filename}")

# Save the encoder and scaler
joblib.dump(encoder, '1000AP_SVM_one_hot_encoder.joblib')
joblib.dump(scaler, '1000AP_SVM_scaler.joblib')
