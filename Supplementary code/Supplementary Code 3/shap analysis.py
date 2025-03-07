import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import datetime
import joblib
import os
import tempfile
import json
import re
import shap
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from adjustText import adjust_text  # Automatically adjust label positions
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
import matplotlib.pyplot as plt

# Ensure directory exists
temp_dir = "C:/Temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.mkdtemp(dir=temp_dir)

# Read data
data_path = r"C:\Users\朱智慧\Desktop\16w\extracted_features.xlsx"
# data_path = r"C:\Users\朱智慧\Desktop\16w\4500extracted_features.xlsx"
# data_path = r"C:\Users\朱智慧\Desktop\16w\1000extracted_features.xlsx"
data = pd.read_excel(data_path)

# Filter features and target variable
X = data.drop(columns=['id', 'AP'])  # Exclude 'pep0', 'pep', 'id', and 'AP' columns from 16w dataset
# X = data.drop(columns=['AP'])  # Exclude 'pep0', 'pep', 'id', and 'AP' columns from 4.5k dataset
y = data['AP'].apply(lambda x: 0 if x < 1.5 else (1 if 1.5 <= x < 2 else 2))  # Multiclass labels

# Save 'pep0' and 'pep' columns
pep_columns = data[['pep0', 'pep']]

# Remove 'pep0' and 'pep' columns from X for model training
X = X.drop(columns=['pep0', 'pep'])

# Split into train and test sets, 80% for training, 20% for testing
X_train, X_test, y_train, y_test, pep_train, pep_test = train_test_split(X, y, pep_columns, test_size=0.2, random_state=42, stratify=y)

# Save feature names before standardization
feature_names = X.columns.tolist()

# Initial scaling for feature selection
initial_scaler = StandardScaler()
X_train_initial_scaled = initial_scaler.fit_transform(X_train)

# XGBoost model parameters initialization
params_xgb = {
    'learning_rate': 0.02,
    'objective': 'multi:softprob', # Adjusted for multiclass classification
    'num_class': 3, # Added for multiclass
    'booster': 'gbtree',
    'max_leaves': 127,
    'verbosity': 1,
    'seed': 42,
    'nthread': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'eval_metric': 'mlogloss'
}
model_xgb = xgb.XGBClassifier(**params_xgb)

# LightGBM model parameters initialization
params_lgb = {
    'learning_rate': 0.02,
    'objective': 'multiclass', # Adjusted for multiclass classification
    'boosting_type': 'gbdt',
    'num_class': 3, # Added for multiclass
    'num_leaves': 127,
    'verbosity': 1,
    'seed': 42,
    'n_jobs': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'metric': 'multi_logloss'
}
model_lgb = lgb.LGBMClassifier(**params_lgb)

# CatBoost model parameters initialization
params_cb = {
    'learning_rate': 0.02,
    'depth': 6,
    'verbose': 100,
    'random_seed': 42,
    'thread_count': -1,
    'colsample_bylevel': 0.6,
    'subsample': 0.7,
    'loss_function': 'MultiClass', # Adjusted for multiclass classification
    'classes_count': 3,
    'task_type': 'CPU',
    'bootstrap_type': 'Bernoulli',  # Explicitly specify bootstrap type for subsample
}
model_cb = cb.CatBoostClassifier(**params_cb)

models = {
    'xgboost': model_xgb,
    'lightgbm': model_lgb,
    'catboost': model_cb,
}

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define the hyperparameter search space for XGBoost
param_distributions_xgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.05, 0.1]
}

# Define the hyperparameter search space for LightGBM
param_distributions_lgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.05, 0.1]
}

# Define the hyperparameter search space for CatBoost
param_distributions_cb = {
    'iterations': [100, 200, 300],
    'depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.05, 0.1]
}

# Use RandomizedSearchCV for hyperparameter tuning of XGBoost, LightGBM, and CatBoost models
def tune_hyperparameters(model, param_distributions, X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='neg_log_loss', # Evaluation metric is negative log loss
        cv=skf,
        verbose=1,
        random_state=42,
        n_jobs=4
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

# Save feature importance and select important features
def select_important_features(X, y, models, feature_names):
    all_important_indices = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, model in models.items():
        feature_importance_scores = np.zeros(X.shape[1])

        for train_index, val_index in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            model.fit(X_train_fold, y_train_fold)

            # Use feature importance based on gain (for tree-based models)
            if model_name == "xgboost":
                importance = model.get_booster().get_score(importance_type='gain')
                # Map feature importance to actual feature names
                importance_dict = {feature_names[int(k[1:])]: v for k, v in importance.items()}
            elif model_name == "lightgbm":
                importance = model.booster_.feature_importance(importance_type='gain')
                importance_dict = dict(zip(feature_names, importance))
            elif model_name == "catboost":
                importance = model.get_feature_importance(type='PredictionValuesChange')
                importance_dict = dict(zip(feature_names, importance))
            else:
                raise ValueError(f"Unsupported model type: {model_name}")

            # Convert feature importance to a numpy array
            fold_importance = np.zeros(X.shape[1])
            for idx, feature in enumerate(feature_names):
                if feature in importance_dict:
                    fold_importance[idx] = importance_dict[feature]

            feature_importance_scores += fold_importance

        feature_importance_scores /= 5 # Average feature importance

        # Normalize feature importance
        feature_importance_normalized = feature_importance_scores / np.sum(feature_importance_scores)
        # Dynamically adjust threshold
        threshold = np.mean(feature_importance_normalized) + np.std(feature_importance_normalized)
        important_indices = np.where(feature_importance_normalized > threshold)[0]
        all_important_indices.extend(important_indices)

        # Save feature importance to Excel file
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance_normalized
        }).sort_values(by='importance', ascending=False)
        importance_df.to_excel(f'feature_importance_{model_name}.xlsx', index=False)

    # Count the number of times each feature was selected
    feature_counter = Counter(all_important_indices)
    # Select features that are selected by at least two models
    final_selected_indices = [idx for idx, count in feature_counter.items() if count >= 2]
    # Returns the final feature set
    final_selected_features = [feature_names[idx] for idx in final_selected_indices]
    # Save the final selected feature set
    pd.DataFrame({'selected_features': final_selected_features}).to_excel('final_selected_features.xlsx', index=False)

    return final_selected_indices

def sanitize_filename(filename):
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(' ', '_')


def evaluate_and_save_results(model, X_test, y_test, pep_test, model_name, model_params):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print(f"Debug - {model_name}:")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_pred_proba shape: {y_pred_proba.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Ensure y_pred is one-dimensional
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
        print(f"y_pred shape after ravel: {y_pred.shape}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute various metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # For multiclass problem, calculate multi-class ROC AUC and PR AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
    pr_auc = average_precision_score(y_test_bin, y_pred_proba, average='macro')

    # Create a filename including model parameters
    file_name_suffix = sanitize_filename(f'{model_name}_{now}_{model_params}')

    # Convert model_params to a string
    params_str = str(model_params)
    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_pred_proba shape: {y_pred_proba.shape}")

    results = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'ROC AUC': [roc_auc],
        'PR AUC': [pr_auc],
        'Params': [params_str]
    })
    results.to_excel(f'{file_name_suffix}_results.xlsx', index=False)
    print(f"Results for {model_name}:")
    print(results)

    # Process each column of y_pred_proba separately
    results_df = pd.DataFrame({
        'pep0': pep_test['pep0'].reset_index(drop=True),
        'pep': pep_test['pep'].reset_index(drop=True),
        'True Labels': y_test.reset_index(drop=True),
        'Predicted Labels': y_pred,
        'Predicted Probability Class 0': y_pred_proba[:, 0],  # Probability for Class 0
        'Predicted Probability Class 1': y_pred_proba[:, 1],  # Probability for Class 1
        'Predicted Probability Class 2': y_pred_proba[:, 2]  # Probability for Class 2
    })
    results_df.to_csv(f'{model_name}_test_results.csv', index=False)
    print("y_pred_proba (first 5 samples):")
    print(y_pred_proba[:5])
    print("results_df head:")
    print(results_df.head())

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{file_name_suffix}_confusion_matrix.png')
    plt.close()

    return results


# Main execution flow
# 1. Feature selection (based on initially scaled data)
selected_indices = select_important_features(X_train_initial_scaled, y_train, models, feature_names)

# 2. Rebuild dataset using a subset of the original (unscaled) data
X_train_selected = X_train.iloc[:, selected_indices]
X_test_selected = X_test.iloc[:, selected_indices]

# 3. Rescale the new feature subset
final_scaler = StandardScaler()
X_train_final = final_scaler.fit_transform(X_train_selected)
X_test_final = final_scaler.transform(X_test_selected)

# Save original data along with means and standard deviations
X_train_original = X_train_selected.copy()
X_test_original = X_test_selected.copy()
feature_means = final_scaler.mean_
feature_stds = final_scaler.scale_

# Save final selected feature names and scaler
selected_feature_names = X_train_selected.columns.tolist()
joblib.dump(selected_feature_names, f'selected_feature_names_{sanitize_filename(now)}.joblib')
joblib.dump(final_scaler, f'final_scaler_{sanitize_filename(now)}.joblib')
joblib.dump(feature_means, f'feature_means_{sanitize_filename(now)}.joblib')
joblib.dump(feature_stds, f'feature_stds_{sanitize_filename(now)}.joblib')

# 4. Train models on the new scaled data and tune hyperparameters
best_model_xgb, best_params_xgb = tune_hyperparameters(model_xgb, param_distributions_xgb, X_train_final, y_train)
best_model_lgb, best_params_lgb = tune_hyperparameters(model_lgb, param_distributions_lgb, X_train_final, y_train)
best_model_cb, best_params_cb = tune_hyperparameters(model_cb, param_distributions_cb, X_train_final, y_train)

# 5. Evaluate and save results
results_list = []
results_list.append(
    evaluate_and_save_results(best_model_xgb, X_test_final, y_test, pep_test, 'XGBoost', best_params_xgb))
results_list.append(
    evaluate_and_save_results(best_model_lgb, X_test_final, y_test, pep_test, 'LightGBM', best_params_lgb))
results_list.append(
    evaluate_and_save_results(best_model_cb, X_test_final, y_test, pep_test, 'CatBoost', best_params_cb))

# Combine all results and save to a single file
results_df = pd.concat(results_list, ignore_index=True)
results_df.to_excel(f'combined_results_{sanitize_filename(now)}.xlsx', index=False)

# 6. Plot multi-class ROC and PR curves
plt.figure(figsize=(16, 12))

# Multi-class ROC curve
plt.subplot(221)
for model, model_name in zip([best_model_xgb, best_model_lgb, best_model_cb], ['XGBoost', 'LightGBM', 'CatBoost']):
    y_pred_proba = model.predict_proba(X_test_final)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and AUC for each class
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})')

# Plot random classifier diagonal line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve')
plt.legend(loc="lower right")

# Multi-class PR curve
plt.subplot(222)
for model, model_name in zip([best_model_xgb, best_model_lgb, best_model_cb], ['XGBoost', 'LightGBM', 'CatBoost']):
    y_pred_proba = model.predict_proba(X_test_final)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    precision = dict()
    recall = dict()
    average_precision = dict()

    # Compute PR curve and AUC for each class
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])

    # Compute micro-average PR curve and AUC
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    average_precision["micro"] = average_precision_score(y_test_bin, y_pred_proba, average="micro")

    # Plot micro-average PR curve
    plt.plot(recall["micro"], precision["micro"], label=f'{model_name} (AP = {average_precision["micro"]:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-average Precision-Recall Curve')
plt.legend(loc="lower left")

# Adjust layout to avoid overlap and save the plot
plt.tight_layout()
plt.savefig(f'combined_roc_pr_curves_{now}.png')
plt.close()

# 7. Save the best models
joblib.dump(best_model_xgb, f'best_model_xgb_{sanitize_filename(now)}.joblib')
joblib.dump(best_model_lgb, f'best_model_lgb_{sanitize_filename(now)}.joblib')
joblib.dump(best_model_cb, f'best_model_cb_{sanitize_filename(now)}.joblib')

print("Model training and evaluation are complete. All results and models have been saved.")


# shap analysis
colors = ["#0000FF", "#FF0000"]
n_bins = 100
cmap_name = 'blue_to_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def save_plot_as_formats(fig, filename_prefix):
    #fig.savefig(f'{filename_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{filename_prefix}.svg', dpi=600, bbox_inches='tight')
    plt.close(fig)

def custom_summary_plot(shap_values, X_test_original, feature_names, model_name, class_idx):

    plt.rcParams['font.family'] = 'Tahoma'

    plt.figure(figsize=(26, 22))

    # Calculate and rank the average absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
    feature_order = np.argsort(mean_abs_shap)

    shap.summary_plot(shap_values[:, :, class_idx], X_test_original, plot_type="bar",
                      feature_names=feature_names, show=False,
                      color=plt.cm.coolwarm(np.linspace(0, 1, 256)))

    ax = plt.gca()

    # Make bars thicker
    for bar in ax.patches:
        bar.set_linewidth(3)  # Make the bar outline thicker
        bar.set_edgecolor('black')  # Optionally change the edge color

    # Add specific SHAP values to each bar
    for i, v in enumerate(mean_abs_shap[feature_order]):
        ax.text(v, i, f'{v:.3f}', ha='left', va='center', fontsize=30)

    plt.xlabel('Mean (|SHAP value|)', fontsize=35, fontweight='bold')
    plt.ylabel('Feature', fontsize=35, fontweight='bold')
    plt.xticks(fontsize=33, fontweight='bold')
    plt.yticks(fontsize=33, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    # Limit number of ticks on x-axis to 5
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_xlim(0, max(mean_abs_shap) + 0.3)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')

    ax.tick_params(axis='x', width=5, length=20)

    save_plot_as_formats(plt.gcf(), f'SHAP_summary_{model_name}_class_{class_idx}')
    plt.close()

    plt.figure(figsize=(26, 80))

    n_bins = 256
    colors = ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090", "#FDAE61", "#F46D43",
              "#D73027", "#A50026"]

    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    shap.summary_plot(shap_values[:, :, class_idx], X_test_original,
                      plot_type="dot", feature_names=feature_names, show=False,
                      cmap=custom_cmap,
                      alpha=0.8,
                      )

    # Get current axis for beeswarm plot
    ax = plt.gca()
    # Resize all scatter
    for collection in ax.collections:
        collection.set_sizes([100])

    plt.xlabel('SHAP value', fontsize=35, fontweight='bold')
    plt.ylabel('Feature', fontsize=35, fontweight='bold')
    plt.xticks(fontsize=30, fontweight='bold')
    plt.yticks(fontsize=32, fontweight='bold')

    # Make borders bold
    for spine in ax.spines.values():
        spine.set_linewidth(5)
        spine.set_color('black')

    # Adjust X-axis tick sizes
    ax.tick_params(axis='x', width=6, length=25)

    # Set exactly 5 ticks on the X-axis
    x_min, x_max = ax.get_xlim()
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    fig = plt.gcf()

    main_ax = fig.axes[0]

    scatter = None
    for collection in main_ax.collections:
        if isinstance(collection, plt.scatter([], []).__class__):
            scatter = collection
            break

    if scatter is None:
        raise ValueError("Could not find scatter plot in the main axes")

    old_cbar = fig.axes[-1]
    pos = old_cbar.get_position()
    cax = fig.add_axes([pos.x0 + 0.02, pos.y0, pos.width * 0.3, pos.height])
    old_cbar.remove()

    vmin, vmax = shap_values[:, :, class_idx].min(), shap_values[:, :, class_idx].max()

    new_cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=custom_cmap), cax=cax)

    new_cbar.outline.set_linewidth(3)
    new_cbar.outline.set_edgecolor('black')

    new_cbar.ax.tick_params(labelsize=30, width=3, length=10)  # 设置刻度标签字体大小，刻度线宽度和长度
    new_cbar.set_label("Feature value", fontsize=30, fontweight='bold', rotation=270, labelpad=30)

    new_cbar.set_ticks([vmin, vmax])
    new_cbar.set_ticklabels(['Low', 'High'])

    for label in new_cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    new_cbar.outline.set_visible(True)

    save_plot_as_formats(plt.gcf(), f'SHAP_beeswarm_{model_name}_class_{class_idx}')
    plt.close()


def find_zero_crossings(x_fit, y_fit):
    """
    Finds the intersections of the LOWESS curve with y=0 by linear interpolation.
    Returns a list of (x, 0) points where the curve crosses y=0.
    """
    crossings = []

    # Check for sign changes in y_fit
    for i in range(1, len(y_fit)):
        if (y_fit[i - 1] < 0 and y_fit[i] > 0) or (y_fit[i - 1] > 0 and y_fit[i] < 0):
            # Perform linear interpolation to find where the curve crosses y=0
            f = interp1d([x_fit[i - 1], x_fit[i]], [y_fit[i - 1], y_fit[i]], kind='linear', fill_value='extrapolate')

            # Use fsolve to find the root of the interpolated function where it crosses y=0
            crossing = fsolve(f, x_fit[i - 1])[0]
            crossings.append(crossing)

    return crossings


def shap_dependence_plot(shap_values, X_test_original, feature_names, class_idx, feature_idx, model_name):
    fig, ax1 = plt.subplots(figsize=(18, 14))

    feature = X_test_original[feature_names[feature_idx]].values
    shap_values_feature = shap_values[:, feature_idx, class_idx]

    # 使用不同的颜色来区分特征
    color_map = {
        'logP': '#5b91be',
        'net.charge': '#42c1b5',
        'pI': '#FDAE61'
    }
    scatter_color = color_map.get(feature_names[feature_idx], '#8172b2')

    edge_color_map = {
        'logP': '#447583',
        'net.charge': '#8bb383',
        'pI': '#CF7A35'
    }
    scatter_edge_color = edge_color_map.get(feature_names[feature_idx], '#554882')

    ax2 = ax1.twinx()
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    counts, bins, _ = ax2.hist(feature, bins=30, alpha=0.3, color='#A9A9A9', zorder=1)
    ax2.set_ylabel('Frequency', fontsize=66, fontweight='bold', color='#707070')
    ax2.tick_params(axis='y', labelcolor='#707070', labelsize=64, width=11, length=30)

    ax2.set_ylim(0, max(counts) * 1.1)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    scatter = ax1.scatter(feature, shap_values_feature, s=500, alpha=0.6, c=scatter_color, edgecolor=scatter_edge_color, linewidth=2.)

    lowess = sm.nonparametric.lowess(shap_values_feature, feature, frac=0.6)
    ax1.plot(lowess[:, 0], lowess[:, 1], color='black', linewidth=5)

    ax1.axhline(y=0, color='k', linestyle='--', linewidth=9)

    intersections = find_zero_crossings(lowess[:, 0], lowess[:, 1])
    for x_inter in intersections:
        ax1.plot(x_inter, 0, marker='o', markersize=33, color='yellow', markeredgecolor='black', markeredgewidth=4)

    ax1.set_xlabel(feature_names[feature_idx], fontsize=66, fontweight='bold')
    ax1.set_ylabel(f'SHAP — {feature_names[feature_idx]}', fontsize=64, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    ax1.tick_params(axis='both', which='major', labelsize=64, width=11, length=30, color='black', labelcolor='black')

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(66)

    for spine in ax1.spines.values():
        spine.set_linewidth(10)
        spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    save_plot_as_formats(fig, f'SHAP_dependence_{model_name}_class_{class_idx}_{feature_names[feature_idx]}')
    plt.close()



def shap_main_effect_plot(shap_values, X_test_original, feature_names, class_idx, feature_idx, model_name, model):
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(18, 14))

    # Get feature values
    feature = X_test_original[feature_names[feature_idx]].values

    # Compute SHAP interaction values
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_test_original)

    # If the model is CatBoost, we reshape interaction values
    if isinstance(interaction_values, list):
        interaction_values = np.array(interaction_values)
    if interaction_values.shape[0] == 3:  # CatBoost case
        interaction_values = np.transpose(interaction_values, (1, 2, 3, 0))

    # Extract main effect values (diagonal elements)
    main_effects = interaction_values[:, feature_idx, feature_idx, class_idx]

    if feature_names[feature_idx] == 'aromatic':
        # Special handling for Aromatic feature
        unique_aromatic_values = np.unique(feature)

        # Group SHAP main effect values by aromatic count
        grouped_shap = {value: main_effects[feature == value] for value in unique_aromatic_values}

        # Create box plots
        bp = ax1.boxplot([grouped_shap[val] for val in sorted(grouped_shap.keys())], patch_artist=True)

        # Customize box plots
        colors = ["#D95319", "#0072BD", "#77AC30", "#7E2F8E", "#ED8120"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(7)
        for median in bp['medians']:
            median.set_linewidth(6.5)
        for whisker in bp['whiskers']:
            whisker.set_linewidth(7)
        for cap in bp['caps']:
            cap.set_linewidth(7)

        # Set x-axis labels
        ax1.set_xticks(range(1, len(grouped_shap) + 1))
        ax1.set_xticklabels(sorted(grouped_shap.keys()))

        # Add scatter plot points with jittered x-axis positions
        for i, key in enumerate(sorted(grouped_shap.keys())):
            y = grouped_shap[key]
            x = np.random.normal(1 + i, 0.04, size=len(y))
            ax1.scatter(x, y, alpha=0.8, color=colors[i % len(colors)])
    else:
        # Original scatter plot for other features
        color_map = {
            'logP': '#5b91be',
            'net.charge': '#42c1b5',
            'pI': '#FDAE61'
        }
        scatter_color = color_map.get(feature_names[feature_idx], '#8172b2')

        # Define edge color map for the scatter plot
        edge_color_map = {
            'logP': '#447583',
            'net.charge': '#8bb383',
            'pI': '#CF7A35'
        }
        scatter_edge_color = edge_color_map.get(feature_names[feature_idx], '#554882')

        # Scatter plot with customized edge color
        scatter = ax1.scatter(feature, main_effects, s=500, alpha=0.6, c=scatter_color, edgecolor=scatter_edge_color,
                              linewidth=2.)

        # LOWESS regression line
        lowess = sm.nonparametric.lowess(main_effects, feature, frac=0.6)
        ax1.plot(lowess[:, 0], lowess[:, 1], color='black', linewidth=5)

        # Zero line
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=9)

        # Find intersections with y=0 and annotate
        intersections = find_zero_crossings(lowess[:, 0], lowess[:, 1])
        for x_inter in intersections:
            ax1.plot(x_inter, 0, marker='o', markersize=33, color='yellow', markeredgecolor='black', markeredgewidth=4)

    # Set labels for x and y axes
    ax1.set_xlabel(feature_names[feature_idx], fontsize=66, fontweight='bold')
    ax1.set_ylabel(
        r'$\mathbf{SHAP}_{\mathbf{main}}$ — ' + f'{feature_names[feature_idx]}',
        fontsize=64, labelpad=15, fontweight='bold'
    )
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    # Customize ticks and labels
    ax1.tick_params(axis='both', which='major', labelsize=64, width=11, length=30, colors='black', labelcolor='black')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(66)

    # Enhance the border of the plot
    for spine in ax1.spines.values():
        spine.set_linewidth(10)
        spine.set_color('black')

    # Limit y-axis to 4 ticks
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))

    # Tight layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save the plot in multiple formats
    save_plot_as_formats(fig, f'main_effect_plot_{model_name}_class_{class_idx}_{feature_names[feature_idx]}')
    plt.close(fig)


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def shap_interaction_plot(interaction_values, X_test_original, feature_names, class_idx, feature1_idx, feature2_idx,
                          model_name):
    # Make sure interaction_values is a numpy array.
    if isinstance(interaction_values, list):
        interaction_values = np.array(interaction_values)

    # If feature1 is Aromatic, exchange feature1 and feature2.
    if feature_names[feature1_idx].lower() == 'aromatic':
        feature1_idx, feature2_idx = feature2_idx, feature1_idx

    fig, ax1 = plt.subplots(figsize=(23, 18))

    feature1 = X_test_original[feature_names[feature1_idx]].values
    feature2 = X_test_original[feature_names[feature2_idx]].values

    # Calculating SHAP Interaction Values
    shap_interaction_vals = interaction_values[:, feature1_idx, feature2_idx, class_idx]

    colors = ["#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090", "#FDAE61", "#F46D43",
              "#D73027"]
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom", colors)

    # Plotting a scatterplot, using feature2 as the colour mapping
    scatter = ax1.scatter(feature1, shap_interaction_vals, c=feature2, cmap=custom_cmap, s=500, alpha=0.8,
                          edgecolor='k', linewidth=2.)

    # Add horizontal reference line (y=0)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=9.2)

    plt.subplots_adjust(left=0.18, right=0.82, top=0.85, bottom=0.15)

    ax1.set_position([0.18, 0.15, 0.60, 0.75])  # [left, bottom, width, height]

    cax = fig.add_axes([0.82, 0.16, 0.04, 0.72])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(feature_names[feature2_idx], fontsize=55, labelpad=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=55, width=6, length=18)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontweight('bold')

    for spine in cax.spines.values():
        spine.set_linewidth(6.5)
        spine.set_color('black')

    cax.set_position([0.82, 0.16, 0.04, 0.72])

    if feature_names[feature2_idx].lower() == 'aromatic':
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.set_ticklabels(['0', '1', '2', '3', '4'])

    ax1.set_xlabel(feature_names[feature1_idx], fontsize=58, labelpad=15, fontweight='bold')
    ax1.set_ylabel(
        r'$\mathbf{SHAP}_{\mathbf{inter}}$ — ' + f'{feature_names[feature1_idx]} and {feature_names[feature2_idx]}',
        fontsize=58, labelpad=15, fontweight='bold')

    ax1.tick_params(axis='both', which='major', labelsize=55, width=10, length=28, colors='black', labelcolor='black')

    x_padding = (feature1.max() - feature1.min()) * 0.05
    y_padding = (shap_interaction_vals.max() - shap_interaction_vals.min()) * 0.05

    ax1.set_xlim(feature1.min() - x_padding, feature1.max() + x_padding)
    ax1.set_ylim(shap_interaction_vals.min() - y_padding, shap_interaction_vals.max() + y_padding)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')

    if feature_names[feature1_idx].lower() == 'aromatic':
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['0', '1', '2', '3', '4'])
    else:
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.set_aspect(1. / ax1.get_data_ratio())

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(55)

    ax1.grid(True, linestyle='--', alpha=0.5)

    for spine in ax1.spines.values():
        spine.set_linewidth(9)
        spine.set_color('black')

    save_plot_as_formats(fig,
                         f'interaction_effect_plot_{model_name}_class_{class_idx}_{feature_names[feature1_idx]}_{feature_names[feature2_idx]}')
    plt.close(fig)


def analyze_aromatic_residues(shap_values, X_test, model_name, feature_names):
    # Get the index of the 'Aromatic' feature
    aromatic_index = feature_names.index('aromatic')
    unique_aromatic_values = np.unique(X_test[:, aromatic_index])
    print("Unique Aromatic values:", unique_aromatic_values)

    # Number of comparisons
    num_comparisons = 4
    # Bonferroni corrected significance levels
    bonferroni_05 = 0.05 / num_comparisons
    bonferroni_01 = 0.01 / num_comparisons
    bonferroni_001 = 0.001 / num_comparisons

    # Create a mapping from standardized values to actual values (0, 1, 2, 3, 4)
    standardized_values = [-1.00016402, 0.24901629, 1.49819659, 2.7473769, 3.9965572]
    actual_values = [0, 1, 2, 3, 4]
    value_mapping = dict(zip(standardized_values, actual_values))

    def map_value(value):
        for std_value in standardized_values:
            if np.isclose(value, std_value, atol=1e-8):
                return value_mapping[std_value]
        return None

    aromatic_counts = X_test[:, aromatic_index]
    aromatic_counts_actual = np.array([map_value(count) for count in aromatic_counts])
    valid_indices = [i for i, val in enumerate(aromatic_counts_actual) if val is not None]

    # Filter out invalid counts
    aromatic_counts_actual = aromatic_counts_actual[valid_indices]
    aromatic_counts = aromatic_counts[valid_indices]

    for class_index in range(3):  # For each class (0, 1, 2)
        # Get SHAP values for the current class
        shap_aromatic = shap_values[:, :, class_index][:, aromatic_index]
        shap_aromatic = shap_aromatic[valid_indices]

        # Grouped SHAP by actual aromatic residue count
        grouped_shap = {}
        for count in unique_aromatic_values:
            mapped_value = map_value(count)
            if mapped_value is not None:
                grouped_shap[mapped_value] = shap_aromatic[aromatic_counts_actual == mapped_value]

        # Visualization
        fig, ax1 = plt.subplots(figsize=(18, 14))

        # Define colors for each box plot group
        predefined_colors = ["#D95319", "#0072BD", "#77AC30", "#7E2F8E", "#ED8120"]
        colors = predefined_colors[:len(actual_values)]

        # Create box plots with different colors
        bp = ax1.boxplot(
            [grouped_shap[val] for val in sorted(grouped_shap.keys())],
            patch_artist=True
        )

        # Set the colors of each box plot
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(7)

        for median in bp['medians']:
            median.set_linewidth(6.5)
        for whisker in bp['whiskers']:
            whisker.set_linewidth(7)
        for cap in bp['caps']:
            cap.set_linewidth(7)

        # Set x-axis labels to actual values
        ax1.set_xticks(range(1, len(grouped_shap) + 1))
        ax1.set_xticklabels(sorted(grouped_shap.keys()))
        ax1.set_xlabel('aromatic', fontsize=66, fontweight='bold', color="k")
        ax1.set_ylabel(f'SHAP — aromatic', fontsize=66, fontweight='bold', color="k")

        # Add scatter plot points for each box plot
        for i, key in enumerate(sorted(grouped_shap.keys())):
            y = grouped_shap[key]
            x = np.random.normal(1 + i, 0.04, size=len(y))
            ax1.scatter(x, y, alpha=0.8, color=colors[i])

        # Calculate mean SHAP values
        mean_shap = {key: np.mean(grouped_shap[key]) for key in grouped_shap}

        # Statistical analysis
        print(f"\nAromatic Residues Analysis for {model_name} - Class {class_index}:")
        grouped_keys = sorted(grouped_shap.keys())
        for i in range(len(grouped_keys) - 1):
            count1, count2 = grouped_keys[i], grouped_keys[i + 1]
            diff = mean_shap[count2] - mean_shap[count1]
            stat, p_value = stats.mannwhitneyu(grouped_shap[count1], grouped_shap[count2], alternative='two-sided')

            print(f"When aromatic residues increase from {count1} to {count2}:")
            print(f"  Mean SHAP value change: {diff:.4f}")
            print(f"  This changes the prediction probability for Class {class_index} by {diff * 100:.2f} percentage points on average.")
            print(f"  Mann-Whitney U statistic: {stat:.4f}, p-value: {p_value:.4f}")

            if p_value < bonferroni_001:
                significance = "*** (p < 0.001)"
            elif p_value < bonferroni_01:
                significance = "** (p < 0.01)"
            elif p_value < bonferroni_05:
                significance = "* (p < 0.05)"
            else:
                significance = "NS (not significant)"

            print(f"  Significance after Bonferroni correction: {significance}")

            # Add significance annotation to the plot
            y_max = max([max(bp['whiskers'][k].get_ydata()) for k in range(len(grouped_shap))])
            x1, x2 = i + 1, i + 2
            y, h, col = y_max + 0.1, 0.05, 'k'
            ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=7, c=col)

            annotation_size = 55
            if p_value < bonferroni_001:
                ax1.text((x1 + x2) * .5, y + h, '***', ha='center', va='bottom', color='k', fontsize=annotation_size, fontweight='bold')
            elif p_value < bonferroni_01:
                ax1.text((x1 + x2) * .5, y + h, '**', ha='center', va='bottom', color='k', fontsize=annotation_size, fontweight='bold')
            elif p_value < bonferroni_05:
                ax1.text((x1 + x2) * .5, y + h, '*', ha='center', va='bottom', color='k', fontsize=annotation_size, fontweight='bold')
            else:
                ax1.text((x1 + x2) * .5, y + h, 'NS', ha='center', va='bottom', color='k', fontsize=annotation_size, fontweight='bold')

        # Customize the plot
        for spine in ax1.spines.values():
            spine.set_linewidth(10)
            spine.set_color('black')

        ax1.tick_params(axis='both', which='major', length=30, width=11, labelsize=64, color='k', labelcolor='k')

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(66)

        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim(y_min, y_max + (y_max - y_min) * 0.05)

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        save_plot_as_formats(fig, f'aromatic_residues_shap_{model_name}_class_{class_index}')
        plt.close()



def create_custom_cmap():
    colors = ["#FFA07A", "#FF4500", "#FFFFFF", "#1E90FF", "#87CEFA"]
    cmap = LinearSegmentedColormap.from_list("custom_seismic", colors, N=256)
    return cmap


def plot_interaction_heatmap(explainer, shap_values, X_test, feature_names, class_idx, model_name):
    # Get interaction values
    interaction_values = explainer.shap_interaction_values(X_test)

    # Check if interaction_values is a list, and convert to a NumPy array if it is
    if isinstance(interaction_values, list):
        interaction_values = np.array(interaction_values)

    # Check the shape of interaction values for CatBoost model, and transpose it to (900, 7, 7, 3)
    if interaction_values.shape[0] == 3:  # If the first dimension is the number of classes (common for CatBoost)
        interaction_values = np.transpose(interaction_values, (1, 2, 3, 0))

    # Print the shape of interaction_values to check the structure
    print(f"interaction_values shape after transpose (if needed): {interaction_values.shape}")

    # Ensure that interaction_values is a 4D array, and handle 4D indexing
    if interaction_values.ndim != 4:
        print(f"Unexpected shape of interaction_values: {interaction_values.shape}")
        return

    # Calculate main effects (average of each feature across each sample)
    shap_main_effects = np.mean(shap_values[:, :, class_idx], axis=0)

    # Calculate interaction effects, take interaction values for class_idx
    interaction_effects = np.mean(interaction_values[:, :, :, class_idx], axis=0)

    # Add main effects to the diagonal
    np.fill_diagonal(interaction_effects, shap_main_effects)

    # Create DataFrame for plotting
    interaction_df = pd.DataFrame(interaction_effects, index=feature_names, columns=feature_names)

    # Check main effect values
    print(f"Main effects for class {class_idx}: {shap_main_effects}")

    # Cluster and create dendrogram
    row_linkage = linkage(interaction_df, method='average')
    col_linkage = linkage(interaction_df.T, method='average')

    custom_cmap = create_custom_cmap()
    vmin = -0.2
    vmax = 0.2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    cg = sns.clustermap(interaction_df, cmap=custom_cmap, annot=True, fmt=".2f", linewidths=1.5,
                        row_linkage=row_linkage, col_linkage=col_linkage, figsize=(14, 14), cbar=False, norm=norm)

    cg.cax.remove()
    cg.fig.set_size_inches(14, 14)
    plt.tight_layout()

    # Set color based on value magnitude: black for values near 0, white for extreme values
    for text in cg.ax_heatmap.texts:
        val = float(text.get_text())
        text_color = 'black' if abs(val) < 0.1 else 'white'  # black near 0, white for extreme values
        text.set_color(text_color)

    plt.setp(cg.ax_heatmap.get_ymajorticklabels(), rotation=0, fontsize=20, fontweight='bold', color='black')
    plt.setp(cg.ax_heatmap.get_xmajorticklabels(), rotation=90, fontsize=20, fontweight='bold', color='black')

    # Adjust the size and font weight of text in the heatmap
    for text in cg.ax_heatmap.texts:
        text.set_size(16)
        text.set_fontweight('bold')

    # Set the line width for the spines of the heatmap
    for spine in cg.ax_heatmap.spines.values():
        spine.set_linewidth(5)

    # Set the line width for dendrogram lines
    dendrogram_lines = cg.ax_row_dendrogram.collections[0]
    dendrogram_lines.set_linewidth(4)

    dendrogram_lines = cg.ax_col_dendrogram.collections[0]
    dendrogram_lines.set_linewidth(4)

    cg.ax_col_dendrogram.set_visible(True)

    # Save the plot in multiple formats
    save_plot_as_formats(cg.fig, f'SHAP_interaction_heatmap_{model_name}_class_{class_idx}')


def shap_analysis(model, X_test, X_test_original, model_name, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    assert shap_values.ndim == 3 and shap_values.shape[2] == 3, "Unexpected shape of shap_values"
    print(f"shap_values shape: {shap_values.shape}")

    if isinstance(X_test_original, np.ndarray):
        X_test_original = pd.DataFrame(X_test_original, columns=feature_names)

    important_features = ['aromatic', 'logP', 'pI', 'net.charge']
    for class_idx in range(3):
        custom_summary_plot(shap_values, X_test_original, feature_names, model_name, class_idx)
        plot_interaction_heatmap(explainer, shap_values, X_test, feature_names, class_idx, model_name)

        for feature in important_features:
            feature_idx = feature_names.index(feature)
            shap_dependence_plot(shap_values, X_test_original, feature_names, class_idx, feature_idx, model_name)
            shap_main_effect_plot(shap_values, X_test_original, feature_names, class_idx, feature_idx, model_name, model)

        interaction_values = explainer.shap_interaction_values(X_test)

        if model_name == 'CatBoost':
            # Converts CatBoost shape (3, 900, 7, 7) to (900, 7, 7, 3).
            interaction_values = np.transpose(interaction_values, (1, 2, 3, 0))

        for i, feature1 in enumerate(important_features):
            for feature2 in important_features[i + 1:]:
                feature1_idx = feature_names.index(feature1)
                feature2_idx = feature_names.index(feature2)
                shap_interaction_plot(interaction_values, X_test_original, feature_names, class_idx, feature1_idx,
                                      feature2_idx, model_name)

    analyze_aromatic_residues(shap_values, X_test, model_name, feature_names)

    mean_abs_shap_class0 = np.abs(shap_values[:, :, 0]).mean(axis=0)
    mean_abs_shap_class1 = np.abs(shap_values[:, :, 1]).mean(axis=0)
    mean_abs_shap_class2 = np.abs(shap_values[:, :, 2]).mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'Class_0': mean_abs_shap_class0,
        'Class_1': mean_abs_shap_class1,
        'Class_2': mean_abs_shap_class2
    }).set_index('feature')

    feature_importance_sorted = feature_importance.sort_values(by='Class_0', ascending=True)

    palette = sns.color_palette("coolwarm", 3)

    fig, ax = plt.subplots(figsize=(16, 12))
    feature_importance_sorted.plot(kind='barh', width=0.8, ax=ax, color=palette)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=20)

    ax.set_xlabel('Mean absolute SHAP value', fontsize=35, fontweight='bold', color='black')
    ax.set_ylabel('Feature', fontsize=35, fontweight='bold', color='black')
    ax.tick_params(axis='x', labelsize=25, width=2, color='black')
    ax.tick_params(axis='y', labelsize=25, width=2, color='black')

    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()
    save_plot_as_formats(fig, f'SHAP_importance_{model_name}_all_classes')
    plt.close(fig)

    return feature_importance_sorted


# 执行SHAP分析
shap_analysis(best_model_xgb, X_test_final, X_test_original, 'XGBoost', selected_feature_names)
shap_analysis(best_model_lgb, X_test_final, X_test_original, 'LightGBM', selected_feature_names)
shap_analysis(best_model_cb, X_test_final, X_test_original, 'CatBoost', selected_feature_names)

print("SHAP analysis completed. All plots have been saved.")