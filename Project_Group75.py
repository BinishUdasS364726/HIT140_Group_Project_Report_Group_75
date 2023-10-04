import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from scipy import stats as scipy_stats


# Load Data
data = pd.read_csv(r"C:\Users\binis\OneDrive\Desktop\binish.conda\po2_data.csv")

# Calculate the number of missing values for each column
missing_values = data.isnull().sum()
print(missing_values)



# Define features and target variables
features = data.drop(columns=['subject#', 'age', 'sex', 'test_time', 'motor_updrs', 'total_updrs'])
X = features.fillna(features.mean())
y_motor = data['motor_updrs']
y_total = data['total_updrs']

# Log transformation
X_log = np.log(X + 1e-8)
X_log = X_log.replace([np.inf, -np.inf], np.nan)
X_log = X_log.fillna(X_log.mean())

# Check for NaN or Inf values in X_log_reduced
print("NaNs in X_log_reduced:", X_log.isna().sum().sum())
print("Infs in X_log_reduced:", np.isinf(X_log).sum().sum())

# If NaNs or Infs are found in X_log_reduced, replace them
if X_log.isna().sum().sum() > 0 or np.isinf(X_log).sum().sum() > 0:
    X_log = X_log.replace([np.inf, -np.inf], np.nan)
    X_log = X_log.fillna(X_log.mean())

# Functions for Baseline and Linear Regression Evaluation
def evaluate_baseline(y_train, y_test, n_features):
    y_pred = [y_train.mean()] * len(y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_test.max() - y_test.min())
    r2_val = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1-r2_val)*(len(y_test)-1)/(len(y_test)-n_features-1)
    return mae, mse, rmse, nrmse, r2_val, adj_r2

def evaluate_lr(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_test.max() - y_test.min())
    r2_val = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1-r2_val)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    return mae, mse, rmse, nrmse, r2_val, adj_r2

# Baseline Model
X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X, y_motor, test_size=0.4, random_state=42)
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.4, random_state=42)

mae_motor_baseline, mse_motor_baseline, rmse_motor_baseline, nrmse_motor_baseline, r2_motor_baseline, adj_r2_motor_baseline = evaluate_baseline(y_train_motor, y_test_motor, X.shape[1])
mae_total_baseline, mse_total_baseline, rmse_total_baseline, nrmse_total_baseline, r2_total_baseline, adj_r2_total_baseline = evaluate_baseline(y_train_total, y_test_total, X.shape[1])
print(f"Baseline for motor_updrs: MAE: {mae_motor_baseline}, MSE: {mse_motor_baseline}, RMSE: {rmse_motor_baseline}, NRMSE: {nrmse_motor_baseline}, r2: {r2_motor_baseline}, adjusted-r2: {adj_r2_motor_baseline}\n")
print(f"Baseline for total_updrs: MAE: {mae_total_baseline}, MSE: {mse_total_baseline}, RMSE: {rmse_total_baseline}, NRMSE: {nrmse_total_baseline}, r2: {r2_total_baseline}, adjusted-r2: {adj_r2_total_baseline}\n")


# Evaluate Linear Regression for Different Splits
splits = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
for train_size, test_size in splits:
    print(f"Evaluation for {int(train_size*100)}-{int(test_size*100)} split:\n")
    
    # Split for motor_updrs
    X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X, y_motor, test_size=test_size, random_state=42)
    mae_motor, mse_motor, rmse_motor, nrmse_motor, r2_motor, adj_r2_motor = evaluate_lr(X_train_motor, X_test_motor, y_train_motor, y_test_motor)
    print(f"For motor_updrs: MAE: {mae_motor}, MSE: {mse_motor}, RMSE: {rmse_motor}, NRMSE: {nrmse_motor}, r2: {r2_motor}, adjusted-r2: {adj_r2_motor}\n")
    
    # Split for total_updrs
    X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X, y_total, test_size=test_size, random_state=42)
    mae_total, mse_total, rmse_total, nrmse_total, r2_total, adj_r2_total = evaluate_lr(X_train_total, X_test_total, y_train_total, y_test_total)
    print(f"For total_updrs: MAE: {mae_total}, MSE: {mse_total}, RMSE: {rmse_total}, NRMSE: {nrmse_total}, r2: {r2_total}, adjusted-r2: {adj_r2_total}\n")
...


# Log-transform and Collinearity Analysis
X_log = np.log(X + 1e-8)

# Calculate VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

vif_df = calculate_vif(X_log)

def print_vif(vif_data):
    print("Collinearity Analysis using Variance Inflation Factor (VIF):\n")
    print("{:<30} {:<15}".format("Feature", "VIF"))
    print("-" * 45)
    for index, row in vif_data.iterrows():
        print("{:<30} {:<15.2f}".format(row["Variable"], row["VIF"]))

print_vif(vif_df)


# Filter out features with high VIF
cols_to_drop = vif_df[vif_df["VIF"] > 50]["Variable"].values
X_log_reduced = X_log.drop(columns=cols_to_drop)

# Metrics storage
metrics = {}

# Model evaluation for log-transformed data
splits = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
for train_size, test_size in splits:
    X_train_log, X_test_log, y_train_total, y_test_total = train_test_split(X_log_reduced, y_total, test_size=test_size, random_state=42)
    mae, mse, rmse, nrmse, r2, adj_r2 = evaluate_lr(X_train_log, X_test_log, y_train_total, y_test_total) 
    split_name = f"log-transformed data (total_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2}

    X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X, y_motor, test_size=test_size, random_state=42)
    mae_motor, mse_motor, rmse_motor, nrmse_motor, r2_motor, adj_r2_motor = evaluate_lr(X_train_motor, X_test_motor, y_train_motor, y_test_motor)
    split_name = f"log-transformed data (motor_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2} 
    
    split_name = f"Gaussian-transformed data (total_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2}
    split_name = f"Gaussian-transformed data (motor_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2}

# Convert the metrics dictionary to a DataFrame and display it
df_metrics = pd.DataFrame(metrics).transpose()
print(df_metrics)



# Standardisation and Gaussian Transformation
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X_gaussian = transformer.fit_transform(X_standardized)

# Model evaluation for Gaussian-transformed data
for train_size, test_size in splits:
    X_train, X_test, y_train, y_test = train_test_split(X_gaussian, y_motor, test_size=test_size, random_state=42)
    mae, mse, rmse, nrmse, r2, adj_r2 = evaluate_lr(X_train, X_test, y_train, y_test)
    split_name = f"Gaussian-transformed data (motor_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2}

for train_size, test_size in splits:
    X_train, X_test, y_train, y_test = train_test_split(X_gaussian, y_motor, test_size=test_size, random_state=42)
    mae, mse, rmse, nrmse, r2, adj_r2 = evaluate_lr(X_train, X_test, y_train, y_test)
    split_name = f"Gaussian-transformed data (total_updrs, {int(train_size*100)}-{int(test_size*100)} split)"
    metrics[split_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R^2': r2, 'Adjusted R^2': adj_r2}
    
# Convert the metrics dictionary to a DataFrame and display it
df_metrics = pd.DataFrame(metrics).transpose()
print(df_metrics)


# Train a linear regression model on the Gaussian transformed data
lr_motor_gaussian = LinearRegression()
lr_motor_gaussian.fit(X_train, y_train)
lr_motor_gaussian.fit(X_train, y_train_total)


# Get feature importance
feature_importance = abs(lr_motor_gaussian.coef_)
feature_importance_total = abs(lr_motor_gaussian.coef_)
feature_names = X.columns

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df_total = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_total})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df_total = feature_importance_df_total.sort_values(by='Importance', ascending=False)

print("Metrics for motor_updrs:")
print(feature_importance_df)
print("\n\nMetrics for total_updrs:")
print(feature_importance_df_total)





# Predict on the test set
y_pred_motor_gaussian = lr_motor_gaussian.predict(X_test)
y_pred_total_gaussian = lr_motor_gaussian.predict(X_test)



# Calculate residuals
residuals = y_test_motor - y_pred_motor_gaussian
residuals_total = y_test_total - y_pred_total_gaussian


# Performance on original data
original_metrics = evaluate_lr(X_train_motor, X_test_motor, y_train_motor, y_test_motor)
original_metrics_total = evaluate_lr(X_train_total, X_test_total, y_train_total, y_test_total)


# Performance on log-transformed data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_reduced, y_motor, test_size=0.4, random_state=42)
log_transformed_metrics = evaluate_lr(X_train_log, X_test_log, y_train_log, y_test_log)

X_train_log_total, X_test_log_total, y_train_log_total, y_test_log_total = train_test_split(X_log_reduced, y_total, test_size=0.4, random_state=42)
log_transformed_metrics_total = evaluate_lr(X_train_log_total, X_test_log_total, y_train_log_total, y_test_log_total)

# Performance on Gaussian-transformed data
X_train_gaussian, X_test_gaussian, y_train_gaussian, y_test_gaussian = train_test_split(X_gaussian, y_motor, test_size=0.4, random_state=42)
gaussian_transformed_metrics = evaluate_lr(X_train_gaussian, X_test_gaussian, y_train_gaussian, y_test_gaussian)

X_train_gaussian_total, X_test_gaussian_total, y_train_gaussian_total, y_test_gaussian_total = train_test_split(X_gaussian, y_total, test_size=0.4, random_state=42)
gaussian_transformed_metrics_total = evaluate_lr(X_train_gaussian_total, X_test_gaussian_total, y_train_gaussian_total, y_test_gaussian_total)

# Consolidate the metrics into a DataFrame for comparison
metric_names = ["MAE", "MSE", "RMSE", "NRMSE", "R^2", "Adjusted R^2"]
df_comparison = pd.DataFrame({
    'Original Data': original_metrics,
    'Log-Transformed Data': log_transformed_metrics,
    'Gaussian-Transformed Data': gaussian_transformed_metrics
}, index=metric_names)



df_comparison_total = pd.DataFrame({
    'Original Data': original_metrics_total,
    'Log-Transformed Data': log_transformed_metrics_total,
    'Gaussian-Transformed Data': gaussian_transformed_metrics_total
}, index=metric_names)

# Adjust the column names
df_comparison.columns = ["Motor - " + col for col in df_comparison.columns]
df_comparison_total.columns = ["Total - " + col for col in df_comparison_total.columns]

# Print both dataframes 
print("Metrics for motor_updrs:")
print(df_comparison)
print("\n\nMetrics for total_updrs:")
print(df_comparison_total)



# Calculate performance gains for Motor_UPDRS and Total_UPDRS
log_gain = ((df_comparison['Motor - Original Data'] - df_comparison['Motor - Log-Transformed Data']) / df_comparison['Motor - Original Data']) * 100
log_gain_total = ((df_comparison_total['Total - Original Data'] - df_comparison_total['Total - Log-Transformed Data']) / df_comparison_total['Total - Original Data']) * 100
gaussian_gain = ((df_comparison['Motor - Original Data'] - df_comparison['Motor - Gaussian-Transformed Data']) / df_comparison['Motor - Original Data']) * 100
gaussian_gain_total = ((df_comparison_total['Total - Original Data'] - df_comparison_total['Total - Gaussian-Transformed Data']) / df_comparison_total['Total - Original Data']) * 100



df_gains = pd.DataFrame({
    'Log-Transformed Gain (%)': log_gain,
    'Gaussian-Transformed Gain (%)': gaussian_gain
})
print("Gains for motor_updrs")
print(df_gains)

df_gains_total = pd.DataFrame({
    'Log-Transformed Gain (%)': log_gain_total,
    'Gaussian-Transformed Gain (%)': gaussian_gain_total
})
print("Gains for total_updrs:")
print(df_gains_total)








# Train and get feature importance for the original data for motor_updrs
lr_original_motor = LinearRegression()
lr_original_motor.fit(X_train_motor, y_train_motor)
feature_importance_original_motor = abs(lr_original_motor.coef_)

# Train and get feature importance for the log-transformed data for motor_updrs
lr_log_motor = LinearRegression()
lr_log_motor.fit(X_train_log, y_train_log)
feature_importance_log_motor = abs(lr_log_motor.coef_)

# Train and get feature importance for the Gaussian-transformed data for motor_updrs
lr_gaussian_motor = LinearRegression()
lr_gaussian_motor.fit(X_train_gaussian, y_train_gaussian)
feature_importance_gaussian_motor = abs(lr_gaussian_motor.coef_)

# Now, for total_updrs:
# Train and get feature importance for the original data
lr_original_total = LinearRegression()
lr_original_total.fit(X_train_motor, y_train_total)
feature_importance_original_total = abs(lr_original_total.coef_)

# Train and get feature importance for the log-transformed data
lr_log_total = LinearRegression()
lr_log_total.fit(X_train_log, y_train_log_total)
feature_importance_log_total = abs(lr_log_total.coef_)

# Train and get feature importance for the Gaussian-transformed data
lr_gaussian_total = LinearRegression()
lr_gaussian_total.fit(X_train_gaussian, y_train_gaussian_total)
feature_importance_gaussian_total = abs(lr_gaussian_total.coef_)

print("Feature importances for original data (motor_updrs):", feature_importance_original_motor)
print("Feature importances for log-transformed data (motor_updrs):", feature_importance_log_motor)
print("Feature importances for Gaussian-transformed data (motor_updrs):", feature_importance_gaussian_motor)
print("Feature importances for original data (total_updrs):", feature_importance_original_total)
print("Feature importances for log-transformed data (total_updrs):", feature_importance_log_total)
print("Feature importances for Gaussian-transformed data (total_updrs):", feature_importance_gaussian_total)

# Summary statistics for feature importances
print("Max importance (Gaussian-transformed, motor_updrs):", np.max(feature_importance_gaussian_motor))
print("Min importance (Gaussian-transformed, motor_updrs):", np.min(feature_importance_gaussian_motor))
print("Mean importance (Gaussian-transformed, motor_updrs):", np.mean(feature_importance_gaussian_motor))
print("Max importance (Gaussian-transformed, total_updrs):", np.max(feature_importance_gaussian_total))
print("Min importance (Gaussian-transformed, total_updrs):", np.min(feature_importance_gaussian_total))
print("Mean importance (Gaussian-transformed, total_updrs):", np.mean(feature_importance_gaussian_total))
print("Max importance (log-transformed, motor_updrs):", np.max(feature_importance_log_motor))
print("Min importance (log-transformed, motor_updrs):", np.min(feature_importance_log_motor))
print("Mean importance (log-transformed, motor_updrs):", np.mean(feature_importance_log_motor))
print("Max importance (log-transformed, total_updrs):", np.max(feature_importance_log_total))
print("Min importance (log-transformed, total_updrs):", np.min(feature_importance_log_total))
print("Mean importance (log-transformed, total_updrs):", np.mean(feature_importance_log_total))



# Continue with the merging approach to consolidate the feature importances:
df_original = pd.DataFrame({'Feature': X.columns, 'Importance_Original': feature_importance_original_motor})
df_original = pd.DataFrame({'Feature': X.columns, 'Importance_Original': feature_importance_original_total})
df_log = pd.DataFrame({'Feature': X_log_reduced.columns, 'Importance_Log': feature_importance_log_motor})
df_log = pd.DataFrame({'Feature': X_log_reduced.columns, 'Importance_Log': feature_importance_log_total})
df_gaussian = pd.DataFrame({'Feature': X.columns, 'Importance_Gaussian': feature_importance_gaussian_motor})
df_gaussian = pd.DataFrame({'Feature': X.columns, 'Importance_Gaussian': feature_importance_gaussian_total})


# Merge the DataFrames based on the 'Feature' column
merged_df = df_original.merge(df_log, on='Feature', how='outer')
merged_df = merged_df.merge(df_gaussian, on='Feature', how='outer')

print(merged_df)

# VISUALIZATION

# Histograms for target variables
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['motor_updrs'], bins=30, kde=True)
plt.title('Distribution of motor_updrs')

plt.subplot(1, 2, 2)
sns.histplot(data['total_updrs'], bins=30, kde=True)
plt.title('Distribution of total_updrs')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Residuals Plot (using the Gaussian transformed data as an example)
model = LinearRegression()
model.fit(X_train, y_train_motor)
y_pred = model.predict(X_test)
residuals = y_test_motor - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()

# Actual vs. Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_gaussian, y_test_gaussian, alpha=0.5)
plt.plot([min(y_test_gaussian), max(y_test_gaussian)], [min(y_test_gaussian), max(y_test_gaussian)], color='red') # y=x line
plt.xlabel('Actual motor_updrs')
plt.ylabel('Predicted motor_updrs')
plt.title('Actual vs Predicted for motor_updrs (Gaussian-transformed data)')
plt.grid(True)
plt.show()



# Q-Q Plot
plt.figure(figsize=(8, 6))
scipy_stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(y='Feature', x='Importance', data=feature_importance_df)
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.show()

# Visualize outliers using box plots
plt.figure(figsize=(15, 10))
sns.boxplot(data=features)
plt.show()



# Function to plot scatter plots for given feature against motor_updrs and total_updrs
def plot_scatter(feature):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    sns.scatterplot(data=data, x=feature, y="motor_updrs", ax=axes[0])
    axes[0].set_title(f"{feature} vs. motor_updrs")
    
    sns.scatterplot(data=data, x=feature, y="total_updrs", ax=axes[1])
    axes[1].set_title(f"{feature} vs. total_updrs")
    
    plt.tight_layout()
    plt.show()

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Exclude target columns from the features to plot
features_to_plot = [feature for feature in numeric_data.columns if feature not in ["motor_updrs", "total_updrs"]]

# Plot scatter plots for all numeric features
for feature in features_to_plot:
    plot_scatter(feature)




# Density plots for all numeric features
for column in numeric_data.columns:
    sns.kdeplot(numeric_data[column], fill=True)
    plt.title(f'Density plot for {column}')
    plt.show()