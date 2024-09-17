# READ ME: 
# MY VITAL FUNCTIONS FOR THE CASE STUDIES
```PY
def select_correlated_iv_features(data, feature_list, iv_dataframe):
    correlation_matrix = data[feature_list].corr()

    # Creating an upper triangle mask
    upper_triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Setting up the plot
    fig, ax = plt.subplots(figsize=(20, 20))
    color_map = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plotting the heatmap with the mask applied
    sns.heatmap(correlation_matrix, cmap=color_map, mask=upper_triangle_mask, vmax=1, center=0.5,
                square=True, linewidths=.5, cbar_kws={"shrink": .6}, annot=True)

    plt.title("Pearson Correlation", fontsize=10)
    plt.show()

    # Resetting the index for easier processing
    correlation_matrix = correlation_matrix.reset_index()
    vars_to_remove = []

    # Identifying variables to remove based on correlation and IV values
    for column in correlation_matrix.columns[1:]:
        for i in range(len(correlation_matrix[column])):
            variable_i = correlation_matrix["index"][i]
            if (abs(correlation_matrix[column][i]) > 0.6) and (variable_i != column):
                iv_var_i = iv_dataframe.loc[iv_dataframe["variable"] == variable_i, "iv"].item()
                iv_var_col = iv_dataframe.loc[iv_dataframe["variable"] == column, "iv"].item()
                remove_variable = column if iv_var_col < iv_var_i else variable_i
                if remove_variable not in vars_to_remove:
                    vars_to_remove.append(remove_variable)

    # Selecting variables based on IV threshold
    selected_features = [var for var in feature_list if var not in vars_to_remove]
    selected_features = list(iv_dataframe[(iv_dataframe['variable'].isin(selected_features)) & 
                                          (iv_dataframe['iv'] > 0.07)]['variable'])
    
    return selected_features, vars_to_remove


def auto_binning_process(data, numerical_vars, categorical_vars):
    binning_results = []
    binning_tables = {}
    
    # Initializing the binning process
    binning_process = BinningProcess(variable_names=numerical_vars + categorical_vars, 
                                     categorical_variables=categorical_vars, 
                                     max_n_bins=5, split_digits=4)
    
    # Fitting the binning process to the data
    binning_process.fit(data, data['Converted'])
    
    # Processing each variable for binning
    for variable in data.columns:
        optimal_binning = binning_process.get_binned_variable(variable)
        binning_table = optimal_binning.binning_table.build()
        binning_table = binning_table[binning_table['Event'] > 0]
        
        # Storing the binning table
        binning_tables[variable] = binning_table
        
        # Appending the IV and binning information to the results
        binning_results.append([variable,
                                binning_table.loc['Totals', 'IV'],
                                binning_table.shape[0] - 1,
                                binning_table.loc[binning_table.drop('Totals')['Count'].idxmax(), 'Bin'],
                                binning_table.drop('Totals')['Count'].max()
                               ])
    
    # Creating a dataframe for the IV values
    iv_dataframe = pd.DataFrame(binning_results, columns=['variable', 'iv', 'unique_bin', 'top_bin', 'freq_bin'])
    
    # Concatenating all binning tables into a single dataframe
    concatenated_binning_table = pd.concat(binning_tables, axis=0)
    
    return binning_process, iv_dataframe, concatenated_binning_table
```
# XGBOOST METHOD
```PY
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Training and testing datasets
x_trained_data = transformed_trained_data
y_train_labels = trained_data['Converted']
x_test_data = transformed_test_data
y_test_labels = test_data['Converted']

# Hyperparameter search space
xgb_hyperparams = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Function to clean column names
def sanitize_column_names(df):
    df.columns = df.columns.str.replace('[', '_', regex=False)
    df.columns = df.columns.str.replace(']', '_', regex=False)
    df.columns = df.columns.str.replace('<', '_', regex=False)
    df.columns = df.columns.str.replace('>', '_', regex=False)
    df.columns = df.columns.astype(str)
    return df

# Clean column names for training and testing data
x_trained_data = sanitize_column_names(x_trained_data)
x_test_data = sanitize_column_names(x_test_data)

# Initialize the XGBoost classifier
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Perform RandomizedSearchCV to find the best hyperparameters
xgb_random_search = RandomizedSearchCV(xgboost_model, xgb_hyperparams, n_iter=50, cv=5, random_state=42, n_jobs=-1)
xgb_random_search.fit(x_trained_data, y_train_labels)

# Retrieve the best model after hyperparameter tuning
best_xgb_model = xgb_random_search.best_estimator_

# Make predictions on the test data
y_test_predictions = best_xgb_model.predict(x_test_data)

# Evaluate the model performance
model_accuracy = accuracy_score(y_test_labels, y_test_predictions)
classification_report_dict = classification_report(y_test_labels, y_test_predictions, output_dict=True)

print(f"Accuracy: {model_accuracy}")

# Get prediction probabilities for calculating ROC curve and AUC score
y_test_pred_probabilities = best_xgb_model.predict_proba(x_test_data)[:, 1]

# Calculate false positive rate (FPR), true positive rate (TPR), and thresholds for ROC curve
false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_test_labels, y_test_pred_probabilities)
roc_auc_score_value = roc_auc_score(y_test_labels, y_test_pred_probabilities)

print("AUC Score:", roc_auc_score_value)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(false_positive_rate, true_positive_rate, color='blue', label=f'ROC Curve (AUC = {roc_auc_score_value:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```
# XGBOOST THRESHOLD OPTIMIZATION
```PY
# Define the range of thresholds to evaluate
threshold_values = np.arange(0.4, 0.9, 0.05)

# Get the predicted probabilities from the best model
y_test_pred_probabilities = best_xgb_model.predict_proba(x_test_data)[:, 1]

# Initialize variables to store the best threshold and the corresponding accuracy
optimal_threshold = 0.0
highest_accuracy = 0.0

# Iterate through each threshold value
for threshold in threshold_values:
    # Apply the threshold to get binary predictions
    y_threshold_predictions = (y_test_pred_probabilities >= threshold).astype(int)
    
    # Calculate accuracy for the current threshold
    current_accuracy = accuracy_score(y_test_labels, y_threshold_predictions)
    
    # Update the best threshold if the current accuracy is higher
    if current_accuracy > highest_accuracy:
        highest_accuracy = current_accuracy
        optimal_threshold = threshold

# Output the best threshold and corresponding accuracy
print(f"Optimal Threshold: {optimal_threshold}")
print(f"Highest Accuracy: {highest_accuracy}")

```
