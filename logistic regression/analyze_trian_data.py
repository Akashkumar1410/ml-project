import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the training data with replaced outliers
train_data_with_no_outliers = pd.read_csv('modified_dataset.csv')

# Identify the features and target variable
X_train_no_outliers = train_data_with_no_outliers.drop('Loan_Status', axis=1)
y_train_no_outliers = train_data_with_no_outliers['Loan_Status']

# Convert categorical variables to numerical using one-hot encoding
X_train_no_outliers = pd.get_dummies(X_train_no_outliers)

# Split the training data into training and validation sets
X_train_no_outliers, X_val_no_outliers, y_train_no_outliers, y_val_no_outliers = train_test_split(
    X_train_no_outliers, y_train_no_outliers, test_size=0.2, random_state=42
)

# Apply StandardScaler to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_no_outliers)
# print(" Print ",X_train_scaled)
X_val_scaled = scaler.transform(X_val_no_outliers)
# print(X_val_scaled)
# Load the original testing data
original_test_data = pd.read_csv('transformed_test_data.csv')

# Identify the features in the testing data
X_test_original = pd.get_dummies(original_test_data)  # Assuming similar preprocessing as training data

# Apply StandardScaler to the testing features
X_test_scaled = scaler.transform(X_test_original)

# Create and train the logistic regression model
model_no_outliers_scaled = LogisticRegression()
model_no_outliers_scaled.fit(X_train_scaled, y_train_no_outliers)

# Make predictions on the validation set
y_pred_no_outliers_scaled = model_no_outliers_scaled.predict(X_val_scaled)

# Evaluate the model performance on the validation set
accuracy_no_outliers_scaled = accuracy_score(y_val_no_outliers, y_pred_no_outliers_scaled)
print(f"Accuracy on the scaled validation set: {accuracy_no_outliers_scaled:.2%}")

# Make predictions on the scaled original test set
test_predictions_scaled = model_no_outliers_scaled.predict(X_test_scaled)


# Optional: Save the predictions to a CSV file
predictions_df_scaled = pd.DataFrame({'Loan_Status_Prediction': test_predictions_scaled})
# predictions_df_scaled.to_csv('test_predictions_scaled.csv', index=False)


# Make predictions on the scaled validation set
y_pred_no_outliers_scaled = model_no_outliers_scaled.predict(X_val_scaled)

# Calculate and print precision, recall, and F1-score
precision_no_outliers_scaled = precision_score(y_val_no_outliers, y_pred_no_outliers_scaled)
recall_no_outliers_scaled = recall_score(y_val_no_outliers, y_pred_no_outliers_scaled)
f1_no_outliers_scaled = f1_score(y_val_no_outliers, y_pred_no_outliers_scaled)

print(f'Precision (scaled): {precision_no_outliers_scaled}')
print(f'Recall (scaled): {recall_no_outliers_scaled}')
print(f'F1-score (scaled): {f1_no_outliers_scaled}')

# Calculate and print confusion matrix
conf_matrix_no_outliers_scaled = confusion_matrix(y_val_no_outliers, y_pred_no_outliers_scaled)
print('Confusion Matrix (scaled):')
print(conf_matrix_no_outliers_scaled)

# Additional classification report
print('Classification Report (scaled):')
print(classification_report(y_val_no_outliers, y_pred_no_outliers_scaled))



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get the probability estimates for the positive class (class 1)
y_prob_no_outliers_scaled = model_no_outliers_scaled.predict_proba(X_val_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_val_no_outliers, y_prob_no_outliers_scaled)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
