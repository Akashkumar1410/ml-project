from sklearn.metrics import hinge_loss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the training data
train_data = pd.read_csv("train_no_outliers.csv")

# Load the test data
test_data = pd.read_csv("transformed_test_data.csv")

# Assuming "Loan_Status" is the target variable
X_train = train_data.drop("Loan_Status", axis=1)
y_train = train_data["Loan_Status"]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and validation data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# print the acaled data
# print(X_train_scaled)

# Create a perceptron model
perceptron_model = Perceptron(max_iter=1000, random_state=42)

# Train the model on the scaled training data
perceptron_model.fit(X_train_scaled, y_train)
# wights and baises of the data
weights = perceptron_model.coef_
bias = perceptron_model.intercept_

print("Weights:", weights)
print("Bias:", bias)



# Make predictions on the scaled validation set
val_predictions = perceptron_model.predict(X_val_scaled)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, val_predictions)
conf_matrix = confusion_matrix(y_val, val_predictions)

print("Accuracy on validation set:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Now, use the trained model to make predictions on the scaled test data
test_data_scaled = scaler.transform(test_data)  # Make sure to scale the test data using the same scaler
test_predictions = perceptron_model.predict(test_data_scaled)

# You can use test_predictions for further analysis or submission
# Print weights and bias with feature names
feature_names = X_train.columns
weights_dict = dict(zip(feature_names, weights.flatten()))

print("Weights:")
for feature, weight in weights_dict.items():
    print(f"{feature}: {weight}")

print("Bias:", bias)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calculate precision, recall, and F1 score
precision = precision_score(y_val, val_predictions_best_model)
recall = recall_score(y_val, val_predictions_best_model)
f1 = f1_score(y_val, val_predictions_best_model)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate AUC-ROC
raw_scores_best_model = best_perceptron_model.decision_function(X_val_scaled)
auc_roc = roc_auc_score(y_val, raw_scores_best_model)

print("AUC-ROC:", auc_roc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_val, raw_scores_best_model)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

