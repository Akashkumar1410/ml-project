
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the training data
train_data = pd.read_csv("train_no_outliers.csv")

# Load the test data
test_data = pd.read_csv("your_test_file.csv")

# Assuming "Loan_Status" is the target variable
X_train = train_data.drop("Loan_Status", axis=1)
y_train = train_data["Loan_Status"]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a perceptron model
perceptron_model = Perceptron(max_iter=1000, random_state=42)

# Train the model on the training data
perceptron_model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = perceptron_model.predict(X_val)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, val_predictions)
conf_matrix = confusion_matrix(y_val, val_predictions)

print("Accuracy on validation set:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Now, use the trained model to make predictions on the test data
test_predictions = perceptron_model.predict(test_data)

# You can use test_predictions for further analysis or submission
