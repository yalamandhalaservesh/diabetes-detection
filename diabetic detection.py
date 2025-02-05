import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Correct raw URL for the dataset
url = "https://raw.githubusercontent.com/yalamandhalaservesh/diabetes-detection/main/diabetes_detection_dataset.csv"

# Read the CSV file
df = pd.read_csv(url)

# Display the first few rows of the dataframe
print(df.head())

# Dropping the 'name' column as it's non-numeric and not needed for prediction
df = df.drop(columns=['name'], errors='ignore')

# Splitting features and target variable
X = df.drop(columns=['Outcome'], errors='ignore')  # Corrected to X for feature set
y = df['Outcome']  # Target variable is 'Outcome'

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)
