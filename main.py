import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_patients = 500
# Generate synthetic patient data
data = {
    'Age': np.random.randint(30, 80, size=num_patients),
    'Income': np.random.randint(20000, 150000, size=num_patients),
    'Insurance': np.random.choice(['Private', 'Medicare', 'Medicaid', 'None'], size=num_patients),
    'ChronicCondition': np.random.choice(['Yes', 'No'], size=num_patients, p=[0.3, 0.7]),
    'Transportation': np.random.choice(['Car', 'Public Transport', 'No Car'], size=num_patients, p=[0.6, 0.3, 0.1]),
    'Readmitted': np.random.choice([0, 1], size=num_patients, p=[0.8, 0.2]) # 0: Not Readmitted, 1: Readmitted
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Insurance', 'ChronicCondition', 'Transportation'], drop_first=True)
# --- 3. Predictive Modeling ---
# Split data into features (X) and target (y)
X = df.drop('Readmitted', axis=1)
y = df['Readmitted']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Logistic Regression model (a simple model for demonstration)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Readmitted', 'Readmitted'], 
            yticklabels=['Not Readmitted', 'Readmitted'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
feature_importances = pd.Series(model.coef_[0], index=X.columns)
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Coefficient Magnitude')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
output_filename = 'feature_importances.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")