import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

pip install xgboost scikit-learn

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/credit_risk_dataset.csv')

df.head()

df.tail()

df.describe()

df.info()

df.shape

from sklearn.preprocessing import OneHotEncoder

# Assuming your dataset is stored in a DataFrame named 'df'
X = df[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']]

y = df['loan_status']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you already have X_train, X_test, y_train, y_test, and the trained XGBoost model

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Let's create a new example for prediction
new_example = pd.DataFrame({
    'person_age': [28],
    'person_income': [60000],
    'person_home_ownership': ['OWN'],
    'person_emp_length': [3.0],
    'loan_intent': ['PERSONAL'],
    'loan_grade': ['B'],
    'loan_amnt': [12000],
    'loan_int_rate': [10.5],
    'loan_percent_income': [0.2],
    'cb_person_default_on_file': ['N'],
    'cb_person_cred_hist_length': [5]
})

# One-hot encode categorical variables
new_example_encoded = pd.get_dummies(new_example, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

# Ensure new_example_encoded has all the necessary columns
missing_columns = set(X_train.columns) - set(new_example_encoded.columns)
for col in missing_columns:
    new_example_encoded[col] = 0

# Reorder the columns to match the order during training
new_example_encoded = new_example_encoded[X_train.columns]

# Make predictions
new_example_pred_proba = model.predict_proba(new_example_encoded)[:, 1]

# Print the predicted probability
print(f'Predicted Probability of Default: {new_example_pred_proba[0]:.4f}')
