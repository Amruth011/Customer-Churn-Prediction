# %%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load the data
df = pd.read_csv("customer_churn_data.csv")

# %%
# Data Cleaning: Drop rows with missing values
df.dropna(inplace=True)

# %%
# Define features (x) and target (y) for the model
# Feature order: 'Age', 'Gender', 'Tenure', 'MonthlyCharges'
x = df[["Age", "Gender", "Tenure", "MonthlyCharges"]]
y = df["Churn"]

# %%
# Convert categorical features to numerical (Gender: Female=1, Male=0; Churn: Yes=1, No=0)
x["Gender"] = x["Gender"].apply(lambda val: 1 if val == "Female" else 0)
y = y.apply(lambda val: 1 if val == "Yes" else 0)

# %%
# Split data into 80% training and 20% testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
# Scale numerical features and save the scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
joblib.dump(scaler, "scaler.pkl")

# %%
# Function to print the accuracy score
def model_performance(predictions):
    """Calculates and prints the accuracy score of the model."""
    print(f"Accuracy Score on model is {accuracy_score(y_test, predictions):.4f}")

# %%
# --- Model Training and Evaluation ---

# Train and evaluate Logistic Regression
print("--- Logistic Regression ---")
log_model = LogisticRegression()
log_model.fit(x_train_scaled, y_train)
y_pred_lr = log_model.predict(x_test_scaled)
model_performance(y_pred_lr)

# %%
# Train and evaluate KNeighborsClassifier using GridSearchCV for best parameters
print("\n--- KNeighborsClassifier ---")
param_grid_knn = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(x_train_scaled, y_train)
print("Best Params:", grid_knn.best_params_)
y_pred_knn = grid_knn.predict(x_test_scaled)
model_performance(y_pred_knn)

# %%
# Train and evaluate Support Vector Classifier (SVC) using GridSearchCV
print("\n--- Support Vector Classifier (SVC) ---")
svm = SVC()
param_grid_svc = {"C": [0.01, 0.1, 0.5, 1], "kernel": ["linear", "rbf", "poly"]}
grid_svc = GridSearchCV(svm, param_grid_svc, cv=5)
grid_svc.fit(x_train_scaled, y_train)
print("Best Params:", grid_svc.best_params_)
y_pred_svc = grid_svc.predict(x_test_scaled)
model_performance(y_pred_svc)

# %%
# Train and evaluate Decision Tree Classifier using GridSearchCV
print("\n--- Decision Tree Classifier ---")
param_grid_dt = {'criterion': ["gini", "entropy"], 'splitter': ['best', 'random'], 'max_depth': [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_tree.fit(x_train_scaled, y_train)
print("Best Params:", grid_tree.best_params_)
y_pred_dt = grid_tree.predict(x_test_scaled)
model_performance(y_pred_dt)

# %%
# Train and evaluate Random Forest Classifier using GridSearchCV
print("\n--- Random Forest Classifier ---")
rfc_model = RandomForestClassifier(random_state=42)
param_grid_rfc = {"n_estimators": [32, 64, 128, 256], "max_features": [2, 3, 4], "bootstrap": [True, False]}
grid_rfc = GridSearchCV(rfc_model, param_grid_rfc, cv=5)
grid_rfc.fit(x_train_scaled, y_train)
print("Best Params:", grid_rfc.best_params_)
y_pred_rfc = grid_rfc.predict(x_test_scaled)
model_performance(y_pred_rfc)

# %%
# Export the best model (SVC) and scaler to pkl files for the app
best_model = grid_svc.best_estimator_
joblib.dump(best_model, "model.pkl")
print("\nBest model (SVC) and scaler have been exported to 'model.pkl' and 'scaler.pkl'.")
