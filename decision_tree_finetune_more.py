from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Load data
# This code depends on your specific situation. For example:
# from sklearn.datasets import load_iris
# X, Y = load_iris(return_X_y=True)

df=pd.read_csv('ACME-HappinessSurvey2020.csv') # creates the pandas dataframe from the CSV file

# Split data into training and validation set
X = df.drop('Y', axis=1)  # This will create a new DataFrame X that includes everything from df except 'Y'
Y = df['Y']  # This will create a Series Y which includes only 'Y' from df
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create an instance of the decision tree
dt = DecisionTreeClassifier(random_state=42)

# Train the decision tree model
dt.fit(X_train, Y_train)

# Predict the training and validation data
Y_train_pred_dt = dt.predict(X_train)
Y_val_pred_dt = dt.predict(X_val)

# Print the training and validation accuracies
print("Training accuracy Base Decision Tree: ", accuracy_score(Y_train, Y_train_pred_dt))
print("Validation accuracy Base Decision Tree: ", accuracy_score(Y_val, Y_val_pred_dt))

# Fine-tune the decision tree model
param_grid_dt = {
    'criterion': ["gini", "entropy"],
    'max_depth': list(range(1, 21)),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 5),
    'max_features': ["sqrt", "log2", None]
}


grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')

grid_search_dt.fit(X_train, Y_train)

print("Best parameters Decision Tree: ", grid_search_dt.best_params_)

# Predict with the fine-tuned model
optimal_params_dt = grid_search_dt.best_params_

dt_optimized = DecisionTreeClassifier(**optimal_params_dt, random_state=42)
dt_optimized.fit(X_train, Y_train)

Y_train_pred_dt_finetune = dt_optimized.predict(X_train)
Y_val_pred_dt_finetune = dt_optimized.predict(X_val)

print("Training accuracy Optimized Decision Tree: ", accuracy_score(Y_train, Y_train_pred_dt_finetune))
print("Validation accuracy Optimized Decision Tree: ", accuracy_score(Y_val, Y_val_pred_dt_finetune))