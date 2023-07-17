import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('ACME-HappinessSurvey2020.csv')

for col in df.columns:
    if col.startswith('X'):
        df[col] = pd.cut(df[col], bins=[0, 2, 5], labels=[0, 1])

# Convert categorical data back to integers
df['X1'] = df['X1'].astype(int)
df['X5'] = df['X5'].astype(int)

# Create interaction term
df['X1_X5'] = df['X1'] * df['X5']

# Display the dataframe
print(df.head())

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()

# Split the data into train and test
X = df.drop('Y', axis=1)
Y = df['Y']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Fit the model
dt.fit(X_train, Y_train)

# Predict the data
Y_train_pred = dt.predict(X_train)
Y_val_pred = dt.predict(X_val)

# Print the accuracy
print("Training accuracy Base Decision Tree: ", accuracy_score(Y_train, Y_train_pred))
print("Validation accuracy Base Decision Tree: ", accuracy_score(Y_val, Y_val_pred))

# Fine-tune the model
param_grid_dt = {'max_depth': list(range(1, 21))}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_search_dt.fit(X_train, Y_train)

# Print the best parameters
print("Best parameters Decision Tree: ", grid_search_dt.best_params_)

# Train the model with the best parameters
dt_optimized = DecisionTreeClassifier(max_depth=grid_search_dt.best_params_['max_depth'], random_state=42)
dt_optimized.fit(X_train, Y_train)

# Predict the data
Y_train_pred_dt_finetune = dt_optimized.predict(X_train)
Y_val_pred_dt_finetune = dt_optimized.predict(X_val)

# Print the accuracy
print("Training accuracy Optimized Decision Tree: ", accuracy_score(Y_train, Y_train_pred_dt_finetune))
print("Validation accuracy Optimized Decision Tree: ", accuracy_score(Y_val, Y_val_pred_dt_finetune))
