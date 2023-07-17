import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

df=pd.read_csv('ACME-HappinessSurvey2020.csv') # creates the pandas dataframe from the CSV file

#below gives a summary of data
print("DATASET GENERAL INFO:")
print(df.head()) #first 5 rows
df.info() #general info/nulls in data
print(df.describe()) #statistics
print("END")

#look for balance in dataset for target/prediction variable
print('BALANCE:')
print(df['Y'].value_counts())

df.hist(bins=50, figsize=(20,15))
plt.show()
#correlation matrix for feature correlation prior to actually applying an ML model
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
print('Correlation Val:')
print(corr_matrix)
plt.show()
#target and validation dataset split
X = df.drop('Y', axis=1)  # This will create a new DataFrame X that includes everything from df except 'Y'
Y = df['Y']  # This will create a Series Y which includes only 'Y' from df
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# Create an instance of the logistic regression
log_reg = LogisticRegression(random_state=42)
# Train the logistic regression model
log_reg.fit(X_train, Y_train)
Y_train_pred = log_reg.predict(X_train)
Y_val_pred = log_reg.predict(X_val)


#we will now fine tune our logisitc regression model using gridsearchCV below (adjust hyperparam)
# Define the parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Create a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)
optimal_C = grid_search.best_params_['C']
print(optimal_C)
#optimal_solver = grid_search.best_params_['solver']
#log_reg_optimized = LogisticRegression(C=optimal_C, solver=optimal_solver, random_state=42)
log_reg_optimized = LogisticRegression(C=optimal_C, random_state=42)
log_reg_optimized.fit(X_train, Y_train)
Y_train_pred_finetune = log_reg_optimized.predict(X_train)
Y_val_pred_finetune = log_reg_optimized.predict(X_val)

# Print the best score
print("Best cross-validation score: ", grid_search.best_score_)
print("Training accuracy Log_Reg Base Model: ", accuracy_score(Y_train, Y_train_pred))
print("Validation accuracy Log_Reg Base Model: ", accuracy_score(Y_val, Y_val_pred))
print("Training accuracy Optimized Log_reg: ", accuracy_score(Y_train, Y_train_pred_finetune))
print("Validation accuracy Optimized Log_reg: ", accuracy_score(Y_val, Y_val_pred_finetune))

# Create an instance of the SVM
svm = SVC(random_state=42)

# Train the SVM model
svm.fit(X_train, Y_train)

# Predict the training and validation data
Y_train_pred_svm = svm.predict(X_train)
Y_val_pred_svm = svm.predict(X_val)

# Print the training and validation accuracies
print("Training accuracy Base SVM: ", accuracy_score(Y_train, Y_train_pred_svm))
print("Validation accuracy Base SVM: ", accuracy_score(Y_val, Y_val_pred_svm))

# Fine-tune the SVM model
param_grid_svm = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)

grid_search_svm.fit(X_train, Y_train)

print("Best parameters SVM: ", grid_search_svm.best_params_)

# Predict with the fine-tuned model
optimal_C_svm = grid_search_svm.best_params_['C']
optimal_gamma_svm = grid_search_svm.best_params_['gamma']

svm_optimized = SVC(C=optimal_C_svm, gamma=optimal_gamma_svm, random_state=42)
svm_optimized.fit(X_train, Y_train)

Y_train_pred_svm_finetune = svm_optimized.predict(X_train)
Y_val_pred_svm_finetune = svm_optimized.predict(X_val)

print("Training accuracy Optimized SVM: ", accuracy_score(Y_train, Y_train_pred_svm_finetune))
print("Validation accuracy Optimized SVM: ", accuracy_score(Y_val, Y_val_pred_svm_finetune))

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
param_grid_dt = {'max_depth': list(range(1, 21))}  # Here we are considering trees of depth 1 to 20

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)

grid_search_dt.fit(X_train, Y_train)

print("Best parameters Decision Tree: ", grid_search_dt.best_params_)

# Predict with the fine-tuned model
optimal_depth_dt = grid_search_dt.best_params_['max_depth']

dt_optimized = DecisionTreeClassifier(max_depth=optimal_depth_dt, random_state=42)
dt_optimized.fit(X_train, Y_train)

Y_train_pred_dt_finetune = dt_optimized.predict(X_train)
Y_val_pred_dt_finetune = dt_optimized.predict(X_val)

print("Training accuracy Optimized Decision Tree: ", accuracy_score(Y_train, Y_train_pred_dt_finetune))
print("Validation accuracy Optimized Decision Tree: ", accuracy_score(Y_val, Y_val_pred_dt_finetune))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
Y_train_pred_rf = rf.predict(X_train)
Y_val_pred_rf = rf.predict(X_val)

print("Training accuracy Base Random Forest: ", accuracy_score(Y_train, Y_train_pred_rf))
print("Validation accuracy Base Random Forest: ", accuracy_score(Y_val, Y_val_pred_rf))

# Hyperparameter tuning for Random Forest
param_grid_rf = {'n_estimators': [10, 50, 100, 200],
                 'max_depth': [None, 2, 4, 6, 8],
                 'min_samples_split': [2, 5, 10]}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, Y_train)

# Print the best parameters
print("Best parameters Random Forest: ", grid_search_rf.best_params_)

# Train the model with the best parameters
rf_optimized = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
rf_optimized.fit(X_train, Y_train)
Y_train_pred_rf_finetune = rf_optimized.predict(X_train)
Y_val_pred_rf_finetune = rf_optimized.predict(X_val)

print("Training accuracy Optimized Random Forest: ", accuracy_score(Y_train, Y_train_pred_rf_finetune))
print("Validation accuracy Optimized Random Forest: ", accuracy_score(Y_val, Y_val_pred_rf_finetune))


# Applying cross-validation to the Optimized Random Forest model (seemed to be the best balance between training and validation accuracy)
scores = cross_val_score(rf_optimized, X_train, Y_train, cv=5)
print("Cross-validation scores RF Optimized: ", scores)
print("Average cross-validation score RF Optimized: ", scores.mean())

#gradient boost model below
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, Y_train)
Y_train_pred_gb = gb.predict(X_train)
Y_val_pred_gb = gb.predict(X_val)

print("Training accuracy Base Gradient Boosting: ", accuracy_score(Y_train, Y_train_pred_gb))
print("Validation accuracy Base Gradient Boosting: ", accuracy_score(Y_val, Y_val_pred_gb))

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {'n_estimators': [10, 50, 100, 200],
                 'learning_rate': [0.001, 0.01, 0.1, 1, 10],
                 'max_depth': [1, 2, 3, 4, 5]}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5)
grid_search_gb.fit(X_train, Y_train)

# Print the best parameters
print("Best parameters Gradient Boosting: ", grid_search_gb.best_params_)

# Train the model with the best parameters
gb_optimized = GradientBoostingClassifier(**grid_search_gb.best_params_, random_state=42)
gb_optimized.fit(X_train, Y_train)
Y_train_pred_gb_finetune = gb_optimized.predict(X_train)
Y_val_pred_gb_finetune = gb_optimized.predict(X_val)

print("Training accuracy Optimized Gradient Boosting: ", accuracy_score(Y_train, Y_train_pred_gb_finetune))
print("Validation accuracy Optimized Gradient Boosting: ", accuracy_score(Y_val, Y_val_pred_gb_finetune))

#i find it weird that the optimized models above have an accuracy worse than the base model


#trying an ensemble method - its commented out becuase it didnt change accuracy much
#from sklearn.ensemble import VotingClassifier

#voting_clf = VotingClassifier(
   # estimators=[('lr', log_reg_optimized), ('svc', svm_optimized),
              #  ('gb', gb_optimized)], voting='hard')

#voting_clf.fit(X_train, Y_train)

#Y_train_pred_ensemble = voting_clf.predict(X_train)
#Y_val_pred_ensemble = voting_clf.predict(X_val)

#print("Training accuracy Ensemble: ", accuracy_score(Y_train, Y_train_pred_ensemble))
#print("Validation accuracy Ensemble: ", accuracy_score(Y_val, Y_val_pred_ensemble))
