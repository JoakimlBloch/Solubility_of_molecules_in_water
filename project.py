#-------------------------------------------------------------------------------------------------------------------------#

# Testing molecules solubility in water with linear regression and random forest algorithms
# logS is the logarithmic solubility value

## logS > 0: The molecule is very soluble in water (solubility greater than 1 mol/L).
## logS = 0: The molecule has a solubility of 1 mol/L.
## logS < 0: The molecule is less soluble in water.
## For example, a logS of -1 means the solubility is 0.1 mol/L.
## A logS of -2 means the solubility is 0.01 mol/L, and so on.

#-------------------------------------------------------------------------------------------------------------------------#

# Loading in data set

import pandas as pd

# Dataframe - GitHub dataprofessor solubility in molecules
df = pd.read_csv('delaney_solubility_with_descriptors.csv')

# Display the dataframe
## print(df)

# -------------------------------------------------------------------------------------------------------------------------#

# Data preparation - seperating the Y dependent variable (logS) from the X independent variables (col. 1-4)
y = df['logS']

# Dropping the 'logS' column from the dataframe and displaying it in col. mode with axis=1 instead of row mode with axis=0
x = df.drop('logS', axis=1)

# Data splitting - splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# -------------------------------------------------------------------------------------------------------------------------#

# Model building - linear regression
from sklearn.linear_model import LinearRegression

# Training the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Applying the model to make a prediction on the training set specified with the linear regression algorithm
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

#-------------------------------------------------------------------------------------------------------------------------#

# Model perfomance - evaluation of the linear regression model with the actual values
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

""" print('LR MSE (Train):', lr_train_mse)
print('LR R2 (Train):', lr_train_r2)
print('LR MSE (Test):', lr_test_mse)
print('LR R2 (Test):', lr_test_r2) """

# Print model results
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = (['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])

### print(lr_results.to_string(index=False))

#-------------------------------------------------------------------------------------------------------------------------#

# Model building - w/ Random Forest
from sklearn.ensemble import RandomForestRegressor

# Training model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Applying the model
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Evaluate model performance
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = (['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])

### print(lr_results.to_string(index=False))

#-------------------------------------------------------------------------------------------------------------------------#

# Comparing the two models, axis=0 for row-wise concatenation
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
## print(df_models)

#-------------------------------------------------------------------------------------------------------------------------#

# Data visualization - comparing the actual vs. predicted logS values
import matplotlib.pyplot as plt
import numpy as np

# Making of graph with x and y axis
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), "#F8766D")
plt.xlabel('Actual logS')
plt.ylabel('Predicted logS')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------#

# Predicting new data

new_data = pd.DataFrame([[1.43870, 129.670, 4.0, 0.153783]])  # New input features
predicted_logS_lr = lr.predict(new_data)
predicted_logS_rf = rf.predict(new_data)

print(f'Predicted logS (Linear Regression): {predicted_logS_lr}')
print(f'Predicted logS (Random Forest): {predicted_logS_rf}')