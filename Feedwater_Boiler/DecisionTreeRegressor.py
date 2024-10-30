import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
data = pd.read_excel("dara.xlsx")
print("Original Data Shape:", data.shape)
print(data.head())

# Convert all columns to numeric, coercing errors, and drop non-numeric rows
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)
print("Shape after dropping non-numeric rows:", data.shape)

# Fill any remaining missing values with forward fill
data.ffill(inplace=True)

# Define feature columns
features = [
    '03BAA20FE351XQ50.UNIT3@NET3', '03LBQ30AA803XZ01.UNIT3@NET3', '03LBQ30CT003XT01.UNIT3@NET3',
    '03LAD10CP001XP01.UNIT3@NET3', '03LAD10CL001-SEL.UNIT3@NET3', '03LCH30CT001XT01.UNIT3@NET3',
    '03LCH30CG551XZ01.UNIT3@NET3', '03LCH30CG552XZ01.UNIT3@NET3', '03LAB20CT001-SEL.UNIT3@NET3',
    '03LAB20CP002-SEL.UNIT3@NET3', '03LAB22CP001XP01.UNIT3@NET3', '03LAB22CT001XT01.UNIT3@NET3'
]

# Create lagged features
for feature in features:
    for lag in range(1, 6):
        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)

print("Shape after creating lagged features:", data.shape)

# Drop any new NaNs due to lagging
data.dropna(inplace=True)
print("Shape after dropping NaNs:", data.shape)

# Separate features and target
X = data[[f'{feature}_lag_{i}' for feature in features for i in range(1, 6)]]
y = data['03BAA20FE351XQ50.UNIT3@NET3']  # Target column

# Print shapes for debugging
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Ensure there are samples before splitting
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("No data available to split. Check your data loading process.")

# Train-test split (75% training, 25% testing)
if X.shape[0] < 4:  # Adjust threshold as necessary
    print("Not enough data to split. Using the entire dataset for training.")
    y_pred = y  # Assuming you'd want to make predictions on the same data
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    # Train the Decision Tree Regressor
    tree_regressor = DecisionTreeRegressor(max_depth=5)
    tree_regressor.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = tree_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

# Display predictions alongside actual values
predictions = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=y.index)
print(predictions)
