from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Load the dataset
print("Loading the dataset...")
dataset = np.loadtxt(r"EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]

# Split the dataset
print("Splitting the dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)),
    ('svr', SVR())
]

# Define meta-model
meta_model = LinearRegression()

# Define stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Wrap with MultiOutputRegressor for multi-output regression
multi_stacking_model = MultiOutputRegressor(stacking_model)

# Train the model
print("Training the model...")
multi_stacking_model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = multi_stacking_model.predict(X_val)

# Evaluate the model
print("Evaluating the model...")
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")


# Save the model
print("Saving the model...")
with open('multi_stacking_model.pkl', 'wb') as f:
    pickle.dump(multi_stacking_model, f)

print("Model saved successfully.")

