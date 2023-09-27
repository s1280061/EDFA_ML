from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Load dataset
print("Loading dataset...")
dataset = np.loadtxt(r"EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split dataset
print("Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)# 割合0.2

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_val: {y_val.shape}")

# Create multiple RandomForestRegressor models with different parameters
print("Creating RandomForestRegressor models...")
model1 = RandomForestRegressor(n_estimators=50, random_state=0, min_samples_leaf=2, n_jobs=-1)
model2 = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_split=4, n_jobs=-1)
model3 = RandomForestRegressor(n_estimators=150, random_state=0, max_leaf_nodes=100, n_jobs=-1)

# Fit the models
print("Fitting models...")
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

print("Models fitted successfully.")

# Save models using pickle
with open('random_forest_model1.pkl', 'wb') as f:
    pickle.dump(model1, f)
with open('random_forest_model2.pkl', 'wb') as f:
    pickle.dump(model2, f)
with open('random_forest_model3.pkl', 'wb') as f:
    pickle.dump(model3, f)

print("Models saved successfully.")

# Make predictions
print("Making predictions...")
pred1 = model1.predict(X_val)
pred2 = model2.predict(X_val)
pred3 = model3.predict(X_val)

print(f"Shape of pred1: {pred1.shape}")
print(f"Shape of pred2: {pred2.shape}")
print(f"Shape of pred3: {pred3.shape}")

# Average predictions
print("Averaging predictions...")
final_pred = (pred1 + pred2 + pred3) / 3


print(f"Shape of final_pred: {final_pred.shape}")

rmse = np.sqrt(mean_squared_error(y_val, final_pred))
print(f"RMSE: {rmse}")


# Load models using pickle for future use
with open('random_forest_model1.pkl', 'rb') as f:
    loaded_model1 = pickle.load(f)
with open('random_forest_model2.pkl', 'rb') as f:
    loaded_model2 = pickle.load(f)
with open('random_forest_model3.pkl', 'rb') as f:
    loaded_model3 = pickle.load(f)

print("Models loaded successfully.")


import matplotlib.pyplot as plt

# Create multiple RandomForestRegressor models with different parameters
print("Creating RandomForestRegressor models...")
model1 = RandomForestRegressor(n_estimators=50, random_state=0, min_samples_leaf=2, n_jobs=-1)
model2 = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_split=4, n_jobs=-1)
model3 = RandomForestRegressor(n_estimators=150, random_state=0, max_leaf_nodes=100, n_jobs=-1)

# Lists to store RMSE values during training
rmse1_history = []
rmse2_history = []
rmse3_history = []

# Fit and predict with model1
print("Fitting model1...")
model1.fit(X_train, y_train)
print("Making predictions with model1...")
pred1 = model1.predict(X_val)
rmse1 = np.sqrt(mean_squared_error(y_val, pred1))
rmse1_history.append(rmse1)

# Fit and predict with model2
print("Fitting model2...")
model2.fit(X_train, y_train)
print("Making predictions with model2...")
pred2 = model2.predict(X_val)
rmse2 = np.sqrt(mean_squared_error(y_val, pred2))
rmse2_history.append(rmse2)

# Fit and predict with model3
print("Fitting model3...")
model3.fit(X_train, y_train)
print("Making predictions with model3...")
pred3 = model3.predict(X_val)
rmse3 = np.sqrt(mean_squared_error(y_val, pred3))
rmse3_history.append(rmse3)

# Average predictions
final_pred = (pred1 + pred2 + pred3) / 3
rmse_final = np.sqrt(mean_squared_error(y_val, final_pred))

# Plot RMSE during training
plt.figure(figsize=(10, 6))
plt.plot(rmse1_history, label="Model 1")
plt.plot(rmse2_history, label="Model 2")
plt.plot(rmse3_history, label="Model 3")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.title("RMSE during Training")
plt.show()

print(f"Final RMSE: {rmse_final}")
