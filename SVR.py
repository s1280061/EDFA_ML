from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Load dataset
dataset = np.loadtxt("EDFA_1_18dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a SVR model
svr = SVR()

# Create the MultiOutputRegressor model
mor_svr = MultiOutputRegressor(svr)

# Fit the model
mor_svr.fit(X_train, y_train)

# Predict
y_pred = mor_svr.predict(X_val)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")

print("Calculating MAE...")
mae = np.mean(np.abs(y_pred - y_val))
print(f"MAE:{mae}")

print("Calculating bias...")
bias = np.mean(y_pred - y_val)
print(f"Mean Bias: {bias}")

def explained_variance_score(y_true, y_pred):
    """
    Explained Variance Score（VAR）を計算します。

    :param y_true: 実際のデータの値 (NumPy 配列など)
    :param y_pred: モデルの予測値 (NumPy 配列など)
    :return: VAR 値
    """
    # 予測値と実際のデータの差を計算
    residual = y_true - y_pred

    # 実際のデータと予測値の差の分散を計算
    residual_variance = np.var(residual)

    # 実際のデータの分散を計算
    data_variance = np.var(y_true)

    # VAR の計算
    var = 1 - (residual_variance / data_variance)

    return var

# Calculate VAR
print("Calculating VAR for final predictions...")
variance = explained_variance_score(y_val, y_pred)
print(f"VAR: {variance}")



def calculate_r_squared(y_true, y_pred):
    """
    R-squared（決定係数）を計算します。

    :param y_true: 実際のデータの値 (NumPy 配列など)
    :param y_pred: モデルの予測値 (NumPy 配列など)
    :return: R-squared 値
    """
    # 実際のデータの平均を計算
    y_mean = np.mean(y_true)

    # 総平方和 (Total Sum of Squares) の計算
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)

    # 回帰平方和 (Regression Sum of Squares) の計算
    regression_sum_of_squares = np.sum((y_pred - y_mean) ** 2)

    # R-squared の計算
    r_squared = 1 - (regression_sum_of_squares / total_sum_of_squares)

    return r_squared

# Calculate R-squared
print("Calculating R-squared for final predictions...")
r_squared = calculate_r_squared(y_val, y_pred)
print(f"R-squared: {r_squared}")


# Save the model
with open("mor_svr.pkl", "wb") as f:
    pickle.dump(mor_svr, f)

