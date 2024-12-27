from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        return mse, rmse, r2
    except Exception as e:
        print(f"An unexpected error occurred during model training/evaluation: {e}")
        return None, None, None
