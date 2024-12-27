## `model_training.py` - Model Training and Evaluation

This module contains functions for training and evaluating a machine learning model for car price prediction.

**Function: `train_and_evaluate_model(X_train, y_train, X_test, y_test)`**

This function takes the training and testing data as input and performs the following:

1.  **Model Instantiation:** Creates an instance of a Linear Regression model (`LinearRegression()`).

2.  **Model Training:** Trains the model on the training data using `model.fit(X_train, y_train)`.

3.  **Prediction:** Makes predictions on the test data using `model.predict(X_test)`.

4.  **Evaluation:** Calculates and returns the following evaluation metrics:
    *   Mean Squared Error (MSE) using `mean_squared_error(y_test, predictions)`.
    *   Root Mean Squared Error (RMSE) by taking the square root of the MSE.
    *   R-squared (Coefficient of Determination) using `r2_score(y_test, predictions)`.

**Error Handling:**

*   Includes a `try...except` block to catch and print any exceptions that occur during model training or evaluation.

**Input:**

*   `X_train`: Training features (NumPy array or Pandas DataFrame).
*   `y_train`: Training target values (NumPy array or Pandas Series).
*   `X_test`: Testing features.
*   `y_test`: Testing target values.

**Output:**

*   `mse`: Mean Squared Error.
*   `rmse`: Root Mean Squared Error.
*   `r2`: R-squared.
*   Returns `None, None, None` if an error occurs.
