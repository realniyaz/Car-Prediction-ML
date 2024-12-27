# Car-Prediction-ML
#### Problem Statement:
The price of a car depends on a lot of factors like the goodwill of the brand of the car, features of the car, horsepower and the mileage it gives and many more. Car price prediction is one of the major research areas in machine learning. So if you want to learn how to train a car price prediction model then this project is for you

This project predicts car prices using machine learning techniques. It involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Project Structure

The project is organized into the following files:

*   **`main.py`:** The main execution script that orchestrates the entire process.
*   **`data_preprocessing.py`:** Contains functions for data loading, cleaning, feature engineering, and data splitting.
*   **`eda.py`:** Contains functions for performing exploratory data analysis (EDA) and generating visualizations.
*   **`model_training.py`:** Contains functions for training and evaluating the machine learning model.

## Data

The dataset used for this project should be a CSV file containing car-related information, including features like Make, Model, Year, Mileage, Condition, and Price (the target variable). An example file name is `CarPricesPrediction.csv`. Place this file in the same directory as the python scripts.

## Setup

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    ```

2.  **Install the required libraries:**

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

    If you are using conda:

    ```bash
    conda install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Place the dataset:** Ensure your CSV data file (`CarPricesPrediction.csv` or similar) is in the same directory as the Python scripts.

## Usage

1.  Navigate to the project directory in your terminal or command prompt.

2.  Run the main script:

    ```bash
    python main.py
    ```

## Code Description

### `data_preprocessing.py`

This module preprocesses the data before it is used for modeling. The main steps are:

*   **Data Loading and Cleaning:** Reads the CSV file, removes whitespace from column names, and handles an optional "Unnamed: 0" index column.
*   **Data Type Conversion:** Converts relevant columns (`Kilometer_Driven`, `Year`, `Price`) to numeric types.
*   **Feature Engineering:** Creates new features like `Age` (car age) and `Mileage_Per_Year`.
*   **Data Splitting:** Splits the data into training and testing sets (80/20 split).
*   **One-Hot Encoding:** Encodes categorical features using OneHotEncoder.
*   **Imputation:** Fills missing values using the mean (for numerical features) or mode (for categorical features) from the training set.
*   **Scaling/Normalization:** Scales numerical features using StandardScaler.

### `eda.py`

This module performs exploratory data analysis (EDA) to understand the data. It includes:

*   **Descriptive Statistics:** Calculates and prints summary statistics for numerical and categorical features.
*   **Visualizations:** Generates histograms, scatter plots, box plots, count plots, and a correlation matrix heatmap.

### `model_training.py`

This module trains and evaluates the machine learning model:

*   **Model Training:** Trains a Linear Regression model on the preprocessed training data.
*   **Evaluation:** Calculates and returns the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared on the test data.

### `main.py`

This script is the entry point for the project. It:

*   Calls the preprocessing functions from `data_preprocessing.py`.
*   Calls the EDA functions from `eda.py`.
*   Calls the training and evaluation functions from `model_training.py`.
*   Prints the evaluation metrics.
*   Includes error handling to manage potential issues during preprocessing or model training.

## Results

After running `main.py`, the script will print the following evaluation metrics:

*   Mean Squared Error (MSE)
*   Root Mean Squared Error (RMSE)
*   R-squared

These metrics provide an indication of the model's performance in predicting car prices.

## Further Improvements

*   **Model Selection:** Experiment with other regression models (e.g., Random Forest, Gradient Boosting, Support Vector Regression) and compare their performance.
*   **Hyperparameter Tuning:** Optimize the hyperparameters of the chosen model using techniques like GridSearchCV or RandomizedSearchCV.
*   **Feature Selection:** Use feature selection techniques to identify the most important features and potentially improve model performance.
*   **More Advanced EDA:** Explore more advanced EDA techniques to gain deeper insights into the data.
*   **Deployment:** Deploy the trained model to a web application or other platform for real-world use.
