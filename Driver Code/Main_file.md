## `main.py` - Main Execution Script

This script orchestrates the entire process of data preprocessing, EDA, model training, and evaluation.

**Main Execution Block (`if __name__ == "__main__":`)**

1.  **Data Path:** Defines the path to the CSV data file.

2.  **Data Preprocessing:** Calls the `preprocess_data()` function from `data_preprocessing.py` to load, clean, engineer features, and split the data into training and testing sets.

3.  **EDA:** Calls the `perform_eda()` function from `eda.py` to perform exploratory data analysis on the original preprocessed data.

4.  **Model Training and Evaluation:** Calls the `train_and_evaluate_model()` function from `model_training.py` to train the model and evaluate its performance.

5.  **Print Results:** Prints the evaluation metrics (MSE, RMSE, R-squared) to the console.

**Error Handling:**

*   Checks if the data preprocessing and model training were successful before proceeding to the next step. If either of these steps fails (returns `None`), the script prints an error message and exits.

**Dependencies:**

*   Imports `data_preprocessing`, `eda`, and `model_training` modules.

**Execution:**

*   Run this script (e.g., `python main.py`) to execute the entire workflow.
