import datapreprocessing
import eda
import model_training

if __name__ == "__main__":
    data_path = r"C:\Users\Dell\Desktop\Car Project\CarPricesPrediction.csv"
    X_train, X_test, y_train, y_test, data_original = datapreprocessing.preprocess_data(data_path)

    if X_train is not None: #Check if preprocessing was successful
        eda.perform_eda(data_original) #Pass the original data to the EDA function
        mse, rmse, r2 = model_training.train_and_evaluate_model(X_train, y_train, X_test, y_test)

        if mse is not None: #Check if model training was successful
            print("\nModel Evaluation:")
            print("Mean Squared Error (MSE):", mse)
            print("Root Mean Squared Error (RMSE):", rmse)
            print("R-squared:", r2)
    else:
        print("Data preprocessing failed. Exiting.")
