import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import datetime

def preprocess_data(data_path):
    try:
        data = pd.read_csv(data_path, encoding='latin1')
        data.columns = data.columns.str.strip()
        print("Columns in the DataFrame:", data.columns)
        print(data.head())
        print(data.dtypes)

        if 'Unnamed: 0' in data.columns:
            data.drop('Unnamed: 0', axis=1, inplace=True)

        if 'Mileage' in data.columns:
            data = data.rename(columns={'Mileage': 'Kilometer_Driven'})

        for col in ['Kilometer_Driven', 'Year', 'Price']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        current_year = datetime.datetime.now().year
        data['Age'] = current_year - data['Year']
        data['Age'] = data['Age'].apply(lambda x: 1 if x==0 else x)
        data['Mileage_Per_Year'] = data['Kilometer_Driven'] / (data['Age'] + 1e-6)

        X = data.drop('Price', axis=1)
        y = data['Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_cols = X_train.select_dtypes(include='object').columns
        if not categorical_cols.empty:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_encoded = pd.DataFrame(ohe.fit_transform(X_train[categorical_cols]), index=X_train.index)
            X_test_encoded = pd.DataFrame(ohe.transform(X_test[categorical_cols]), index=X_test.index)

            # Convert column names to strings *before* concatenating
            X_train_encoded.columns = X_train_encoded.columns.astype(str)
            X_test_encoded.columns = X_test_encoded.columns.astype(str)

            X_train = X_train.drop(columns=categorical_cols)
            X_test = X_test.drop(columns=categorical_cols)

            X_train = pd.concat([X_train, X_train_encoded], axis=1)
            X_test = pd.concat([X_test, X_test_encoded], axis=1)
        else:
            print("No categorical columns to encode")

        # Convert ALL column names to strings AFTER concatenation
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        for column in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[column]):
                if X_train[column].isnull().any():
                    train_mean = X_train[column].mean()
                    X_train.loc[:, column] = X_train[column].fillna(train_mean)
                    X_test.loc[:, column] = X_test[column].fillna(train_mean)
            else:
                if not X_train[column].mode().empty:
                    train_mode = X_train[column].mode()[0]
                    X_train.loc[:, column] = X_train[column].fillna(train_mode)
                    X_test.loc[:, column] = X_test[column].fillna(train_mode)
                else:
                    print(f"No mode found for column {column}. Imputation skipped.")

        numerical_cols = X_train.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            scaler = StandardScaler()
            X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])
        else:
            print("No numerical columns to scale")

        return X_train, X_test, y_train, y_test, data

    except FileNotFoundError:
        print("Error: CarPricesPrediction.csv not found at the specified path.")
        return None, None, None, None, None
    except pd.errors.ParserError:
        print("Error: There was a problem parsing the CSV file. Check the file format.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None, None, None, None, None
