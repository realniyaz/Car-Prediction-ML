import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(data):
    try:
        print("\nDescriptive Statistics:\n", data.describe(include='all'))

        numerical_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(include='object').columns

        for col in numerical_cols:
            plt.figure()
            data[col].hist()
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

        for col in numerical_cols:
            if col != 'Price':
                plt.figure()
                data.plot.scatter(x=col, y='Price')
                plt.title(f'Scatter Plot of {col} vs. Price')
                plt.xlabel(col)
                plt.ylabel('Price')
                plt.show()

        for col in categorical_cols:
            plt.figure()
            sns.boxplot(x=col, y='Price', data=data)
            plt.title(f'Box Plot of Price by {col}')
            plt.xticks(rotation=90)
            plt.show()

        for col in categorical_cols:
            plt.figure()
            sns.countplot(x=col, data=data)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=90)
            plt.show()

        correlation_matrix = data.corr(numeric_only=True)
        plt.figure()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred during EDA: {e}")
