## `eda.py` - Exploratory Data Analysis

This module contains functions for performing Exploratory Data Analysis (EDA) on the car price dataset.

**Function: `perform_eda(data)`**

This function takes a Pandas DataFrame (`data`) as input and performs the following EDA tasks:

1.  **Descriptive Statistics:**
    *   Prints descriptive statistics of the DataFrame using `data.describe(include='all')`. This includes statistics for both numerical and categorical columns.

2.  **Visualizations:**

    *   **Histograms:** Generates histograms for all numerical columns to visualize their distributions.
        *   Uses `data[col].hist()` to create the histograms.
        *   Sets appropriate titles and labels for each plot.
    *   **Scatter Plots:** Creates scatter plots to visualize the relationship between each numerical feature and the `Price` (target variable).
        *   Uses `data.plot.scatter(x=col, y='Price')` to create the scatter plots.
        *   Sets appropriate titles and labels.
    *   **Box Plots:** Generates box plots to visualize the distribution of `Price` for each categorical feature.
        *   Uses `seaborn.boxplot(x=col, y='Price', data=data)` for better-looking box plots.
        *   Rotates x-axis labels for readability.
    *   **Count Plots:** Creates count plots to visualize the frequency of different categories in each categorical feature.
        *   Uses `seaborn.countplot(x=col, data=data)`.
        *   Rotates x-axis labels for readability.
    *   **Correlation Matrix:** Computes and visualizes the correlation matrix for all numerical features.
        *   Uses `data.corr(numeric_only=True)` to calculate the correlation matrix.
        *   Uses `seaborn.heatmap()` to create a heatmap for better visualization.

**Error Handling:**

*   Includes a `try...except` block to catch and print any exceptions that occur during EDA.

**Input:**

*   `data`: A Pandas DataFrame containing the car price data (after basic preprocessing like duplicate removal and renaming the Mileage column).

**Output:**

*   None (displays plots and prints descriptive statistics).
