# Optimizing-Harvesting-for-Maximum-Profit
## Predicting Best Month to Harvest Vegetables, Fruits & Flowers


![image](https://github.com/VKunjir/Optimizing-Harvesting-for-Maximum-Profit/assets/98226339/4b308533-0219-471f-8233-0932f3f02f58)


## Overview
This project aims to predict the best month for harvesting various agricultural goods such as vegetables, fruits, and flowers in order to maximize farmers' profits. By leveraging machine learning techniques, we can forecast the optimal time to harvest different crops, taking into account factors such as seasonal variations, market demand, and price fluctuations.

## Implementation
The project is implemented using Python and several machine learning libraries including pandas, scikit-learn, and Streamlit for building the web application. The following steps outline the implementation process:

1. **Data Collection**: The data is collection from krushi upana samete haveli pune containing historical prices and seasonal trends of past 3 years agricultural goods are synthetically for training the machine learning model.

2. **Data Preprocessing**: The collected data is preprocessed to handle missing values, encode categorical variables, and scale numerical features as necessary. This step ensures that the data is in a suitable format for training the machine learning model.

3. **Model Training**: Various machine learning models are trained using the preprocessed data. These models include:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - K-Nearest Neighbors
    - Neural Network
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost

4. **Model Evaluation**: The trained models are evaluated using appropriate evaluation metrics such as Mean Squared Error (MSE) or R-squared score to assess their performance and determine the best-performing model. We found Decision tree give R^2 score of 99% thus we used it for final model creation.

5. **Web Application Development**: Using Streamlit, a web application framework for Python, a user-friendly interface is developed where users can input the agricultural good, the year, and the number of top prices they want to predict. Upon submission, the application retrieves the predicted prices for the specified inputs and displays the top months for harvesting.


## Libraries Used
- pandas: For data manipulation and preprocessing.
- scikit-learn: For building and training machine learning models.
- Streamlit: For building the interactive web application.
- numpy: For numerical computations.
- matplotlib: For data visualization.
- seaborn: For statistical data visualization.
- XGBoost, LightGBM, CatBoost: Additional machine learning libraries for boosting algorithms.
- warnings: For handling warning messages.

## Contributors
- Viraj Kunjir, https://github.com/VKunjir


