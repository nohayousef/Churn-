Customer Churn Prediction Project

This project focuses on building a machine learning model to predict customer churn using an Artificial Neural Network (ANN). The project is implemented in Python using libraries such as TensorFlow/Keras, Pandas, and Scikit-learn.

Objective

The main objective of this project is to develop a predictive model that can determine whether a customer is likely to leave a bank, based on various customer attributes. This is a typical classification problem in the domain of customer relationship management and business analytics.

Model

A feedforward Artificial Neural Network (ANN) is used for classification. The model was trained using a dataset of bank customers, with features including:

Credit Score

Geography

Gender

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Dataset

The dataset is based on the Churn_Modelling.csv dataset (commonly used in ML tutorials), which contains 10,000 entries of bank customers with the target variable being Exited (0 = stayed, 1 = left).

Key Steps

Data Preprocessing

Categorical variable encoding (LabelEncoder, OneHotEncoder)

Feature scaling using StandardScaler

Train-test split

Model Building

A sequential ANN model with:

Input layer

Two hidden layers (with ReLU activation)

Output layer (with sigmoid activation for binary classification)

Training & Evaluation

Compiled using adam optimizer and binary_crossentropy loss

Model evaluated on the test set using accuracy metrics

Results

The model achieved around 84% accuracy on the test data (based on the final results shown in the notebook).

The confusion matrix and classification report show good performance on both classes.

Requirements

Python 3.7+

Jupyter Notebook

TensorFlow / Keras

Pandas

NumPy

Scikit-learn

Matplotlib (optional for visualization)

Future Improvements

Perform hyperparameter tuning

Add dropout or regularization to avoid overfitting

Use more complex models or ensemble methods for better performance

Deploy the model using Streamlit or Flask for live predictions
