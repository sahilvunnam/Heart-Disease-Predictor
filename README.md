# Heart-Disease-Predictor

This project explores various machine learning and deep learning techniques to predict heart disease from patient data. Using Python and popular libraries such as Pandas, Scikit-Learn, and TensorFlow, the code evaluates multiple classification models and visualizes important insights in the dataset.

Project Structure
Data Loading and Exploration:

Loads the heart disease dataset using Pandas.
Examines the dataset's shape and distribution of key variables, including gender, chest pain type, and fasting blood sugar.
Visualizes relationships between different features (e.g., Thal, sex, slope) and their correlation with heart disease.
Data Preprocessing:

One-hot encodes categorical variables like chest pain type and slope.
Normalizes the dataset to improve the model's training process.
Model Selection and Evaluation:

Implements various machine learning models including:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Naive Bayes
Evaluates each model's accuracy on the test set, with results displayed as a bar plot for easy comparison.
Manual Logistic Regression Implementation:

Demonstrates forward and backward propagation with custom functions for weight and bias updates.
Visualizes cost reduction over iterations, showing the improvement of model accuracy.
Neural Network with TensorFlow:

Trains a simple neural network using a Sequential model with multiple Dense layers, dropout, and batch normalization.
Uses EarlyStopping to avoid overfitting.
Visualization Examples
The project includes several visualizations to illustrate feature distributions and correlations with heart disease:

Count plots showing categorical feature distributions (e.g., sex, Thal).
Cross-tabulations showing feature correlations with heart disease, such as age and fasting blood sugar.
Dependencies
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
TensorFlow (for neural network)
How to Run
Clone the repository.
Install dependencies via pip install -r requirements.txt.
Run the main notebook or script to explore the data, visualize insights, and evaluate model accuracies.
