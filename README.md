## Lung Cancer Prediction Using Machine Learning

### Project Overview

This project compares three machine learning models: k-Nearest Neighbors (k-NN), Regularized Logistic Regression, 
and Support Vector Machine (SVM) for the purpose of predicting lung cancer based on symptoms. Models are trained 
and tuned for optimal performance and their prediction accuracy is compared.

### Dataset Description

The data comes from a [Kaggle competition dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer), 
featuring a comprehensive collection of symptoms: age, gender, smoking status, yellow fingers, anxiety, peer pressure, 
fatigue, chronic disease, allergies, wheezing, alcohol consumption, coughing, shortness of breath, swalling difficulty,
and chest pain. 

### Approach

- Data Preprocessing
Data preprocessing, performed with Pandas and Scikit-Learn's Standard Scaler,
involved encoding categorical variables, and normalizing numerical values to ensure optimal model performance.

- Hyperparameter Optimization
A random search strategy was employed for hyperparameter tuning across 30 iterations for each model.
The hyperparameters considered were:
1) Number of neighbors, distance metric used, and weighting for distances for k-NN
2) Type of regularization (L1 or L2), and regularization parameter for Logistic Regression
3) Kernel type, regularization parameter, kernel parameter for SVM
For each hyperparameter choice, 10-fold cross-validation was used to evaluate model results, and the model
with highest average accuracy was saved. All models were implemented through Scikit-Learn.

### Visualization of Results
To aid in the interpretation of the models' predictions and understand the significance of different features, 
visualization techniques were employed using matplotlib and seaborn. This included plotting the raw data distributions 
and visualizing the weights and feature importance derived from the Regularized Logistic Regression and SVM models. 
