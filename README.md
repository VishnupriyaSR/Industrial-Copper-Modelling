# Industrial-Copper-Modelling

#Introduction

This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads.

#Key Technologies and Skills

Python

Numpy

Pandas

Scikit-Learn

Matplotlib

Seaborn

Pickle
      
Streamlit

#Approach: 

1)Data Understanding: Identify the types of variables (continuous, categorical) and their distributions.

2)Data Preprocessing: 

Handle missing values with mean/median/mode.

Treat Outliers using IQR or Isolation Forest from sklearn library.

Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.

Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable

3)EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot.

4)Feature Engineering:

Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. 

5)Model Building and Evaluation:.

Split the dataset into training and testing/validation sets. 

Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 

Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.

Interpret the model results and assess its performance based on the defined problem statement.

Same steps for Regression modelling.

6)Model GUI: Using streamlit module, create interactive page.






