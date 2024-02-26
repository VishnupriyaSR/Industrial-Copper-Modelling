# Industrial-Copper-Modelling

Introduction

This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads.

Key Technologies and Skills

Python
Numpy
Pandas
Scikit-Learn
Matplotlib
Seaborn
Pickle
Streamlit

Approach: 

1)Data Understanding: Identify the types of variables (continuous, categorical) and their distributions.

2)Data Preprocessing: 

Handle missing values with mean/median/mode.
Treat Outliers using IQR or Isolation Forest from sklearn library.
Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.
Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable


