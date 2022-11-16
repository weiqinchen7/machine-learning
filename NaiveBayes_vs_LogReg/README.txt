Author: Weiqin Chen
RIN: 662027033
Email: chenw18@rpi.edu

There are 2 executable python scripts:
    1. logisticRegression: Logistic Regression for two-class classification problem
    2. naiveBayesDiscrete: Naive Bayes Model with Univariate Gaussian Approximation for multi-class classification problem

Requirements for the dataset:
    1. Must be in .csv format
    2. Header is not allowed in the dataset
    3. Row for cases, column for features
    4. The first column should be the indicator of classes
    5. Missing value is not allowed in the dataset
    6. Logistic Regression in this project only supports two-class problems

To run the scripts, use the following commands in the terminal
    python naiveBayesGaussian.py ./spam.csv 100 "5 10 15 20 25 30"
    
    python logisticRegression.py ./spam.csv 100 "5 10 15 20 25 30"
