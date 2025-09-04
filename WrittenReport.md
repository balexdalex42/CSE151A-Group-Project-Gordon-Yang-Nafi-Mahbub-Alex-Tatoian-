# Written Report

## Introduction
For this project, we will try to classify asteroids as potentially hazardous (PHA) using the NASA Small Body Database (SBDB) dataset which contains detailed orbital and physical properties of over 958,000 asteroids. We first had to try preprocessing the data to remove irrelevant or noisy features, we then experimented with multiple machine learning models we learned in this course. We initially developed a Decision Tree Classifier and a Support Vector Machine (SVM) to establish a baseline performance, addressing challenges such as class imbalance and feature selection. We then tried building on these, trying an unsupervised learning approach using Principal Component Analysis (PCA) to reduce dimensionality followed by a K Nearest Neighbors (KNN) classifier to enhance prediction accuracy. Our final submission evaluates the performance of all the models we tried, identifies underfitting or overfitting tendencies, and proposes improvements contributing to the goal of accurately identifying hazardous asteroids.

## Figures

## Methods Section

### Decision Tree Classifier
The notebook for the Decision Tree Classifier can be found **[here](/ms3decisiontree.ipynb)**.


For the decision tree we finished preprocessing by imputing missing numeric values with the median and categorical values with the most frequent value fit on the training split. We also one hot encoded NEO and equinox so the model sees clean numeric inputs and added a simple log transform to skewed positive columns diameter, moid_ld, a and q to reduce skew and keep splits from being driven by extreme values. We did not scale features since trees are scale independent, ID fields stayed dropped and any impossible physical values were turned into missing values and handled by the computer.

### SVM Classifier

For the SVM Classifier, all `object` dtype columns, except the output `pha` class, were one-hot encoded, and `float64`, which made up the remainder, were z-score normalized. The `pha` column was encoded such that `'Y' => 1` and `'N' => 0`. At this stage, I did 3:1 of the dataset into training and testing sets. Additionally, due to the huge imbalance issue of our `pha` output class, I also ran an oversampling using the `ADASYN` class from `imblearn.over_sampling` to balance out the output class only on the `X_train` and `y_train` data. This final step pushed the shapes of these to `(699251, 29)` and `(699251,)` respectively.

## Results Section

## Discussion Section

## Conclusion
