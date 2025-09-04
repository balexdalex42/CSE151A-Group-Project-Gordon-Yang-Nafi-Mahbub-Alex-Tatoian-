# Written Report

## Introduction
For this project, we will try to classify asteroids as potentially hazardous (PHA) using the NASA Small Body Database (SBDB) dataset which contains detailed orbital and physical properties of over 958,000 asteroids. We first had to try preprocessing the data to remove irrelevant or noisy features, we then experimented with multiple machine learning models we learned in this course. We initially developed a Decision Tree Classifier and a Support Vector Machine (SVM) to establish a baseline performance, addressing challenges such as class imbalance and feature selection. We then tried building on these, trying an unsupervised learning approach using Principal Component Analysis (PCA) to reduce dimensionality followed by a K Nearest Neighbors (KNN) classifier to enhance prediction accuracy. Our final submission evaluates the performance of all the models we tried, identifies underfitting or overfitting tendencies, and proposes improvements contributing to the goal of accurately identifying hazardous asteroids.

## Figures

## Methods Section

### Decision Tree Classifier
The notebook for the Decision Tree Classifier can be found **[here](/ms3decisiontree.ipynb)**.


For the decision tree we finished preprocessing by imputing missing numeric values with the median and categorical values with the most frequent value fit on the training split. We also one hot encoded NEO and equinox so the model sees clean numeric inputs and added a simple log transform to skewed positive columns diameter, moid_ld, a and q to reduce skew and keep splits from being driven by extreme values. We did not scale features since trees are scale independent, ID fields stayed dropped and any impossible physical values were turned into missing values and handled by the computer.

### SVM Classifier

The notebook for the SVM trial can be found **[here](/model_svm.ipynb)**.

For the SVM Classifier, all `object` dtype columns, except the output `pha` class, were one-hot encoded, and `float64`, which made up the remainder, were z-score normalized. The `pha` column was encoded such that `'Y' => 1` and `'N' => 0`. At this stage, I did 3:1 of the dataset into training and testing sets. Additionally, due to the huge imbalance issue of our `pha` output class, I also ran an oversampling using the `ADASYN` class from `imblearn.over_sampling` to balance out the output class only on the `X_train` and `y_train` data. This final step pushed the shapes of these to `(699251, 29)` and `(699251,)` respectively.

## Results Section

## First Model: Decision Tree Classifier 

The decision tree model follows the usual pattern where with very small depth it cannot split enough, so both train and validation errors are high and it underfits. As we grow the depth the train error drops fast and the validation error goes down, then starts creeping back up while train error heads toward zero, which is the overfitting side. We chose the depth right around the bottom of the validation curve where the gap between train and validation is small. So the model sits in the balanced zone rather than the underfit or overfit ends.

After seeing the success of the decision tree classifier, we wanted to give a shot with the Support Vector Machine (SVM) as our output class is binary and the decision boundary would likely be easy to determine, and so we gave it a try below.

## Second Model Trial: SVM using Linear and RBF Kernels
Surprisingly, we got a much lower precision for identifying **hazardous** asteroids when running SVM than we did with the decision tree models. Also another surprise was that the kernel choice did not matter. Both linear and rbf kernels resulted in similar precision of 0.35 and 0.36 for **hazardous** asteroid classification. So what this tells us is that this model returns a lot of false positives, meaning we predict a large amount of asteroids to be **hazardous** when in reality, they weren't. So the silver lining here is that in the context of planetary defense, this model works fairly well and errs on the side of caution, as all other metrics, including accuracy, are nearly 100%. Lastly, comparing the training and testing metrics, with all training metrics always at near 100%, this model is **overfitting**.



## Discussion Section

## Conclusion
