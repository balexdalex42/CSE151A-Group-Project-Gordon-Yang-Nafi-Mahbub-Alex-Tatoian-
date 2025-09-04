# Asteroid Threat Detection - CSE151A Final Project

## Introduction
For this project, we will try to classify asteroids as potentially hazardous (PHA) using the NASA Small Body Database (SBDB) dataset which contains detailed orbital and physical properties of over 958,000 asteroids. We first had to try preprocessing the data to remove irrelevant or noisy features, we then experimented with multiple machine learning models we learned in this course. We initially developed a Decision Tree Classifier and a Support Vector Machine (SVM) to establish a baseline performance, addressing challenges such as class imbalance and feature selection. We then tried building on these, trying an unsupervised learning approach using Principal Component Analysis (PCA) to reduce dimensionality followed by a K Nearest Neighbors (KNN) classifier to enhance prediction accuracy. Our final submission evaluates the performance of all the models we tried, identifies underfitting or overfitting tendencies, and proposes improvements contributing to the goal of accurately identifying hazardous asteroids.




Milestone 2:

README:

README: **[readme](/ms2rm.md)**

Notebook:

**[notebook](/ms2.ipynb)**

Milestone 3:

README: **[readme](/ms3rm.md)**

Notebook:

**[prep](/prep_dataset.ipynb)**
**[svm](/model_svm.ipynb)**
**[decision tree](/ms3decisiontree.ipynb)**

Milestone 4:

Writeup:

**[Final Writeup](/WrittenReport.md)**

Notebook:

**[KNN](/ms4.ipynb)**


















## Data Preprocessing
For this part, we dropped columns that had a large amount of `NaN` data, like the `prefix`, `diameter`, and `albedo`. The `equinox` column also had only one unique value, which suggests that this feature will not help in our classification problem, so it was removed too. Names and identifier columns, `id`, `spkid`, `name`, `pdes`, `full_name`, and `prefix`, of the asteroids were removed as well as none of that information will give insight into its potential hazard. Additionally, some columns are essentially duplicates, of each other since they are expressing the same data in different units or a different format. Lastly, all the `sigma` columns were removed, as these are measurement uncertainties of existing attributes and will not give us anymore information for our classifier. In the end, we decided that we had more than enough data (932335) compared to the original size (958524) to just drop the remaining rows that had `NaN` data without having to do any imputation. There are only 18 columns left after removal of columns as well. Since we tried out two different models, additional data preprocessing steps follow below, specific to each model.

For simplicity, we ran this preprocessing step in a different notebook, located **[here](/prep_dataset.ipynb)**, and then saved it as a csv under `/archive/prep_dataset.csv` and then directly loaded that file into our model training notebooks. This **[notebook](/prep_dataset.ipynb)** must be run first before `ms3decisiontree.ipynb` and `model_svm.ipynb` or they will not yield correct results.

### For Decision Tree Classifier
The notebook for the Decision Tree Classifier can be found **[here](/ms3decisiontree.ipynb)**.


For the decision tree we finished preprocessing by imputing missing numeric values with the median and categorical values with the most frequent value fit on the training split. We also one hot encoded NEO and equinox so the model sees clean numeric inputs and added a simple log transform to skewed positive columns diameter, moid_ld, a and q to reduce skew and keep splits from being driven by extreme values. We did not scale features since trees are scale independent, ID fields stayed dropped and any impossible physical values were turned into missing values and handled by the computer.

### For SVM Classifier

For the SVM Classifier, all `object` dtype columns, except the output `pha` class, were one-hot encoded, and `float64`, which made up the remainder, were z-score normalized. The `pha` column was encoded such that `'Y' => 1` and `'N' => 0`. At this stage, I did 3:1 of the dataset into training and testing sets. Additionally, due to the huge imbalance issue of our `pha` output class, I also ran an oversampling using the `ADASYN` class from `imblearn.over_sampling` to balance out the output class only on the `X_train` and `y_train` data. This final step pushed the shapes of these to `(699251, 29)` and `(699251,)` respectively.

## First Model: Decision Tree Classifier 

The decision tree model follows the usual pattern where with very small depth it cannot split enough, so both train and validation errors are high and it underfits. As we grow the depth the train error drops fast and the validation error goes down, then starts creeping back up while train error heads toward zero, which is the overfitting side. We chose the depth right around the bottom of the validation curve where the gap between train and validation is small. So the model sits in the balanced zone rather than the underfit or overfit ends.

After seeing the success of the decision tree classifier, we wanted to give a shot with the Support Vector Machine (SVM) as our output class is binary and the decision boundary would likely be easy to determine, and so we gave it a try below.

## Second Model Trial: SVM using Linear and RBF Kernels
Surprisingly, we got a much lower precision for identifying **hazardous** asteroids when running SVM than we did with the decision tree models. Also another surprise was that the kernel choice did not matter. Both linear and rbf kernels resulted in similar precision of 0.35 and 0.36 for **hazardous** asteroid classification. So what this tells us is that this model returns a lot of false positives, meaning we predict a large amount of asteroids to be **hazardous** when in reality, they weren't. So the silver lining here is that in the context of planetary defense, this model works fairly well and errs on the side of caution, as all other metrics, including accuracy, are nearly 100%. Lastly, comparing the training and testing metrics, with all training metrics always at near 100%, this model is **overfitting**.

The notebook for the SVM trial can be found **[here](/model_svm.ipynb)**.

## Conclusion
**What is the conclusion of your 1st model? What can be done to possibly improve it?**  

### First Model: Decision Tree Classifier

We think the decision tree is a solid first model, as with class weights and light preprocessing it delivers strong recall on PHA and better precision than the SVM run. Most misses are borderline cases that look similar in brightness or orbit, so the tree occasionally flags extra positives. To tighten it up we can prune with ccp alpha and a larger minimum leaf, tune the decision threshold from the validation PR curve and if recall still needs help try some light SMOTE on the training split. After that we can compare the model against a calibrated linear SVM as a fast margin baseline.

### Second Model: SVM using Linear and RBF Kernels

We found that SVM was a decent model in terms of the practicality of the model in real life, where there mostly problems with precision: non-hazardous asteroids are likely to be flagged. Our model accuracy was high, thus this ends up being a much better case than hazardous asteroids not being flagged! In order to try to increase our precision, we can potentially use class weights differently as ADASYN makes the model focuses on boosting recall, in favor of precision. We can also potentially decrease our regularization parameter, making the SVM margin wider which might decrease the amount of false positives. Finally, SVM ended up being extremely slow, with our implementation using the linear kernel to be around 70 minutes and the implenetation using the RBF kernel to be around 2 hours! This is a substantial amount of time that would scale very fast, if say, our dataset was much larger. We think that using the super computer, potentially writing a few cuda kernels, or using pytorch (for parallelized matrix multiplication) can reduce this time cost significantly. All it boils down to is how much computing power we need, and unfortunately using our own devices without algorithm optimizations or parallelism will greatly slow training time.
