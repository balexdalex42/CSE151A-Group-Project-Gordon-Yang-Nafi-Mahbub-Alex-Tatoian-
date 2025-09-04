# Written Report

## Introduction
For this project, we will try to classify asteroids as potentially hazardous (PHA) using the NASA Small Body Database (SBDB) dataset which contains detailed orbital and physical properties of over 958,000 asteroids. We first had to try preprocessing the data to remove irrelevant or noisy features, we then experimented with multiple machine learning models we learned in this course. We initially developed a Decision Tree Classifier and a Support Vector Machine (SVM) to establish a baseline performance, addressing challenges such as class imbalance and feature selection. We then tried building on these, trying an unsupervised learning approach using Principal Component Analysis (PCA) to reduce dimensionality followed by a K Nearest Neighbors (KNN) classifier to enhance prediction accuracy. Our final submission evaluates the performance of all the models we tried, identifies underfitting or overfitting tendencies, and proposes improvements contributing to the goal of accurately identifying hazardous asteroids.

## Figures

## Methods Section

### Data Exploration
### Size of the dataset
Size: 958524 x 45\
There are a total of 958524 observations in the dataset, each with 45 columns, amounting to over 400 MB.
### Description of features
#### Categorical Data Columns
- **id**, **spkid**, **full_name**, **name**: Uniquely identifies each asteroid observed.
- **pdes**: The primary designation. Another identifier for each observed asteroid.
- **prefix**: The comet designation prefix. Only value given is 'A' meaning a comet that was mistakenly identified as an asteroid.
- **orbit_id**, **epoch**, **epoch_mjd**, **epoch_cal**, **equinox**: Describe the orbital characteristics of the asteroid.
- **class**: describes what kind of asteroid it is, specifically the region it's in. MBA is for example, "Main Belt Asteroid," in the Asteroid Belt between the orbits of Mars and Jupiter. This is a string data type.
#### Our Target Categories
- **neo**: Boolean flag for a "near-earth object" asteroid. "Y" if it is a near-earth object.
- **pha**: Boolean flag for a "potentially hazardous asteroid." "Y" if it is considered a potentially hazardous asteroid.
#### Continuous Data Columns
- **H**: The absolute magnitude (or brightness) of the asteroid, as a float type, if it were 1 AU from both the Sun and the Earth.
- **diameter**: The diameter of the asteroid as a float type.
- **diameter_sigma**: The uncertainty of the asteroid's diameter measurement, also given as a float.
- **e**: The eccentricity of the asteroid's orbit around the sun as a float. A value of 0 is a perfect circular orbit. Larger values under 1 are more elliptical. Any value greater than 1 means the orbital trajectory of the asteroid is hyperbolic and would thus be on an escape trajectory out of the solar system, and no longer orbiting the sun.
- **a**: The semi-major axis of the asteroid's orbit, meaning its longest radius. This is a float type.
- **q**: The perihelion distance of the asteroid's orbit, meaning its shortest distance it gets to the Sun on its orbital trajectory. This is a float type.
- **i**: Orbital inclination in degrees, which is the angle of the orbit with respect to the plane of the solar system.
- **om**: The longitude in degrees of the ascending node, which is the angle of the asteroid's orbit with respect to some reference plane.
- **w**: Argument of perihelion, which is the angle between the ascending node of the asteroid's orbit and the perihelion.
- **ma**: The mean anomaly, which is the fraction of the asteroid's orbital period that has elapsed, expressed as a float data type.
- **ad**: Aphelion distance, which is the distance at which the asteroid orbits furthest from the Sun.
- **n**: Mean motion, which is the average angular velocity of an asteroid to complete one orbit around the Sun.
- **tp**, **tp_cal**: Time of perihelion passage, expressed as Barycentric Dynamical Time (TDB) and its calendar time.
- **per**, **per_y**: The asteroid's orbital period expressed in units of days and years respectively.
- **moid**, **moid_ld**: The minimum distance between the orbit of the asteroid with respect to Earth's, expressed in astronomical units (AU) and lunar distance (ld), respectively.
- **sigma_e**, **sigma_a**, **sigma_q**, **sigma_i**, **sigma_om**, **sigma_w**, **sigma_ma**, **sigma_ad**, **sigma_n**, **sigma_per**, **sigma_tp**: The measurement uncertainties of each of their respective attributes.
- **rms**: The root mean square of the orbital fit.
### Missing and duplicate data in dataset
Most of the entries have at least some data in some columns. An interesting observation is that the asteroid name is missing from the vasy majority of the entries. So are the diameter attributes, and albedos, which may be because the observation telescopes may not be good enough to measure extremely small objects, which make up the vast majority of asteroids. In one of the target classes, "neo," there are only four missing from the entire set. Lastly, there are some nineteen thousand values missing from the sigma_ columns, which are the measurement uncertainties of various attributes.

By running a simple line of code: `df[df.duplicated()]` we were able to verify that no duplicate entries exist in the dataset at all.


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

### First Model: Decision Tree Classifier

We think the decision tree is a solid first model, as with class weights and light preprocessing it delivers strong recall on PHA and better precision than the SVM run. Most misses are borderline cases that look similar in brightness or orbit, so the tree occasionally flags extra positives. To tighten it up we can prune with ccp alpha and a larger minimum leaf, tune the decision threshold from the validation PR curve and if recall still needs help try some light SMOTE on the training split. After that we can compare the model against a calibrated linear SVM as a fast margin baseline.

### Second Model: SVM using Linear and RBF Kernels

We found that SVM was a decent model in terms of the practicality of the model in real life, where there mostly problems with precision: non-hazardous asteroids are likely to be flagged. Our model accuracy was high, thus this ends up being a much better case than hazardous asteroids not being flagged! In order to try to increase our precision, we can potentially use class weights differently as ADASYN makes the model focuses on boosting recall, in favor of precision. We can also potentially decrease our regularization parameter, making the SVM margin wider which might decrease the amount of false positives. Finally, SVM ended up being extremely slow, with our implementation using the linear kernel to be around 70 minutes and the implenetation using the RBF kernel to be around 2 hours! This is a substantial amount of time that would scale very fast, if say, our dataset was much larger. We think that using the super computer, potentially writing a few cuda kernels, or using pytorch (for parallelized matrix multiplication) can reduce this time cost significantly. All it boils down to is how much computing power we need, and unfortunately using our own devices without algorithm optimizations or parallelism will greatly slow training time.


## Conclusion
