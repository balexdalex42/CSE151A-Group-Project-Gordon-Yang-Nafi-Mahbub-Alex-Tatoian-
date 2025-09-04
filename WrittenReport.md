# Written Report

## Introduction
For this project, we wanted to try classifying asteroids as potentially hazardous (PHA) using the NASA Small Body Database (SBDB) dataset, which contains detailed orbital and physical properties of over 958,000 asteroids. We picked this because it’s a real life challenge tying into planetary defense, like with NASA's DART mission which aims to prove whether we can strike an asteroid to alter its orbit and cause one heading towards Earth to miss instead, saving numerous lives. An extremely good predictive model that can classify a newly discovered asteroid as hazardous or not could go a long way to save many lives and our biosphere, and give humanity the time it needs to prepare its response.


## Figures
<img width="442" height="421" alt="Screenshot 2025-09-03 at 10 52 35 PM" src="https://github.com/user-attachments/assets/9727ed88-4305-48c1-a4cd-cda61d2986eb" />
<img width="442" height="421" alt="Screenshot 2025-09-03 at 10 55 19 PM" src="https://github.com/user-attachments/assets/86791350-ac4e-42f5-9dc1-d564c75da71b" />
<img width="442" height="421" alt="Screenshot 2025-09-03 at 10 55 29 PM" src="https://github.com/user-attachments/assets/0d84b0f1-b2f6-4d4b-8239-ef683d59bf58" />


## Methods
First, we had to preprocess the data to remove irrelevant or noisy features, then we tried experimenting with a bunch of machine learning models. We started with a Decision Tree Classifier and a Support Vector Machine (SVM) to set a baseline, tackling stuff like class imbalance and feature selection. Then we built on that, trying an unsupervised learning approach with Principal Component Analysis (PCA) to cut down dimensionality, followed by a K-Nearest Neighbors (KNN) classifier to boost prediction accuracy.
Having a solid predictive model here mattered a ton, since getting PHA classification right lets us spot dangers early and plan mitigation, which could save lives and protect us from rare but devastating asteroid hits. It’s important because it helps keep the planet safe, pushes forward research on near Earth objects, and gives space agencies the data they need to make smart calls, showing how big a deal predictive analytics can be for our world. Our final submission checks out how all the models we tried performed, figures out where they underfit or overfit, and mentions some ideas for improvement with the goal to help nail down hazardous asteroid predictions.
### Data Exploration
#### Size of the dataset
Size: 958524 x 45\
There are a total of 958524 observations in the dataset, each with 45 columns, amounting to over 400 MB.
#### Categorical Features
- **id**, **spkid**, **full_name**, **name**: Uniquely identifies each asteroid observed.
- **pdes**: The primary designation. Another identifier for each observed asteroid.
- **prefix**: The comet designation prefix. Only value given is 'A' meaning a comet that was mistakenly identified as an asteroid.
- **orbit_id**, **epoch**, **epoch_mjd**, **epoch_cal**, **equinox**: Describe the orbital characteristics of the asteroid.
- **class**: describes what kind of asteroid it is, specifically the region it's in. MBA is for example, "Main Belt Asteroid," in the Asteroid Belt between the orbits of Mars and Jupiter. This is a string data type.
#### Our Target Features
- **neo**: Boolean flag for a "near-earth object" asteroid. "Y" if it is a near-earth object.
- **pha**: Boolean flag for a "potentially hazardous asteroid." "Y" if it is considered a potentially hazardous asteroid.
#### Continuous Features
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

#### Missing and duplicate data in dataset
Most of the entries have at least some data in some columns. An interesting observation is that the asteroid name is missing from the vasy majority of the entries. So are the diameter attributes, and albedos, which may be because the observation telescopes may not be good enough to measure extremely small objects, which make up the vast majority of asteroids. In one of the target classes, "neo," there are only four missing from the entire set. Lastly, there are some nineteen thousand values missing from the sigma_ columns, which are the measurement uncertainties of various attributes.

### Data Preprocessing
For this part, we dropped columns that had a large amount of `NaN` data, like the `prefix`, `diameter`, and `albedo`. The `equinox` column also had only one unique value, which suggests that this feature will not help in our classification problem, so it was removed too. Names and identifier columns, `id`, `spkid`, `name`, `pdes`, `full_name`, and `prefix`, of the asteroids were removed as well as none of that information will give insight into its potential hazard. Additionally, some columns are essentially duplicates, of each other since they are expressing the same data in different units or a different format. Lastly, all the `sigma` columns were removed, as these are measurement uncertainties of existing attributes and will not give us anymore information for our classifier. In the end, we decided that we had more than enough data (932335) compared to the original size (958524) to just drop the remaining rows that had `NaN` data without having to do any imputation. There are only 18 columns left after removal of columns as well. Since we tried out two different models, additional data preprocessing steps follow below, specific to each model.

By running a simple line of code: `df[df.duplicated()]` we were able to verify that no duplicate entries exist in the dataset at all.  
You can check out the Preprocessing code **[here](https://github.com/balexdalex42/CSE151A-Group-Project-Gordon-Yang-Nafi-Mahbub-Alex-Tatoian-/blob/main/ms4.ipynb)** or also **[here](https://github.com/balexdalex42/CSE151A-Group-Project-Gordon-Yang-Nafi-Mahbub-Alex-Tatoian-/blob/Milestone3/prep_dataset.ipynb).**


### Model 1: Decision Tree Classifier
The notebook for the Decision Tree Classifier can be found **[here](/ms3decisiontree.ipynb)**.

For the decision tree we finished preprocessing by imputing missing numeric values with the median and categorical values with the most frequent value fit on the training split. We also one hot encoded NEO and equinox so the model sees clean numeric inputs and added a simple log transform to skewed positive columns diameter, moid_ld, a and q to reduce skew and keep splits from being driven by extreme values. We did not scale features since trees are scale independent, ID fields stayed dropped and any impossible physical values were turned into missing values and handled by the computer.

### Model 2: K-Nearest_Neighbors
The notebook for the KNN Classifier can be found **[here](https://github.com/balexdalex42/CSE151A-Group-Project-Gordon-Yang-Nafi-Mahbub-Alex-Tatoian-/blob/main/ms4.ipynb)**.

For the KNN Classifier, we first preprocessed the data by applying one-hot encoding to all categorical variables that were retained. We then separated the output class and performed an 80/20 train-test split. To reduce dimensionality, we applied PCA, reducing the original 27 features to a maximum of 8. We performed PCA using a scaled SVD on both the training and testing inputs, as PCA requires scaled data. Next, we iterated over different numbers of PCA dimensions ($k$) (up to 8) and KNN neighbors (num_neighbors) (up to 15). This effectively performed a double cross-validation to determine the optimal `(k, num_neighbors)` pair that maximized the macro F1 score.

### Additional Model Trial: SVM Classifier
The notebook for the SVM trial can be found **[here](/model_svm.ipynb)**.

For the SVM Classifier, all `object` dtype columns, except the output `pha` class, were one-hot encoded, and `float64`, which made up the remainder, were z-score normalized. The `pha` column was encoded such that `'Y' => 1` and `'N' => 0`. At this stage, I did 3:1 of the dataset into training and testing sets. Additionally, due to the huge imbalance issue of our `pha` output class, I also ran an oversampling using the `ADASYN` class from `imblearn.over_sampling` to balance out the output class only on the `X_train` and `y_train` data. This final step pushed the shapes of these to `(699251, 29)` and `(699251,)` respectively.

## Results Section

### Model 1: Decision Tree Classifier 

The decision tree model follows the usual pattern where with very small depth it cannot split enough, so both train and validation errors are high and it underfits. As we grow the depth the train error drops fast and the validation error goes down, then starts creeping back up while train error heads toward zero, which is the overfitting side. We chose the depth right around the bottom of the validation curve where the gap between train and validation is small. So the model sits in the balanced zone rather than the underfit or overfit ends.

After seeing the success of the decision tree classifier, we wanted to give a shot with the Support Vector Machine (SVM) as our output class is binary and the decision boundary would likely be easy to determine, and so we gave it a try below.

Train:
<img width="509" height="192" alt="Screenshot 2025-09-03 at 11 05 37 PM" src="https://github.com/user-attachments/assets/77f50504-37a5-44f8-adba-b1c6788f364b" />
Test:
<img width="509" height="192" alt="Screenshot 2025-09-03 at 11 01 13 PM" src="https://github.com/user-attachments/assets/c6bdb6ba-dfd5-43e7-976d-35942be37b9c" />



### Model 2: K-Nearest-Neighbors

### Additional Model Trial: SVM using Linear and RBF Kernels
Surprisingly, we got a much lower precision for identifying **hazardous** asteroids when running SVM than we did with the decision tree models. Also another surprise was that the kernel choice did not matter. Both linear and rbf kernels resulted in similar precision of 0.35 and 0.36 for **hazardous** asteroid classification. So what this tells us is that this model returns a lot of false positives, meaning we predict a large amount of asteroids to be **hazardous** when in reality, they weren't. So the silver lining here is that in the context of planetary defense, this model works fairly well and errs on the side of caution, as all other metrics, including accuracy, are nearly 100%. Lastly, comparing the training and testing metrics, with all training metrics always at near 100%, this model is **overfitting**.

Train:
<img width="509" height="192" alt="Screenshot 2025-09-03 at 11 05 13 PM" src="https://github.com/user-attachments/assets/16bf630d-7e91-43a4-b789-8d916cfcb938" />
Test:
<img width="509" height="153" alt="Screenshot 2025-09-03 at 11 00 13 PM" src="https://github.com/user-attachments/assets/532450d6-3600-4f93-97aa-1a815db29849" />



## Discussion Section

### Model 1: Decision Tree Classifier

We think the decision tree is a solid first model, as with class weights and light preprocessing it delivers strong recall on PHA and better precision than the SVM run. Most misses are borderline cases that look similar in brightness or orbit, so the tree occasionally flags extra positives. To tighten it up we can prune with ccp alpha and a larger minimum leaf, tune the decision threshold from the validation PR curve and if recall still needs help try some light SMOTE on the training split. After that we can compare the model against a calibrated linear SVM as a fast margin baseline.

### Model 2: K-Nearest-Neighbors

### Additional Model Trial: SVM using Linear and RBF Kernels

We found that SVM was a decent model in terms of the practicality of the model in real life, where there mostly problems with precision: non-hazardous asteroids are likely to be flagged. Our model accuracy was high, thus this ends up being a much better case than hazardous asteroids not being flagged! In order to try to increase our precision, we can potentially use class weights differently as ADASYN makes the model focuses on boosting recall, in favor of precision. We can also potentially decrease our regularization parameter, making the SVM margin wider which might decrease the amount of false positives. Finally, SVM ended up being extremely slow, with our implementation using the linear kernel to be around 70 minutes and the implenetation using the RBF kernel to be around 2 hours! This is a substantial amount of time that would scale very fast, if say, our dataset was much larger. We think that using the super computer, potentially writing a few cuda kernels, or using pytorch (for parallelized matrix multiplication) can reduce this time cost significantly. All it boils down to is how much computing power we need, and unfortunately using our own devices without algorithm optimizations or parallelism will greatly slow training time.

## Conclusion
Given more time, we could experiment with an ANN and some different existing architecture. We think that the non-linearity of the ReLU activation function at each node in each hidden layer may be able to learn complex decision boundaries and better predict hazardous asteroids to be actually hazardous. In addition, considering this is a binary classification, we could also have experimented with a logsitic regression, using a customized binary cross entropy loss function that would heavily penalize the model for predicting that an asteroid is **not hazardous** when it really was **hazardous** so that the model will be more robust and have a better recall and precision metric. Through oversampling methods on SVM, we were able to get recall high enough, but precision remained fairly low, which results in many **hazardous** predictions when the asteroid wasn't. We were honestly very surprised at the success of the decision tree, which was much quicker and had better performance than the slower to train SVM.

## Statement of Collaboration
Gordon: I primarily did the data preprocessing part, dropping unusable feature columns from the dataset and dropping any rows that had empty entries. The notebook I wrote also helped save the preprocessed data to be usable in future model training. In Milestone 3, I experimented with the Support Vector Machine, SVC, from the `sklearn` library and ran the training with both `linear` and `rbf` kernels. For both Milestone 2 and 3, I typed up the writeup for the respective sections that I had done myself for the README.

