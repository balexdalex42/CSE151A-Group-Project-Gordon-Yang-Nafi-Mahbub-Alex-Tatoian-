# CSE151A-Group-Project-Gordon-Yang-Nafi-Mahbub-Alex-Tatoian-
## Environment Setup Requirements
### Dataset download
The entire dataset from kaggle can be found [here](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset). At the top left of the screen, click **Download** and then **Download dataset as zip**. Unzip the `archive` folder into the main project directory.
### Directory structure
The following directory structure must be followed when downloading the dataset:
```
CSE151A-Group-Project-Gordon-Yang-Nafi-Mahbub-Alex-Tatoian-/
├─ archive/
│  ├─ dataset.csv
├─ README.md
├─ ms2.ipynb
├─ .gitignore
```

### Other requirements
- Have Python 3 installed on your machine.
- Install the pandas, numpy, and matplotlib libraries

## Data Exploration
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

## Data Plotting:

**All data plots are on the [ipynb file](/ms2.ipynb) in the same branch**

## Preprocessing Plan:

We plan to use PHA as the label by mapping Y to 1 and N to 0 and drop any row with a missing or invalid PHA. We would have to drop ids spkid, id, pdes, name, full_name, and orbit_id since they do not explain behaviour and could potentially act as noise and trick our model. We will set types correctly by treating NEO, PHA, and Equinox as categorical and coercing H, Diameter, Albedo, Diameter_sigma, Epoch, e, a, q, i, tp, and moid_ld to numbers. Missing data can be handled with averages from the training split with numeric features using the median and categorical features using the mode. Columns that are mostly missing and low value will be removed and rows with mostly missing data will be removed aswell.

We can de duplicate by using a stable key such as spkid or pdes with name and keep the most complete or most recent record by Epoch. We will try to enforce sensible ranges so impossible values become missing and are removed, we can check for outliers with simple interquartile rules and cap extremes after we split using training statistics.

We might standardize numeric features with a z score and one hot encode NEO and Equinox, using the training validation test split on PHA and fit every preprocessing step on the training split so we can avoid leakage from validation or test splits. We expect PHA equals 1 to be rare and will need to measure the class ratio and keep it through stratified splits and enable class weights during modeling. If needed we can also rebalance the training data with SMOTE and light undersampling. We will set the decision threshold from validation and report precision recall area under the curve, recall, and F1 since they reflect performance on rare positives, instead of plain accuracy.
