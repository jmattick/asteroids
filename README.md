# asteroids

Data was downloaded from: https://www.kaggle.com/shrutimehta/nasa-asteroids-classification

The dataset contains information about 4687 asteroids and a hazardous classification. 
The features in the dataset are:
'Neo Reference ID', 'Name', 'Absolute Magnitude', 'Est Dia in KM(min)',
       'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
       'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
       'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date',
       'Epoch Date Close Approach', 'Relative Velocity km per sec',
       'Relative Velocity km per hr', 'Miles per hour',
       'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)',
       'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body',
       'Orbit ID', 'Orbit Determination Date', 'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion', and 'Equinox'.
Some of the features can be eliminated as duplicates since they are the same 
metric using different units (distance, velocity).

The data was read using pandas. Features that were duplicates were dropped in addition to 
features that were the same in every sample (Orbiting body, Equinox). IDs and dates were
also dropped. The data was split into 70% training data, 15% development data, and 
15% test data using sklearn's train_test_split. Data was normalized using sklearn's 
StandardScaler. 

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read data using pandas
data = pd.read_csv("nasa.csv")

# select features
features = ['Absolute Magnitude', 'Est Dia in KM(min)',
       'Est Dia in KM(max)',
       'Epoch Date Close Approach', 'Relative Velocity km per sec',
       'Miss Dist.(kilometers)',
       'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion', 'Hazardous']

# set target
y = data.Hazardous

# subset data
X = data[features]

# split data into 70% training 30% testing / development
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=1, train_size=0.7)

# split test / development data equally
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, random_state=1, train_size=0.5)

# scale data using sklearn StandardScaler
sc = StandardScaler()

# fit scaler on training data
sc.fit(X_train)

# transform training and validation data
X_train = sc.transform(X_train)
X_dev = sc.transform(X_dev)
X_test = sc.transform(X_test)
```

To help test models a function was created to return accuracy 
when given a training and test dataset and a model. 

```python
from sklearn.metrics import accuracy_score

def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the accuracy of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return accuracy_score(val_y, val_predict)
```

The first model tested was an SVM implemented in sklearn  
sklearn.svm.SVC. Using the default parameters, the accuracy was
0.9516 using the development dataset.

```python
from sklearn.svm import SVC

# default SVM 
svm = SVC()

# calculate accuracy
accuracy = test_model(X_train, X_dev, y_train, y_dev, svm)
```

A function was created to help optimize the regularization parameter in the SVM
model. This function returns a list of the values used and accuracy of the models.

```python
def test_svm_c(train_X, val_X, train_y, val_y,c_list=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]):
    """Function to test SVM using different regularization parameters"""
    res = []
    for c in c_list:
        model = SVC(C=c)
        res.append(test_model(train_X, val_X, train_y, val_y, model))

    return c_list, res

c_list, acc = test_svm_c(X_train, X_dev, y_train, y_dev)
```

The best value of C was 100. This resulted in an accuracy of 0.9644.
The accuracy at each value of C is shown in the following plot. 

![c optimization](svm_c_tests.png)

