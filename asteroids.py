import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

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
            'Mean Motion']

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


def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the accuracy of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return accuracy_score(val_y, val_predict)


def test_svm_c(train_X, val_X, train_y, val_y,c_list=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]):
    """Function to test SVM using different regularization parameters"""
    res = []
    for c in c_list:
        model = SVC(C=c)
        res.append(test_model(train_X, val_X, train_y, val_y, model))

    return c_list, res

c_list, acc = test_svm_c(X_train, X_dev, y_train, y_dev)
plt.plot(c_list, acc)
plt.xscale('log')
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.savefig("svm_c_tests.png")
plt.close()
print(c_list)
print(acc)

# default SVM
svm = SVC()
baseline_stratified = DummyClassifier(strategy='stratified')
baseline_most_frequent = DummyClassifier(strategy='most_frequent')

print(test_model(X_train, X_dev, y_train, y_dev, svm))
print(test_model(X_train, X_dev, y_train, y_dev, baseline_stratified))
print(test_model(X_train, X_dev, y_train, y_dev, baseline_most_frequent))
