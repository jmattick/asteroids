import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from math import sqrt
import heapq

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


class KNN(object):
    """Implementation of K-nearest Neighbor Classifier"""
    def __init__(self, k=5, random_state=1):
        self.k = k
        self.random_state = 1


    def euclidean_dist(self, p1, p2):
        """Calculate euclidean distance between two datapoints"""
        # initialize distance to zero
        dist = 0.0
        # loop through all values in data[pomt
        for i in range(len(p1)-1):
            # add squared difference
            dist += (p1[i] - p2[i])**2
        # return square root of distance
        return sqrt(dist)


    def find_nearest_neighbors(self, train_x, train_y, p):
        """Cacluate k nearest neighbors given a trainin dataset and a test row"""
        # initialize list to act as priority queue of distances
        dists = []
        # loop through the training data
        for (row, y) in zip(train_x, train_y):
            # calculate the euclidean distance at each row
            dist = self.euclidean_dist(row, p)
            # add the distance and data to the priority queue
            heapq.heappush(dists, (dist, row, y))
        # return k nearest neighbors
        return heapq.nsmallest(self.k, dists)


    def fit(self, X, y):
        self.train_x = X
        self.train_y = y


    def predict(self, X):
        # initialize predictions list
        predictions = []
        # loop through each row in the dataset
        for row in X:
            # get k nearest neighbors
            knn = self.find_nearest_neighbors(self.train_x, self.train_y, row)
            # dictionary to hold votes for target values
            votes = dict()
            # loop through all neighbors
            for n in knn:
                # get target value
                value = n[-1]
                # increment target vote by 1 if already in dictrionary
                if value in votes:
                    votes[value] = votes[value] + 1
                # else initialize vote to 1
                else:
                    votes[value] = 1
            # append target value with max votes to predictions
            predictions.append(max(votes, key=votes.get))
        return predictions


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


def test_k(train_X, val_X, train_y, val_y,k_list=[5, 6, 7, 8, 9, 10]):
    """Function to test SVM using different regularization parameters"""
    res = []
    for k in k_list:
        model = KNN(k=k)
        res.append(test_model(train_X, val_X, train_y, val_y, model))

    return k_list, res


# plot SVM accuracy vs C
c_list, acc = test_svm_c(X_train, X_dev, y_train, y_dev)
plt.plot(c_list, acc)
plt.xscale('log')
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.savefig("svm_c_tests.png")
plt.close()

# test models on development dataset
# print('Accuracy using development dataset')

# default SVM
svm = SVC()
baseline_stratified = DummyClassifier(strategy='stratified')
baseline_most_frequent = DummyClassifier(strategy='most_frequent')
# print('svm default:')
# print(test_model(X_train, X_dev, y_train, y_dev, svm))

# SVM
svm_c100 = SVC(C=100)
baseline_stratified = DummyClassifier(strategy='stratified')
baseline_most_frequent = DummyClassifier(strategy='most_frequent')
# print('svm C=100:')
# print(test_model(X_train, X_dev, y_train, y_dev, svm_c100))

# KNN model (k=8)
knn = KNN(k=8)
# print('knn k=8:')
# print(test_model(X_train, X_dev, y_train, y_dev, knn))

# plot KNN accuracy vs C
k_list, acc = test_k(X_train, X_dev, y_train, y_dev)
plt.plot(k_list, acc)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.savefig("knn_k_tests.png")
plt.close()

# baseline models
# print('baselines:')
# print(test_model(X_train, X_dev, y_train, y_dev, baseline_stratified))
# print(test_model(X_train, X_dev, y_train, y_dev, baseline_most_frequent))

# test models on development dataset
# print('Accuracy using test dataset')
test_acc = []
models = ['SVM (C=100)', 'KNN (k=8)', 'Stratified', 'Most Frequent']
test_acc.append(test_model(X_train, X_test, y_train, y_test, svm_c100))
test_acc.append(test_model(X_train, X_test, y_train, y_test, knn))
test_acc.append(test_model(X_train, X_test, y_train, y_test, baseline_stratified))
test_acc.append(test_model(X_train, X_test, y_train, y_test, baseline_most_frequent))

# print(test_acc)

# plot accuracy for each model
plt.bar(models, test_acc)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy of model")
plt.savefig("model_comparison_test.png")
plt.close()

