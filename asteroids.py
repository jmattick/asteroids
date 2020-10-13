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

