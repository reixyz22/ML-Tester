import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv("data.csv")  # first we load our data in, so we can inspect and modify it
print(df.columns)  # my first observation here with this data is that some of it needs to be modified to properly use it

cols = ["Email",  # this can safely be removed
        "Address",  # I think this should be distilled to state, which then could perhaps become one of around 6 regions
        "Avatar",  # This is a set of possible colors, we will determine the possibilities and act accordingly
        "Time on App",  # the rest of the data should be easier to work with.
        "Time on Website",
        "Length of Membership",
        "Yearly Amount Spent"
        ]

df = df.drop(["Email"], axis=1)  # Email is of no value to the neural net, as a feature.
df = df.drop(["Address"], axis=1)  # Best to revisit this one later.

print(df["Avatar"].unique())
df = df.drop(["Avatar"], axis=1)  # Given that there are far too many unique colors it's also Best to revisit this
# one later.
print(df.head())

for label in cols[:-1]:  # these three graphs help us understand the distribution of our data.
    if label in df.columns:
        plt.hist(df[label], color='blue', alpha=0.7, density=True)
        plt.title(f'Histogram of {label}')
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.show()

for label in cols[:-1]:  # This helps us visualize the relations to our target variable.
    if label in df.columns:
        plt.figure(figsize=(10, 6))  # Creates a new figure with a specified size
        plt.scatter(df[label], df['Yearly Amount Spent'], alpha=0.5)
        plt.title(f'Scatter Plot of {label} vs. Yearly Amount Spent')
        plt.xlabel(label)
        plt.ylabel('Yearly Amount Spent')
        plt.grid(True)  # Optional, adds a grid to the plot for better readability
        plt.show()

#  time on website doesn't appear to be useful to us given its cloud like shape.
df = df.drop(["Time on Website"], axis=1)

# we now only have 2 features. I feel it's best to start by testing a Support Vector

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


# first let's split the data properly # uses the .sample function from np to set segments splits at 60%,80%

def split(dataframe):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_train)
    y = scaler.transform(y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y


train, X_train, y_train = split(train)
valid, X_valid, y_valid = split(valid)
test, X_test, y_test = split(test)



svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
# print(classification_report(y_test, y_pred))
