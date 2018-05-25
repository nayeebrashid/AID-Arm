import numpy as np
import pandas as pd
path = "day6.csv"
names = ['min', 'max', 'rms', 'integration', 'class']
dataset = pd.read_csv(path, names=names)

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:, 4].values

X_test = [1.1681, 1.7986,1.5918, 157.7810]
X_test = np.asarray(X_test)
X_test = X_test.reshape(1,-1)

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10, random_state=42)

from sklearn import svm

model = svm.SVC()

model.fit(X,y)

training_acc = model.score(X,y)
#validation_acc = model.score(X_test,Y_test)

print("training accuracy: %2f"%training_acc)
#print("test accuracy: %2f"%validation_acc)

Y_pred = model.predict(X_test)
#print(Y_test)
print(Y_pred)