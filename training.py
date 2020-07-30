import feature_extract
import numpy as np
import cv2
import os
import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler

X = []
Y = []

extract= feature_extract.Extract(win_size = (64, 64))

pos_folder = "Pos_img"
for i in os.listdir(pos_folder):
    image = cv2.imread(os.path.join(pos_folder, i))
    feature = extract.compute(image)
    X.append(feature)
    Y.append([1])

neg_folder = "Neg_img"
for i in os.listdir(neg_folder):
    image = cv2.imread(os.path.join(neg_folder, i))
    feature = extract.compute(image)
    X.append(feature)
    Y.append([0])
    
X = np.stack(X, axis=0)
Y = np.stack(Y, axis=0)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

shuffler = np.random.permutation(len(X))
X = X[shuffler]
Y = Y[shuffler]

x_train = X[:14500]
y_train = Y[:14500]
x_val = X[14500:16500]
y_val = Y[14500:16500]
x_test = X[16500:]
y_test = Y[16500:]

clf=svm.LinearSVC(loss = 'squared_hinge', penalty = "l2", fit_intercept = False, dual = False)
clf.fit(x_train, y_train)

y_val_predict = clf.predict(x_val)
for i in range(len(y_val_predict)):
    if (y_val_predict[i]!=y_val[i]):
        x_train = np.vstack((x_train, x_val[i]))
        y_train = np.vstack((y_train, y_val[i]))
        
clf.fit(x_train,y_train)

y_test_predict = clf.predict(x_test)
for i in range(len(y_test_predict)):
    if (y_test_predict[i]!=y_test[i]):
        x_train = np.vstack((x_train, x_test[i]))
        y_train = np.vstack((y_train, y_test[i]))
        
clf.fit(x_train,y_train)


file1 = open('model.pkl','wb')
pickle.dump(clf, file1)
file1.close()

file2 = open('scaler.pkl', 'wb')
pickle.dump(scaler, file2)
file2.close()

