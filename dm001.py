#!/usr/bin/env python
# -*- coding: UTF-8 -*-

''' Machine Learning and Data Mining
    18 May 2016
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, linear_model
from sklearn import neighbors
# from sklearn.neural_network import MLPClassifier
from mlp import MLPClassifier # my own MLP

from matplotlib.colors import ListedColormap
import pdb


def plotclaissified(X, ytarget, y_classified, alpha):
    group0 = np.where(ytarget==0)
    group1 = np.nonzero(ytarget)

    y = y_classified *1
    n = y.size
    idx = np.where(ytarget!=y)
    idx = idx[1]
    wrongsample = idx.size
    errorrate = wrongsample*1.0/n
    fig, ax = plt.subplots()
    ax.scatter(X[group0,0], X[group0,1], color='Green', label='group 0')
    # pdb.set_trace()
    ax.scatter(X[group1,0], X[group1,1], color='MidnightBlue', label='group 1')
    if len(idx) != 0: 
        ax.scatter(X[idx, 0], X[idx, 1], color='OrangeRed', label='misfits')
        msg = 'regularization: ' + str(alpha)+ '   error rate:' + str(errorrate)
        # pdb.set_trace()
        
    else:
			  msg = 'regularization: ' + str(alpha)+ '   no error!'
    plt.title(msg)
    ax.legend()
    return errorrate

# read excel file
with pd.ExcelFile('romyBiVarNormProb4.xls') as xls:
	df = pd.read_excel(xls, 'Data')

'''
 df : DataFrame, automatic indices (1000), 4 columns
 len(df.index) # the other one being: len(df.columns)
'''
# scatter plot
grouped = df.groupby('Group')
fig, ax = plt.subplots()
df.plot(kind='scatter', x='X', y='Y')
for category, group in grouped:
	ax.plot(group['X'], group['Y'], marker='o8'[category], linestyle='', 
	       label='Group'+str(category))
ax.legend()
#plt.show()

# split the training data and the test data
idx_shuffled = np.random.permutation(len(df.index))
df2 = df.reindex(idx_shuffled)
trainingdataframe = df2.iloc[:600]
testdataframe = df2.iloc[600:]
fig2,ax2 = plt.subplots()
trainingdataframe.plot(kind='scatter', x='X', y='Y', ax=ax2,label='training')
plt.hold(True)
testdataframe.plot(kind='scatter', x='X', y='Y', color='Salmon',ax=ax2,label='test')
ax2.legend()

# the data
train_data = trainingdataframe.as_matrix();
test_data = testdataframe.as_matrix();

'''
# Logistic Regression 


logreg = linear_model.LogisticRegression(verbose=1)
logreg.fit(train_data[:,2:], train_data[:,1])

ax.plot(train_data[:,2], (-train_data[:,2]*logreg.coef_[0,0]-
                         logreg.intercept_[0])/logreg.coef_[0,1],
        color='HotPink')

# plt.show()

# KNN (unsupervised) classification with visualisation

cmap_light = ListedColormap(['LightSeaGreen', 'LightSkyBlue'])
cmap_bold = ListedColormap(['SeaGreen', 'SkyBlue'])

h = 0.02 # step size in the mesh
n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
clf.fit(train_data[:,2:], (train_data[:,1]))

# boundary definition
padding = 1
x_min, x_max = train_data[:, 2].min()-padding, train_data[:, 2].max()+padding
y_min, y_max = train_data[:, 3].min()-padding, train_data[:, 3].max()+padding

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

zcolor = clf.predict(np.c_[xx.ravel(), yy.ravel()])
zcolor = zcolor.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, zcolor, cmap=cmap_light)

plt.scatter(train_data[:,2], train_data[:,3],c=train_data[:,1],cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# plt.show()
'''
''' only available in version 0.18.development!

# Neural Networks: scikit-learn mlpclassifier

mlp_clf = MLPClassifier(activation='relu', algorithm='l-bfgs', 
                        alpha=1e-5, hidden_layer_sizes=(100,2),
                        random_state=1, verbose=1)
mlp_clf.fit(train_data[:,2:], train_data[:,1])

res_labels = mlp_clf.predict(test_data[:,2:])
#visual
'''
X = train_data[:, 2:]
y = train_data[:, 1]
alpha = 0.001
mlp_clf = MLPClassifier(activationfunc='relu',
                        tol=1e-4, hidden_layer_sizes=(10,200, 2), 
                        verbose=1, max_iter =500, lam=alpha)
while mlp_clf.trainedflag is False:
	  mlp_clf.fit(X, y)
	  
xtest = test_data[:, 2:]
ytarget = test_data[:, 1]
ytest = mlp_clf.predict(xtest)
y_1 = abs(ytest-1)
y_0 = abs(ytest)
Y = y_1 <= y_0
erate = plotclaissified(X, ytarget, Y, alpha)

plt.show()


