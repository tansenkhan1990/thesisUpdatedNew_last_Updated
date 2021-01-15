import numpy as np

#create a 3x4 random matrix
x = np.random.rand(3,4)
y = np.random.rand(3,4)
print('x metrix: ', x)
print('#####################')
print('y metrix: ', y)
print('#####################')
'''
slice a part of these matrix, 
so that we can take only first two 
columns
'''
x = x[:,:2]
y = y[:,:2]
print('after slicing x: ', x)
print('######################')
print('after slicing y: ', y)
print('#######################')
#elementwise multiply these two metrices
z = np.multiply(x,y)
print('final matrix: ', z)
#########################################################
import pandas as pd

#Read the comma seperated dataset using pandas
data = pd.read_table('/home/mirza/PycharmProjects/frame_work/dataset/normal/demo_data', sep=',')

#Show the data
print(data)
print('###############################################')
#Find the row whose value is greater than certain threshold of some particular column
print(data.loc[(data['A']>50) & (data['C']>100)])
################################################################################################
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Load iris dataset, where X is the feature vector and y is the label
X, y = datasets.load_iris(return_X_y=True)
X_reduced = X[:10,:-1]
y_reduced = y[:10]
print('Feature vector: X= ', X_reduced)
print('Their corresponding label: y= ', y_reduced)

#Divide the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.33, random_state=42)

print('Training Feature vector: X= ', X_train)
print('Training label: y= ', y_train)

print('Testing Feature vector: X= ', X_test)
print('Testing label: y= ', y_test)
######################################################################
import torch

# Initialize random 3X4 matrices
x = torch.rand(3, 4)
y = torch.rand(4, 2)
z = torch.rand(3, 4)
# Operations
#add two matrices
k = torch.add(x,z)
#multiply two matrices
l = torch.matmul(x, y)

#Print both results
print(k)
print(l)

