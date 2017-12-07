
# coding: utf-8

# In[16]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
mpl.rc('figure', figsize=[10,6])
from sklearn.linear_model import LogisticRegression


# In[17]:

df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')


# In[18]:

#get sample
X = df.iloc[:, range(2,32)]
y = df.color[:]

#mean-variance normalization
Xn = (X - X.mean(axis = 0))/X.std(axis = 0)

#Add ones
Xn = np.hstack((np.ones((Xn.shape[0],1)), Xn))

#thetas?
Xn.shape


# In[19]:

def J_logistic(X, theta, y):
    h = 1/(1+np.exp(-X * theta))
    y = np.asarray(y)
    h = np.asarray(h)
    J = (-y*np.log(h) - (1-y)*np.log(1-h)).mean(axis=0)
    return J


# In[20]:

def gradient_descent(X, y, alpha, max_iter):
    y = np.asmatrix(y).T
    X = np.asmatrix(X)
    m = X.shape[0]
    theta = np.asmatrix([0.0]*X.shape[1]).T
    
    if theta.shape[0] != X.shape[1] or X.shape[0] != y.shape[0] or y.shape[1] != 1:
        print("incompatible shapes X {}, y {}, theta {}".format(X.shape, y.shape, theta.shape))
    
    costs = []
    thetas = []
    
    for i in range(max_iter):
        h = 1/(1 + np.exp(-X * theta))
        theta += (alpha/m) * X.T * (y-h)
        temp_cost = J_logistic(X,theta,y)
        costs.append(temp_cost)
        thetas.append(theta.copy())
    
    #plot costs vs iteration
    plt.figure()
    plt.plot(costs)
    plt.title("cost vs iterations for alpha = 3")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    
    
    return costs, thetas


# In[21]:

costs, thetas = gradient_descent(Xn, y, 3, 1000)


# In[22]:

thetas


# In[27]:

clf1 = LogisticRegression()
clf1.fit(Xn, y)
clf1.coef_


# In[4]:

c1 = 'mradius'
c2 = 'mtexture'

clf = LogisticRegression()
clf.fit(df[[c1, c2]], df['color'])

plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

x = np.linspace(df[c1].min(), df[c1].max(), 1000)
y = np.linspace(df[c2].min(), df[c2].max(), 1000)
xx, yy = np.meshgrid(x,y)
predictions = clf.predict(
        np.hstack(
            (xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1))
        ))
predictions = predictions.reshape(xx.shape)

plt.contour(xx, yy, predictions, [0.0])


# In[ ]:



