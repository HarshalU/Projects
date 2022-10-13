#!/usr/bin/env python
# coding: utf-8

# In[51]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np


# In[52]:


X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=85)


# In[53]:


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[54]:


# calculate the rss when give the attribute and s, and the x and y data
def calculate_RSS(X_rss, y_rss, attr, s):
    d_1 = y_rss[X_rss[:, attr]<=s]
    d_2 = y_rss[X_rss[:, attr]>s]
    
    if len(d_1)==0:
        rss_1 = 0
    else:
        rss_1 = ((d_1 - d_1.mean())**2).sum()
        
    if len(d_2)==0:
        rss_2 = 0
    else:
        rss_2 = ((d_2 - d_2.mean())**2).sum() 
    
    return rss_1 + rss_2


# In[399]:


class TreeNode:
    # treenode - automatically trains itself by building the tree beneath itself when initialised in a recursive manner
    def __init__(self, X, y, depth_level, max_depth=3, features_sample=None):
        self.depth_level = depth_level
        self.max_depth   = max_depth
        if features_sample:
            self.features_sample = X.shape[1]
        else:
            self.features_sample = X.shape[1]/3
            
        if depth_level==max_depth or len(y)<=1:
            self.type             = "leaf"
            self.prediction_value = np.mean(y)
            self.y                = y
        else:
            self.type  = "split"
            
            self.split_info = self.create_split(X_create_split=X, y_create_split=y)

            left_indices  = X[:, self.split_info['best_attribute']]<=self.split_info['best_split_value']
            right_indices = X[:, self.split_info['best_attribute']]>self.split_info['best_split_value']
            
            self.X_left = X[left_indices, :]
            self.y_left = y[left_indices]

            self.X_right = X[right_indices, :]
            self.y_right = y[right_indices]
            
            if len(self.y_right)==0 or len(self.y_left)==0:
                self.type             = "leaf"
                self.prediction_value = np.mean(y)
                self.y                = y
            else:
                self.create_lower_branches()
    
    # create lower branches - makes two new tree nodes below itself using the respective subsets of the data
    def create_lower_branches(self):
        self.left  = TreeNode(self.X_left, 
                              self.y_left, 
                              depth_level=self.depth_level+1,
                              max_depth = self.max_depth)
        self.right = TreeNode(self.X_right, 
                              self.y_right, 
                              depth_level=self.depth_level+1,
                              max_depth = self.max_depth)
    
    # searches through the attributes and data to create the split
    def create_split(self, X_create_split, y_create_split):
        min_RSS = np.inf
        size = int(np.round(self.features_sample))
        attributes_considered = np.random.choice(X_create_split.shape[1], size=size, replace=False)
        for attribute in attributes_considered:
            unique_values = np.unique(X_create_split[:, attribute])
            
            for s in unique_values:
                rss = calculate_RSS(X_create_split, y_create_split, attr=attribute, s=s)
                if rss < min_RSS:
                    min_RSS = rss
                    best_attribute = attribute
                    best_split_value = s
        return {'min_RSS': min_RSS, 
                'best_attribute': best_attribute, 
                'best_split_value': best_split_value}
    
    # predicts for a single observation
    # again works recursively, retrieving the prediction value of the relevant branch for each split
    def predict(self,X):
        if self.type=="leaf":

            return self.prediction_value
        else:
            
            if X[self.split_info['best_attribute']] <= self.split_info['best_split_value']:

                return self.left.predict(X)
            else:

                return self.right.predict(X)
    
    # predicts for a group of observation
    def predict_set(self, X):
        return np.array([self.predict(x) for x in X])
    

        


# In[410]:


## to test the Decision Tree, the mean squared error on the training set for a very high depth was checked - 
## this should be 0 as there should be a leaf made for each value in the dataset

treenode = TreeNode(X = X_train[:], y = y_train[:], features_sample=True, depth_level=0, max_depth=5000) 
predictions = treenode.predict_set(X_train)
print("Acccuracy for very high depth on train: ", ((predictions-y_train)**2).mean())
print("(Should be close to 0)")


# In[303]:


# checking the MSE on the test set for a single tree
treenode = TreeNode(X = X_train[:], y = y_train[:], depth_level=0, max_depth=3) 
predictions = treenode.predict_set(X_test)
((predictions-y_test)**2).mean()


# In[347]:


class RandomForest:
    # the random forest uses a collection of trees, where each tree sees a boostrap of the data (selection with replacement)
    # and each split sees a random subset of the features
    
    def __init__(self, number_trees=100, data_proportion_per_tree=0.66, max_depth=3):
        self.number_trees = number_trees
        self.data_proportion_per_tree = data_proportion_per_tree
        self.amount_per_tree = int(data_proportion_per_tree*X_train.shape[0])
        self.max_depth = max_depth
    
    # train by making the trees and store them
    def train(self, X, y):
        self.trees = []
        for _ in range(self.number_trees):
            indices_to_send = np.random.choice(X_train.shape[0], size = self.amount_per_tree, replace=True)
            
            # this initialisation of the tree automatically trains it
            tree = TreeNode(X = X_train[indices_to_send], y = y_train[indices_to_send], depth_level=0, max_depth=self.max_depth)
            self.trees.append(tree)
            
    # predict by predicting for each tree, then takng the mean
    def predict(self, X):
        return np.array([np.mean([tree.predict(x) for tree in self.trees]) for x in X])


# In[369]:


# function to get MSE quickly
def get_MSE(y_predictions, y_labels):
    return ((predictions-y_labels)**2).mean()


# In[415]:


train_mse_depth = []
test_mse_depth  = []

# loop to explore effect of depth on train and test MSE
for depth in range(2, 30):
    print("\rDoing depth: {}".format(depth), end="")
    rf = RandomForest(number_trees=100, data_proportion_per_tree=0.66, max_depth=depth)
    rf.train(X_train, y_train)
    predictions = rf.predict(X_train)
    train_mse_depth.append(get_MSE(predictions, y_train))
    predictions = rf.predict(X_test)
    test_mse_depth.append(get_MSE(predictions, y_test))


# In[416]:


import matplotlib.pyplot as plt

plt.title("MSE against depth")
plt.plot(list(range(2, 30)), train_mse_depth, label="Train")
plt.plot(list(range(2, 30)), test_mse_depth, label="Test")
plt.ylabel("MSE")
plt.legend()
plt.xlabel("Depth")


# In[382]:


train_mse = []
test_mse  = []

# loop to explore effect of number of trees on train and test MSE
number_trees_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]
for number_trees in number_trees_list:
    print("\rDoing number_trees: {}".format(number_trees), end="")
    rf = RandomForest(number_trees=number_trees, data_proportion_per_tree=0.66, max_depth=8)
    rf.train(X_train, y_train)
    predictions = rf.predict(X_train)
    train_mse.append(get_MSE(predictions, y_train))
    predictions = rf.predict(X_test)
    test_mse.append(get_MSE(predictions, y_test))


# In[383]:


import matplotlib.pyplot as plt

plt.title("MSE against number trees")
plt.plot(number_trees_list, train_mse, label="train")
plt.plot(number_trees_list, test_mse, label="test")
plt.ylabel("MSE")
plt.legend()
plt.xlabel("Number of trees")

