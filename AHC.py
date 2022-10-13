#!/usr/bin/env python
# coding: utf-8

# In[5]:


from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import cdist


# In[6]:


nci_data = np.genfromtxt("ncidata.txt")
nci_data = nci_data.T
print(nci_data.shape)


# In[7]:


from collections import OrderedDict
import itertools
from itertools import permutations

class Node:
    ## PART A
    def __init__(self, data, indices, height=0, left=None, right=None):
        self.indices = indices
        self.left_node   = left
        self.right_node  = right
        self.observations = indices # storing the indices is far more data efficient than storing the values
        self.height       = height
        if self.indices.shape[0]==1:
            self.is_leaf = True        
            self.centroid = data[self.indices, :].mean(axis=0)
        else:
            self.is_leaf = False 
            self.centroid = data[self.indices, :].mean(axis=0)
    
    ## PART B
    def split_cluster():
        # this
        pass
        
class AHC: ## PART A & PART B & PART C
    def __init__(self, linkage_function="centroid linkage"):
        self.linkage_function_name = linkage_function
        
    def create_dendrogram(self, data):        
        self.linkage_value_holder = {}
        
        # to start - make every item into a node in the dataset
        # I choose to store nodes in a dictionary, such that they can be removed easily via their names
        # when they merge into nodes above
        node_tracker = OrderedDict({}) # cannot use dict comprehensive with OrderedDict sadly
        for ind, x in enumerate(data): # loop through the data, making a node for every data point 
            # the node will itself detect it is a leaf node due to the fact it recieves an array with shape[0]=1 
            node_tracker["node_"+str(ind)] = Node(data=data, indices=np.array([ind])) 
            
        # we track nodes via their key in the dictionary - this makes it easy to remove them when needed    
        node_value = data.shape[0]
        
        self.get_all_distances(data) # make the distance array which stores all the distances
        i = 0
        self.all_node_holder = node_tracker.copy()
        while i < 100000:
            self.remaining_nodes = len(node_tracker)
            #print("\nRemaining nodes: {}".format(len(node_tracker)), end="")
                
            i+=1
            if len(node_tracker)==1:
                break     
            
            # linkage function tells us which nodes are the cloest 
            min_linkage_pair, min_linkage_value = self.calculate_linkage(node_tracker)
            
            node_1 = node_tracker[min_linkage_pair[0]]
            node_2 = node_tracker[min_linkage_pair[1]]
            
            # create the new indices
            new_indices = np.concatenate((node_1.indices, node_2.indices))
            
            # create a new new node which 'owns' the lower nodes
            node_tracker["node_"+str(node_value)] = Node(data=data, 
                                                         indices=new_indices, 
                                                         left=node_1, 
                                                         right=node_2, 
                                                         height=min_linkage_value)
            self.all_node_holder["node_"+str(node_value)] = node_tracker["node_"+str(node_value)]
            
            # remove the other nodes
            node_tracker.pop(min_linkage_pair[0])
            node_tracker.pop(min_linkage_pair[1])
            
            node_value+=1
            
        return node_tracker["node_"+str(node_value-1)]
    
    def calculate_linkage(self, node_tracker):
        # the make pairs method tells us which pairs of nodes we need to compare
        # this changes every iteration as nodes are created and deleted
        if self.linkage_function_name=="centroid linkage":
            
            pairs = self.make_pairs(list(node_tracker.keys()))
            min_distance = np.inf
            for pair in pairs:
                # get the distance between the centroids of two nodes
                distance = self.get_distance(node_tracker[pair[0]].centroid, node_tracker[pair[1]].centroid)
                
                if distance<min_distance:
                    min_distance = distance
                    key_tuple = pair
            print("pair: ", pair, ", distance: ", min_distance)
            print(node_tracker[pair[0]].centroid)
            print(node_tracker[pair[1]].centroid)
            return key_tuple, min_distance
        else:
            # for these three linkage methods, a distance array between all points is extremely helpful
            # it means we only work out these distances once, and compare the distances needed by using
            # the respective nodes indices
            pairs = self.make_pairs(list(node_tracker.keys()))
            min_distance_found = np.inf
  
            for pair in pairs:
                indices_1 = node_tracker[pair[0]].indices
                indices_2 = node_tracker[pair[1]].indices
                
                # the indices allow us to find the ind
                
                if self.linkage_function_name=="simple linkage":
                    min_distance = self.distance_array[indices_1[:, None], indices_2[None, :]].min()
                elif self.linkage_function_name=="complete linkage":
                    min_distance = self.distance_array[indices_1[:, None], indices_2[None, :]].max()
                elif self.linkage_function_name=="average linkage":
                    min_distance = self.distance_array[indices_1[:, None], indices_2[None, :]].mean()
                    
                if min_distance<min_distance_found:
                    min_distance_found = min_distance
                    key_tuple = pair
                        
            return key_tuple, min_distance_found
        
    # when given keys of the nodes, this presents us with pairs (key_1, key_2) of which we need to compare
    def make_pairs(self, keys):
        pairs = []
        for i in range(len(keys)):
            for j in range(i+1,len(keys)):
                pair = (keys[i],keys[j])
                pairs.append(pair)
        return pairs

    # simple method to get a distance
    def get_distance(self, array_1, array_2):
        return (((array_1 - array_2)**2).sum())**0.5
    
    # makes all the distances between the points
    # the cdist method does this very quickly as it is written in C
    def get_all_distances(self, data):
        self.distance_array = cdist(data, data)
     


# In[8]:


ahc = AHC(linkage_function="centroid linkage")
dendrogram = ahc.create_dendrogram(nci_data)


# In[9]:


def getClusters(dendrogram, K): ## PART B
    
    # this loops through the dendrogram list, splitting the dendrogram into two that is the highest
    # it stops when the number of dendrograms satifies the number desired (K)
    
    dendrogram_list = [dendrogram]
    i = 0
    while len(dendrogram_list)<K or i > 100000:
        i+=1
        dend_heights = [dend.height for dend in dendrogram_list]
        print("cluster heights: ", dend_heights)
        max_index = np.argmax(dend_heights)
        dend_to_split = dendrogram_list.pop(max_index)
        dendrogram_list.append(dend_to_split.left_node)
        dendrogram_list.append(dend_to_split.right_node)
    return {'cluster_'+str(ind):dend.indices for ind, dend in enumerate(dendrogram_list)}

print("clusters: ", getClusters(dendrogram, 25))


# In[10]:


print("Centroid linkage")

ahc = AHC(linkage_function="centroid linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 10))

print("\n")

ahc = AHC(linkage_function="centroid linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 25))

print("\n\n")


# In[11]:


print("Average linkage")

ahc = AHC(linkage_function="average linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 10))

print("\n")

ahc = AHC(linkage_function="average linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 25))

print("\n\n")


# In[12]:


print("Complete linkage")

ahc = AHC(linkage_function="complete linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 10))


print("\n")

ahc = AHC(linkage_function="complete linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 25))

print("\n\n")


# In[15]:


print("Simple linkage")

ahc = AHC(linkage_function="simple linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 10))

print("\n")

ahc = AHC(linkage_function="simple linkage")
dendrogram = ahc.create_dendrogram(nci_data)

print("clusters: ", getClusters(dendrogram, 25))

print("\n\n")

