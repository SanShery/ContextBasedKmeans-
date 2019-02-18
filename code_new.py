from __future__ import division
from sklearn.cluster import KMeans 
from numbers import Number
#from pandas import DataFrame
import sys, codecs, numpy
import sklearn
from sklearn.manifold import TSNE
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


class autovivify_list(dict):
  '''A pickleable version of collections.defaultdict'''
  def __missing__(self, key):
    '''Given a missing key, set initial value to an empty list'''
    value = self[key] = []
    return value

  def __add__(self, x):
    '''Override addition for numeric types when self is empty'''
    if not self and isinstance(x, Number):
      return x
    raise ValueError

  def __sub__(self, x):
    '''Also provide subtraction method'''
    if not self and isinstance(x, Number):
      return -1 * x
    raise ValueError
    
def build_word_vector_matrix(vector_file, n_words,datafile):
  '''Return the vectors and labels for the first n_words in vector file'''

  zoo_array = []
  
  numpy_arrays = []
  labels_array = []

  with codecs.open(datafile, 'r', 'utf-8') as f1:
    for c1, r in enumerate(f1):
      srz = r.split(',')
      #print("found "+srz[0]+"\n")
      zoo_array.append(srz[0])
 
  print(len(zoo_array))
 
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()

      if sr[0] in zoo_array:
        labels_array.append(sr[0])
        #print("found "+sr[0]+"\n")
        numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )

        if c == n_words:
          return numpy.array( numpy_arrays ), labels_array
  print(len(labels_array))
  return numpy.array( numpy_arrays ), labels_array
  
def find_word_clusters(labels_array, cluster_labels):
  '''Return the set of words in each cluster'''
  cluster_to_words = autovivify_list()
  for c, i in enumerate(cluster_labels):
    cluster_to_words[ i ].append( labels_array[c] )
  return cluster_to_words
  
def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass  
  
if __name__ == "__main__":
  input_vector_file = sys.argv[1] # Vector file input (e.g. glove.6B.300d.txt)
  n_words = int(sys.argv[2]) # Number of words to analyze 
  reduction_factor = float(sys.argv[3]) # Amount of dimension reduction {0,1}
  datafile = sys.argv[4]
  
  df, labels_array = build_word_vector_matrix(input_vector_file, n_words,datafile)

  tsne = TSNE(n_components =2,random_state = 0) # dimensions reduction
  df_3d = tsne.fit_transform(df)
  
  n_clusters =3 # Number of clusters to make
  kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
  kmeans_model.fit(df_3d)
  
  cluster_labels  = kmeans_model.labels_
  cluster_inertia   = kmeans_model.inertia_
  cluster_to_words  = find_word_clusters(labels_array, cluster_labels)
  

  for c in cluster_to_words:
    print(cluster_to_words[c])
    print("\n")
    

  tsne = TSNE(n_components = 2,random_state = 0)
  df_2d = tsne.fit_transform(df)


  plt.scatter(x=df_2d[:,0],y=df_2d[:,1],c=cluster_labels)

  plt.show()
