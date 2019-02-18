# ContextBasedKmeans-
a new algorithm for retaining the context of words when clustering them in NLP Algorithms for optimal classification.
Utilized:
Stanford’s GloVe dataset for global context, executed K Means++ on it and used Tsne for dimension reduction and cluster plotting.
These are the steps
involved in the process as a whole:
1. The GloVe dataset was downloaded. The option rests in choosing one of 50D,100D,200D or 300D. Based on clustering results in this particular
implementation we have taken the 300D data file. The GloVe data file contains a large number of pre-trained word vector representations. GloVe is
a form of unsupervised learning algorithm. The GloVe file can be downloaded from glove.6B.zip. We create the word vector matrix for the GloVe
file and zoo dataset to get labels as well as vectors in the GloVe file.
2. We then add a file on which we want to perform the clustering. Now the basic idea here is to perform clustering for non-numeric data. We have
taken the zoo dataset from UCI machine learning kmeans++ used for clustering. We add the data file as a pickled file here. A picked file is a type that is
native to python can directly convert imported file.
3. The words in the zoo dataset are compared with those in the GloVe dataset. Word embeddings refers to an attribute of Natural Language
Processing where words are given a real number vector representation and aspires to club semantic alikeness among linguistic things. The related
word embeddings are generated for each of the animals in the zoo dataset using the GloVe vector representation of those words.
4. We then use TSNE for reducing the dimensionality of the data here to two dimensions. TSNE lowers the dimensions of the word vectors
and gives good visualization. It is a non-linear method that is greatly suitable for converting data with high dimensions to two or three dimensions
and is further generating plots for ease of visualization. It puts the similar data at points close in distance and the dissimilar data is placed
at a distance apart. 
5. The data is then clustered using K means++ unlike the traditional K means algorithm used in papers before. K means++ has method for choosing the
initial values of cluster center thus overcoming the major problem faced by K means of generating low quality clusters at times in terms of
optimality. We apply K means++ on the word vectors of the pickled file to generate their clusters based on similarity between them.
6. The K means++ algorithm has issue of depicting the number of clusters to be formed for the minimum inertia and least no of clusters in total.
We have used the Elbow method here for deciding the number of clusters formed by the algorithm. We generate a graph and find the optimal number
of clusters using the elbow method.

Results:
The dataset objects are then clustered using K Means++ by using the word embeddings generated previously. This method is superior to those
used previously as firstly it is it regards dependency within the data at a more refined level than in case of K means algorithm. Further, 
by augmenting K means with the arbitrary seeding method we get K means ++ which is O (log k) competitive to the optimum clustering.
An improvement in both the fastness as well as precision is seen when switching from K means to context aware K means++ clustering. 
The clustering here can work on synthetic data and can hence be applied for confidential datasets as well.
