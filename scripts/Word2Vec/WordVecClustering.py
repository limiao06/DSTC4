#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
A module for globally sharing Word Vectors

Miao Li
limiaogg@126.com
'''
import time, sys
from WordVecs import GetModel
from sklearn.cluster import KMeans


def Clustering(n_cluster):
	model = GetModel()
	word_vectors = model.syn0

	start = time.time()
	kmeans_clustering = KMeans( n_clusters = num_clusters )
	idx = kmeans_clustering.fit_predict( word_vectors )

	# Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	print >>sys.stderr, "Time taken for K Means clustering: ", elapsed, "seconds."

	word_centroid_map = dict(zip(model.index2word, idx ))


