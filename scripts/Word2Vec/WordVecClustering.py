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


def Clustering(num_clusters, max_iter_num=100, num_init=5):
	kmeans_clustering = KMeans(n_clusters=num_clusters,max_iter=max_iter_num,n_init=num_init,precompute_distances=True)
	
	model = GetModel()
	word_vectors = model.syn0

	start = time.time()
	idx = kmeans_clustering.fit_predict(word_vectors)

	# Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	print >>sys.stderr, "Time taken for K Means clustering: ", elapsed, "seconds."

	word_centroid_map = dict(zip(model.index2word, idx))

	return word_centroid_map,idx


if __name__ =="__main__":
	import argparse, json
	parser = argparse.ArgumentParser(description='Clustering the word2vec vectors.')
	parser.add_argument('--OUT', dest='OUT', required=True, help='The output file.')
	parser.add_argument('--NCluster', dest='NCluster', type=int, required=True, help='The number of clusters.')
	parser.add_argument('--NIter', dest='NIter', type=int, help='The number of max iterations.')
	parser.add_argument('--NInit', dest='NInit', type=int, help='The number of init runs.')
	args = parser.parse_args()

	word_centroid_map, idx = Clustering(args.NCluster, args.NIter, args.NInit)

	out_file = file(args.OUT, 'w')
	out_json = {}
	out_json['wordmap'] = word_centroid_map
	out_json['idx'] = idx
	json.dump(out_json, out_file, indent=4)
	out_file.close()

