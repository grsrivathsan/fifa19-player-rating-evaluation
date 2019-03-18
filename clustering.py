import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN


class Clustering:

	def __init__(self, file_path, index_col, clusters=None):
		self.file_path = file_path
		self.index_col = index_col
		self.clusters = clusters
		self.data = self._extract_data()

	def _extract_data(self):
		file_data = pd.read_csv(self.file_path, index_col=[self.index_col])
		return file_data

	def agglomerative(self, data, clusters=None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='single')
		# We can change affinity to euclidean,l1,l2,manhatten,cosine and Change linkage to single,complete and average
		cluster.fit_predict(data)
		print(set(cluster.labels_))

	def k_means(self, data, clusters=None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		kmeans = KMeans(n_clusters=clusters)
		kmeans.fit(data)
		print(kmeans.labels_)
		print(kmeans.cluster_centers_)

	def k_means_plus_plus(self, data, clusters=None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		kmeanspp = KMeans(n_clusters=clusters, init='k-means++')
		kmeanspp.fit(data)
		print(kmeanspp.labels_)
		print(set(kmeanspp.labels_))

	def db_scan(self, data, clusters=None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		dbs = DBSCAN(eps=clusters, min_samples=2).fit(data)
		p = dbs.labels_
		print(p)
		print(set(p))

	def run_clustering(self, clusters = None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		skills = self.data.loc[:, "LS":"GKReflexes"]
		self.agglomerative(skills,clusters)
		self.k_means(skills,clusters)
		self.k_means_plus_plus(skills,clusters)

clustering = Clustering("./data_normalized.csv","Id", 7)
clustering.run_clustering()