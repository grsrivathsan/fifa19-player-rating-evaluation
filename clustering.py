import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


class Clustering:

	def __init__(self, file_path, index_col, clusters=None):
		self.file_path = file_path
		self.index_col = index_col
		self.clusters = clusters
		self.data = self._extract_data()
		self.skills = self.data.loc[:, "Crossing":"GKReflexes"]

		self.POS_ATTACK = "Attack"
		self.POS_MID = "Mid"
		self.POS_DEF = "Def"
		self.POS_GK = "GK"
		self.skills_attack = ["Agility", "Balance", "Acceleration", "SprintSpeed", "Dribbling", "BallControl",
							  "Jumping", "Finishing", "ShotPower", "Positioning"]
		self.skills_mid = ["Balance", "ShortPassing", "Agility", "Stamina", "Acceleration", "SprintSpeed",
						   "BallControl", "Dribbling", "Jumping", "Aggression"]
		self.skills_def = ["SprintSpeed", "Acceleration", "Stamina", "Agility", "Balance", "Jumping", "Strength",
						   "Aggression", "StandingTackle", "HeadingAccuracy"]
		self.skills_gk = ["GKReflexes", "GKDiving", "GKPositioning", "GKHandling", "GKKicking"]

		self.data_by_pos = self.split_data_by_positions(self.data)
		self.run_pca(self.skills, 3)

	def _extract_data(self):
		file_data = pd.read_csv(self.file_path, index_col=[self.index_col])
		return file_data

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

	def split_data_by_positions(self, data):
		data_by_pos = dict()
		data_by_pos[self.POS_ATTACK] = data[data['Position'].str.endswith('S', na=False)]
		data_by_pos[self.POS_ATTACK].append(data[data['Position'].str.endswith('F', na=False)])
		data_by_pos[self.POS_ATTACK].append(data[data['Position'].str.endswith('T', na=False)])

		data_by_pos[self.POS_MID] = data[data['Position'].str.endswith('M', na=False)]
		data_by_pos[self.POS_DEF] = data[data['Position'].str.endswith('B', na=False)]
		data_by_pos[self.POS_GK] = data[data['Position'].str.endswith('K', na=False)]
		return data_by_pos

	def pca(self, data, components):
		scaler = StandardScaler()
		scaler.fit(data)
		scaled_data = scaler.transform(data)
		pca = PCA(n_components=components)
		pca.fit_transform(scaled_data)
		print(pca.components_)
		print()
		print(pca.explained_variance_ratio_)
		self.plot_pca(scaled_data[:, 0:2], np.transpose(pca.components_[0:2, :]))
		return pca

	def run_pca(self, data, components):
		pca_attack = self.pca(self.data_by_pos[self.POS_ATTACK][self.skills_attack], 4)
		pca_mid = self.pca(self.data_by_pos[self.POS_MID][self.skills_mid], 4)
		pca_def = self.pca(self.data_by_pos[self.POS_DEF][self.skills_def], 4)
		pca_gk = self.pca(self.data_by_pos[self.POS_GK][self.skills_gk], 4)

	def plot_pca(self, score, coeff, labels=None):
		xs = score[:, 0]
		ys = score[:, 1]
		n = coeff.shape[0]
		scalex = 1.0 / (xs.max() - xs.min())
		scaley = 1.0 / (ys.max() - ys.min())
		plt.scatter(xs * scalex, ys * scaley)
		for i in range(n):
			plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
			if labels is None:
				plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(i + 1), color='black', ha='center',
						 va='center', fontsize=8)
			else:
				plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center')

		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		plt.xlabel("PC{}".format(1))
		plt.ylabel("PC{}".format(2))
		plt.grid()
		plt.show()

	def run_clustering(self, clusters=None):
		if not clusters and not self.clusters:
			print("No of clusters required!")
			return

		if not clusters:
			clusters = self.clusters

		self.k_means(self.skills, clusters)
		# self.k_means_plus_plus(self.skills,clusters)


clustering = Clustering("./data_normalized.csv", "Id", 7)
# clustering.run_clustering()
