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
		self.data = self.extract_data()

	def extract_data(self):
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
		print("Done Agglomerative clustering")

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
		#print(kmeanspp.labels_)
		#print(set(kmeanspp.labels_))
		return kmeanspp.labels_;




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

		#skills = self.data.loc[:, "Crossing":"GKReflexes"]
		skills = self.data[["GKReflexes","GKDiving"]]
		print("skills")
		print((skills))
		# skills1 = self.data.loc[:,"Aggression"]
		# skills2 = self.data.loc[:,"Jumping"]
		#self.agglomerative(skills,clusters)
		#self.k_means(skills,clusters)
		kpplabels = self.k_means_plus_plus(skills,clusters)
		return kpplabels

def plotGraph(cluster1,id1,cluster2,id2,cluster3,id3,cluster4,id4,list1,list2):
	print("Plotting Graph")
	for i in range(0, len(cluster1)):
		plt.scatter(cluster1[i][0], cluster1[i][1], color='yellow')
		#plt.text(cluster1[i][0], cluster1[i][1],id1)

	for i in range(0, len(cluster2)):
		plt.scatter(cluster2[i][0], cluster2[i][1], color='red')
		#plt.text(cluster2[i][0], cluster2[i][1],id2)

	for i in range(0, len(cluster3)):
		plt.scatter(cluster3[i][0], cluster3[i][1], color='green')
		#plt.text(cluster3[i][0], cluster3[i][1], id3)

	for i in range(0, len(cluster4)):
		plt.scatter(cluster4[i][0], cluster4[i][1], color='blue')
		#plt.text(cluster4[i][0], cluster4[i][1], id4)

	plt.xlabel("GKReflexes")
	plt.ylabel("GKDiving")
	print("Plotting Graph")
	plt.show()

def clusterAttr(cluster1,cluster2,cluster3,cluster4,list1,list2):
	coords = []
	for i in range(len(list1)):
		coords.append([list1[i],list2[i]])
	fc1 = []
	fc2 = []
	fc3 = []
	fc4 = []

	for i in cluster1:
		fc1.append(coords[i])
	for i in cluster2:
		fc2.append(coords[i])
	for i in cluster3:
		fc3.append(coords[i])
	for i in cluster4:
		fc4.append(coords[i])

	plotGraph(fc1, cluster1, fc2, cluster2, fc3, cluster3, fc4, cluster4,list1,list2);

#clustering = Clustering("./data_normalized.csv","Id", 4)
clustering = Clustering("E:/Courses/Semester2/Data Mining/Project/gitRepo/Data-Mining-Project/data_normalizedSkills.csv","Id", 4)
kapplabels = clustering.run_clustering()
print("KAppLables:",kapplabels)
print("Uniq Kapp lables:",set(kapplabels))

print("Here")
data = clustering.extract_data();
#print(data)
id = data.index.tolist()

#Add the player with id into the cluster
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
k = 0
for i in kapplabels:
	if(i == 0):
		cluster1.append(id[k])
	elif(i==1):
		cluster2.append(id[k])
	elif(i==2):
		cluster3.append(id[k])
	elif(i==3):
		cluster4.append(id[k])
	k = k+1;


#print(len(cluster1),len(cluster2),len(cluster3),len(cluster4))
#print(cluster1)

dribbleList = data["Dribbling"].tolist()
bcList = data["BallControl"].tolist();
#clusterAttr(cluster1,cluster2,cluster3,cluster4,dribbleList,bcList)

jmList = data["Jumping"].tolist()
agList = data["Aggression"].tolist()
#clusterAttr(cluster1,cluster2,cluster3,cluster4,jmList,agList)

strList = data["Strength"].tolist()
haList = data["HeadingAccuracy"].tolist()
#clusterAttr(cluster1,cluster2,cluster3,cluster4,strList,haList)

gkrList = data["GKReflexes"].tolist()
gkdList = data["GKDiving"].tolist()
clusterAttr(cluster1,cluster2,cluster3,cluster4,gkrList,gkdList)

wageList = data["Wage"].tolist()
OverallList = data["Overall"].tolist();



# print("OV:")
# ov1 = []
# ov2 = []
# ov3 = []
# ov4 = []
#
# for i in cluster1:
# 	ov1.append(OverallList[i])
# for i in cluster2:
# 	ov2.append(OverallList[i])
# for i in cluster3:
# 	ov3.append(OverallList[i])
# for i in cluster4:
# 	ov4.append(OverallList[i])
#
# print((ov1))

# print(len(wageList))
# print(len(OverallList))
# print(len(id))
# coords = []
# for i in range(len(wageList)):
# 	coords.append([wageList[i],OverallList[i]])
#
# fc1 = []
# fc2 = []
# fc3 = []
# fc4 = []
#
# for i in cluster1:
# 	fc1.append(coords[i])
# for i in cluster2:
# 	fc2.append(coords[i])
# for i in cluster3:
# 	fc3.append(coords[i])
# for i in cluster4:
# 	fc4.append(coords[i])
# print("Plotting Graph")
# plotGraph(fc1,cluster1,fc2,cluster2,fc3,cluster3,fc4,cluster4);
