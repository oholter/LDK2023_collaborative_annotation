from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class RequirementCluster:
    def __init__(self, data, n_clusters=10):
        self.data = data
        self.vectors = [d['vec'] for d in data]
        self.n_clusters = n_clusters

        self.model = self.kmeans()
        self.labels = self.model.labels_
        self.bins = self.sort_into_bins()

        #self.display_clusters(self.bins)

    #def kmeans(vectors, k):
    def kmeans(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.vectors)
        return kmeans


    def sort_into_bins(self):
        bins = {}
        for i, label in enumerate(self.labels):
            if label not in bins:
                bins[label] = []
            bins[label].append(self.data[i])
        return bins

    def closest(self):
        closest = []
        for i in range(self.n_clusters):
            #cluster = data[kmeans.labels_ == i]
            cluster = np.array([d['vec'] for d in self.bins[i]])
            distances = np.linalg.norm(cluster - self.model.cluster_centers_[i], axis=1)
            #closest.append(cluster[np.argmin(distances)])
            closest.append(self.bins[i][np.argmin(distances)])

        #return closest
        #return [d['req'] for d in closest]
        return [{k: r[k] for k in set(list(r.keys())) - set(['vec'])} for r in closest]

        #objects = []
        #for r in closest:
            #current_object = {k: r[k] for k in set(list(r.keys())) - set(['vec'])}
            #objects.append(current_object)
#
        #return objects




    def display_clusters(self, include_single=True):
        for label, cluster in self.bins.items():
            sents = [e['req'] for e in cluster]
            if len(sents) == 1 and not include_single:
                continue
            print("Cluster {}: {}".format(label, sents))


    def elbow(self):
        distortions = []
        inertias = []
        silhouettes = []
        mapping1 = {}
        mapping2 = {}
        mapping3 = {}
        K = range(2, 81)

        X = np.array(self.vectors)


        for k in tqdm(K):
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeanModel.inertia_)
            mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = kmeanModel.inertia_
            silhouette = silhouette_score(X, kmeanModel.labels_)
            mapping3[k] = silhouette
            silhouettes.append(silhouette)

        for key, val in mapping1.items():
                print(f'{key} : {val}')

        input()

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()


        for key, val in mapping2.items():
            print(f'{key} : {val}')

        input()

        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()

        for key, val in mapping3.items():
            print(f'{key} : {val}')

        plt.plot(K, silhouettes, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette')
        plt.title('The Silhouette method')
        plt.show()

