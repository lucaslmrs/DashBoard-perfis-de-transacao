from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from k_means_constrained import KMeansConstrained

class KMeansRunner:
    def __init__(self) -> None:
        pass

    def kMeansRes(self, scaled_data, k, alpha_k=0.02):
        '''
        Parameters 
        ----------
        scaled_data: matrix 
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters
        Returns 
        -------
        scaled_inertia: float
            scaled inertia value for current k           
        '''
        
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # fit k-means
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia
    
    def chooseBestKforKMeans(self, scaled_data, k_range):
        ans = []
        for k in k_range:
            scaled_inertia = self.kMeansRes(scaled_data, k)
            ans.append((k, scaled_inertia))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        return best_k, results

    def apply_kmeans(self, X, range_list=range(2, 10)):

        # Range de possíveis números de clusters
        range_n_clusters = range_list

        # Lista para armazenar os valores de silhueta
        silhouette_scores = {
            'n_clusters': [],
            'scores': []
            }

        # Calcular a pontuação de silhueta para diferentes números de clusters
        for n_clusters in tqdm(range_n_clusters):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            X[f'{n_clusters}_kmeans'] = cluster_labels
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores['n_clusters'].append(n_clusters)
            silhouette_scores['scores'].append(silhouette_avg)


        # # Plotar o gráfico de silhueta
        # plt.plot(silhouette_scores['n_clusters'], silhouette_scores['scores'], marker='o')
        # plt.xlabel('Número de Clusters')
        # plt.ylabel('Pontuação de Silhueta')
        # plt.title('Método da Silhueta para Escolha do Número de Clusters')
        # plt.show()

        df_scores = pd.DataFrame(silhouette_scores)

        return X, df_scores
    

class KMeansConstrainedRunner:
    def __init__(self) -> None:
        pass

    def apply_kmeans(self, X, range_list=range(2, 10)):

        # Range de possíveis números de clusters
        range_n_clusters = range_list

        # Lista para armazenar os valores de silhueta
        silhouette_scores = {
            'n_clusters': [],
            'scores': []
            }

        # Calcular a pontuação de silhueta para diferentes números de clusters
        for n_clusters in tqdm(range_n_clusters):
            kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=10, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            X[f'{n_clusters}_kmeans'] = cluster_labels
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores['n_clusters'].append(n_clusters)
            silhouette_scores['scores'].append(silhouette_avg)


        # # Plotar o gráfico de silhueta
        # plt.plot(silhouette_scores['n_clusters'], silhouette_scores['scores'], marker='o')
        # plt.xlabel('Número de Clusters')
        # plt.ylabel('Pontuação de Silhueta')
        # plt.title('Método da Silhueta para Escolha do Número de Clusters')
        # plt.show()

        df_scores = pd.DataFrame(silhouette_scores)

        return X, df_scores