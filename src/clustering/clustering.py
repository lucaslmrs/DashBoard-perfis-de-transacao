from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansRunner:
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